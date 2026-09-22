"""Passage-level topic discovery and visualization-ready projections."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from allegorical_plato.preprocessing import (
    classify_term,
    clean_text,
    detect_proper_name_tokens,
    tokenize,
    topic_features,
)


@dataclass(frozen=True)
class TopicAnalysis:
    """Tidy tables produced by a passage-level topic analysis."""

    passages: pl.DataFrame
    topic_terms: pl.DataFrame
    passage_topics: pl.DataFrame
    cluster_topics: pl.DataFrame


def build_passages(utterances: pl.DataFrame, *, target_words: int = 200) -> pl.DataFrame:
    """Combine consecutive utterances without crossing dialogue boundaries."""
    if target_words < 1:
        raise ValueError("target_words must be positive")
    required = {"work", "speaker", "text"}
    if missing := required - set(utterances.columns):
        raise ValueError(f"Utterances are missing columns: {', '.join(sorted(missing))}")

    indexed = utterances.with_row_index("_row")
    rows: list[dict[str, Any]] = []
    for work_frame in indexed.partition_by("work", maintain_order=True):
        if "sequence" in work_frame.columns and work_frame["sequence"].null_count() == 0:
            work_frame = work_frame.sort("sequence", maintain_order=True)
        chunk: list[dict[str, Any]] = []
        chunk_words = 0
        passage_number = 1
        for utterance in work_frame.iter_rows(named=True):
            text = clean_text(str(utterance["text"]))
            word_count = len(tokenize(text))
            if not text or word_count == 0:
                continue
            chunk.append({**utterance, "_clean_text": text, "_words": word_count})
            chunk_words += word_count
            if chunk_words >= target_words:
                rows.append(_passage_row(chunk, passage_number))
                passage_number += 1
                chunk = []
                chunk_words = 0
        if chunk:
            rows.append(_passage_row(chunk, passage_number))

    if not rows:
        raise ValueError("No non-empty passages could be constructed")
    return pl.DataFrame(rows)


def _passage_row(chunk: list[dict[str, Any]], passage_number: int) -> dict[str, Any]:
    work = str(chunk[0]["work"])
    speaker_words: dict[str, int] = {}
    for utterance in chunk:
        speaker = str(utterance["speaker"])
        speaker_words[speaker] = speaker_words.get(speaker, 0) + int(utterance["_words"])
    speakers = list(speaker_words)
    dominant_speaker = max(speaker_words, key=speaker_words.get)
    total_words = sum(speaker_words.values())
    row: dict[str, Any] = {
        "passage_id": f"{work}__{passage_number:04d}",
        "work": work,
        "passage_number": passage_number,
        "speaker": speakers[0] if len(speakers) == 1 else "mixed",
        "dominant_speaker": dominant_speaker,
        "dominant_speaker_share": speaker_words[dominant_speaker] / total_words,
        "speakers": "|".join(speakers),
        "text": " ".join(str(item["_clean_text"]) for item in chunk),
        "_utterance_texts": [str(item["_clean_text"]) for item in chunk],
        "word_count": total_words,
        "utterance_count": len(chunk),
    }
    if "original_text" in chunk[0]:
        row["original_text"] = " ".join(
            clean_text(str(item["original_text"]))
            for item in chunk
            if item["original_text"] is not None
        )
    if "sequence" in chunk[0]:
        values = [item["sequence"] for item in chunk if item["sequence"] is not None]
        row["sequence_start"] = min(values, default=None)
        row["sequence_end"] = max(values, default=None)
    if "utterance_id" in chunk[0]:
        values = [str(item["utterance_id"]) for item in chunk if item["utterance_id"]]
        row["utterance_id_start"] = values[0] if values else None
        row["utterance_id_end"] = values[-1] if values else None
        row["source_utterance_ids"] = "|".join(values)
    if "source_passage_id" in chunk[0]:
        values = list(
            dict.fromkeys(
                str(item["source_passage_id"]) for item in chunk if item["source_passage_id"]
            )
        )
        row["source_passage_ids"] = "|".join(values)
    if "section_start" in chunk[0]:
        starts = [str(item["section_start"]) for item in chunk if item["section_start"]]
        ends = [str(item["section_end"]) for item in chunk if item["section_end"]]
        row["section_start"] = starts[0] if starts else None
        row["section_end"] = ends[-1] if ends else None
    if "stephanus_markers" in chunk[0]:
        markers = list(
            dict.fromkeys(
                str(marker)
                for item in chunk
                for marker in (item["stephanus_markers"] or [])
                if marker
            )
        )
        row["stephanus_start"] = markers[0] if markers else None
        row["stephanus_end"] = markers[-1] if markers else None
        row["stephanus_markers"] = "|".join(markers)
    if "source_path" in chunk[0]:
        paths = list(
            dict.fromkeys(str(item["source_path"]) for item in chunk if item["source_path"])
        )
        row["source_path"] = "|".join(paths)
    return row


def analyze_topics(
    passages: pl.DataFrame,
    *,
    language: str,
    n_topics: int = 8,
    n_clusters: int = 8,
    terms_per_topic: int = 15,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = 42,
) -> TopicAnalysis:
    """Fit NMF topics, cluster passages, and create a two-dimensional projection."""
    if passages.height < 2:
        raise ValueError("Topic analysis requires at least two passages")
    if n_topics < 2:
        raise ValueError("n_topics must be at least two for the two-dimensional projection")
    if min(n_clusters, terms_per_topic, min_df) < 1:
        raise ValueError("Topic, cluster, term, and document-frequency counts must be positive")

    texts = passages["text"].to_list()
    has_boundaries = "_utterance_texts" in passages.columns
    analysis_documents = (
        ["\n".join(items) for items in passages["_utterance_texts"].to_list()]
        if has_boundaries
        else texts
    )
    vectorizer = TfidfVectorizer(
        analyzer=partial(topic_features, language=language),
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(analysis_documents)
    max_topics = min(matrix.shape)
    if n_topics > max_topics:
        raise ValueError(
            f"n_topics={n_topics} exceeds the available matrix dimension ({max_topics})"
        )
    if n_clusters > passages.height:
        raise ValueError(f"n_clusters={n_clusters} exceeds passage count ({passages.height})")

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=1_000,
    )
    weights = model.fit_transform(matrix)
    shares = np.divide(
        weights,
        weights.sum(axis=1, keepdims=True),
        out=np.zeros_like(weights),
        where=weights.sum(axis=1, keepdims=True) > 0,
    )
    features = vectorizer.get_feature_names_out()
    topic_terms = _topic_terms(
        model.components_,
        features,
        language,
        terms_per_topic,
        proper_names=detect_proper_name_tokens(texts),
    )

    clusters = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(
        shares
    )
    coordinates = PCA(n_components=2, random_state=random_state).fit_transform(shares)
    primary_topics = shares.argmax(axis=1)
    public_passages = passages.drop("_utterance_texts") if has_boundaries else passages
    passage_result = public_passages.with_columns(
        pl.Series("primary_topic", primary_topics),
        pl.Series("topic_share", shares.max(axis=1)),
        pl.Series("cluster", clusters),
        pl.Series("x", coordinates[:, 0]),
        pl.Series("y", coordinates[:, 1]),
    )

    passage_ids = passages["passage_id"].to_list()
    passage_topics = pl.DataFrame(
        {
            "passage_id": np.repeat(passage_ids, n_topics),
            "topic": np.tile(np.arange(n_topics), passages.height),
            "weight": weights.ravel(),
            "share": shares.ravel(),
        }
    )
    cluster_topics = (
        passage_topics.join(passage_result.select("passage_id", "cluster"), on="passage_id")
        .group_by("cluster", "topic")
        .agg(pl.col("share").mean().alias("mean_share"), pl.len().alias("passages"))
        .sort("cluster", "mean_share", descending=[False, True])
        .with_columns(
            pl.col("mean_share").rank("ordinal", descending=True).over("cluster").alias("rank")
        )
    )
    return TopicAnalysis(passage_result, topic_terms, passage_topics, cluster_topics)


def _topic_terms(
    components: np.ndarray,
    features: np.ndarray,
    language: str,
    terms_per_topic: int,
    proper_names: frozenset[str],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic, component in enumerate(components):
        best = component.argsort()[::-1][:terms_per_topic]
        rows.extend(
            {
                "language": language,
                "topic": topic,
                "rank": rank,
                "term": str(features[index]),
                "term_category": classify_term(
                    str(features[index]),
                    language=language,
                    proper_name_tokens=proper_names,
                ),
                "weight": float(component[index]),
            }
            for rank, index in enumerate(best, start=1)
        )
    return pl.DataFrame(rows)
