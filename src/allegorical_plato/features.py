"""Interpretable, dialogue-aware lexical feature extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from allegorical_plato.preprocessing import (
    classify_term,
    clean_text,
    detect_proper_name_tokens,
    dialogue_words,
    function_words,
    tokenize,
)

SOCRATES_LABELS = {
    "eng": frozenset({"socrates", "Soc.", "SOC.", "Socrates."}),
    "grc": frozenset({"socrates", "ΣΩ."}),
}
VISIBLE_CATEGORIES = (
    "content_word",
    "dialogue_formula",
    "proper_name",
    "function_word",
)


def group_socrates(utterances: pl.DataFrame, *, language: str) -> pl.DataFrame:
    """Group the canonical Socrates identity, with exact legacy-label support."""
    try:
        labels = SOCRATES_LABELS[language]
    except KeyError as error:
        raise ValueError(f"No Socrates label mapping for language {language!r}") from error
    grouped = utterances.with_columns(
        pl.when(pl.col("speaker").is_in(labels))
        .then(pl.lit("socrates"))
        .otherwise(pl.lit("non_socrates"))
        .alias("speaker")
    )
    if grouped.filter(pl.col("speaker") == "socrates").is_empty():
        raise ValueError(f"No Socrates utterances found for language {language!r}")
    return grouped


def _categories(
    terms: Sequence[str],
    texts: Sequence[str],
    language: str,
    matrix: Any,
) -> np.ndarray:
    non_names = function_words(language) | dialogue_words(language)
    proper_names = detect_proper_name_tokens(texts) - non_names
    categories = np.asarray(
        [classify_term(term, language=language, proper_name_tokens=proper_names) for term in terms],
        dtype=object,
    )
    # Repeated terms used overwhelmingly in short turns are response formulas,
    # even when a finite hand-built inventory does not contain their inflection.
    presence = matrix > 0
    document_frequency = np.asarray(presence.sum(axis=0)).ravel()
    short_mask = np.asarray([len(tokenize(text)) <= 8 for text in texts])
    short_frequency = np.asarray(presence[short_mask].sum(axis=0)).ravel()
    short_rate = np.divide(
        short_frequency,
        document_frequency,
        out=np.zeros(len(terms)),
        where=document_frequency > 0,
    )
    contextual_formula = (
        (categories == "content_word") & (document_frequency >= 3) & (short_rate >= 0.8)
    )
    categories[contextual_formula] = "dialogue_formula"
    return categories


def _rank_by_category(
    scores: np.ndarray,
    categories: np.ndarray,
    *,
    top_n: int,
) -> list[tuple[str, int, int]]:
    ranked: list[tuple[str, int, int]] = []
    for category in VISIBLE_CATEGORIES:
        candidates = np.flatnonzero((categories == category) & (scores > 0))
        best = candidates[np.argsort(scores[candidates])[::-1][:top_n]]
        ranked.extend((category, int(index), rank) for rank, index in enumerate(best, start=1))
    return ranked


def distinctive_terms(
    utterances: pl.DataFrame,
    *,
    top_n: int = 15,
    min_utterances: int = 5,
    language: str,
) -> pl.DataFrame:
    """Rank per-speaker terms without constructing cross-utterance n-grams."""
    counts = utterances.group_by("speaker").len(name="utterance_count")
    speakers = counts.filter(pl.col("utterance_count") >= min_utterances)["speaker"].to_list()
    selected = utterances.filter(pl.col("speaker").is_in(speakers))
    if selected.is_empty():
        raise ValueError(f"No speakers have at least {min_utterances} utterances")

    texts: Sequence[str] = [clean_text(text) for text in selected["text"].to_list()]
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    categories = _categories(terms, texts, language, matrix)
    speaker_values = selected["speaker"].to_numpy()
    count_lookup = dict(counts.select("speaker", "utterance_count").iter_rows())
    rows: list[dict[str, object]] = []
    for speaker in sorted(speakers):
        scores = np.asarray(matrix[speaker_values == speaker].mean(axis=0)).ravel()
        rows.extend(
            {
                "language": language,
                "comparison": "speakers",
                "group": speaker,
                "term_category": category,
                "term": terms[term_index],
                "score": float(scores[term_index]),
                "rank": rank,
                "utterance_count": count_lookup[speaker],
                "term_form": "surface",
                "method": "utterance_mean_tfidf",
            }
            for category, term_index, rank in _rank_by_category(scores, categories, top_n=top_n)
        )
    return pl.DataFrame(rows)


def _eligible_works(utterances: pl.DataFrame, min_utterances: int) -> list[str]:
    coverage = (
        utterances.group_by("work", "speaker")
        .len()
        .pivot(on="speaker", index="work", values="len")
        .fill_null(0)
    )
    required = {"socrates", "non_socrates"}
    if not required.issubset(coverage.columns):
        raise ValueError("Contrast requires socrates and non_socrates groups")
    return coverage.filter(
        (pl.col("socrates") >= min_utterances) & (pl.col("non_socrates") >= min_utterances)
    )["work"].to_list()


def contrast_terms(
    utterances: pl.DataFrame,
    *,
    language: str,
    top_n: int = 15,
    min_utterances: int = 5,
    min_dialogues: int = 5,
    min_support_rate: float = 0.6,
    max_work_share_allowed: float = 0.6,
) -> pl.DataFrame:
    """Contrast two groups within dialogues and aggregate their log-ratio effects."""
    works = sorted(_eligible_works(utterances, min_utterances))
    if len(works) < 2:
        raise ValueError("Contrast requires at least two dialogues containing both groups")
    selected = utterances.filter(pl.col("work").is_in(works))
    texts: Sequence[str] = [clean_text(text) for text in selected["text"].to_list()]
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    categories = _categories(terms, texts, language, matrix)
    work_values = selected["work"].to_numpy()
    group_values = selected["speaker"].to_numpy()
    groups = ("non_socrates", "socrates")
    vocabulary_size = len(terms)

    term_counts = np.zeros((len(works), len(groups), vocabulary_size), dtype=np.float64)
    token_counts = np.zeros((len(works), len(groups)), dtype=np.float64)
    for work_index, work in enumerate(works):
        for group_index, group in enumerate(groups):
            mask = (work_values == work) & (group_values == group)
            counts = np.asarray(matrix[mask].sum(axis=0)).ravel()
            term_counts[work_index, group_index] = counts
            token_counts[work_index, group_index] = counts.sum()

    # Each dialogue supplies one effect, so Laws-sized works cannot dominate the corpus.
    alpha = 0.5
    left = (term_counts[:, 0] + alpha) / (token_counts[:, 0, None] + alpha * vocabulary_size)
    right = (term_counts[:, 1] + alpha) / (token_counts[:, 1, None] + alpha * vocabulary_size)
    effects = np.log2(left / right)
    observed = term_counts.sum(axis=1) > 0
    effects = np.where(observed, effects, np.nan)
    observed_count = observed.sum(axis=0)
    effect_sum = np.nansum(effects, axis=0)
    mean_effect = np.divide(
        effect_sum,
        observed_count,
        out=np.zeros(vocabulary_size),
        where=observed_count > 0,
    )
    squared_deviation = np.where(observed, (effects - mean_effect) ** 2, 0)
    effect_std = np.sqrt(
        np.divide(
            squared_deviation.sum(axis=0),
            observed_count - 1,
            out=np.full(vocabulary_size, np.nan),
            where=observed_count > 1,
        )
    )
    effect_se = np.divide(
        effect_std,
        np.sqrt(observed_count),
        out=np.full(vocabulary_size, np.nan),
        where=observed_count > 1,
    )

    utterance_counts = dict(selected.group_by("speaker").len().iter_rows())
    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(groups):
        direction = 1 if group_index == 0 else -1
        directed_effects = effects * direction
        directed_mean = mean_effect * direction
        supporting = np.nansum(directed_effects > 0, axis=0)
        opposing = np.nansum(directed_effects < 0, axis=0)
        support_rate = np.divide(
            supporting,
            observed_count,
            out=np.zeros(vocabulary_size),
            where=observed_count > 0,
        )

        loo_support = np.zeros(vocabulary_size)
        for work_index in range(len(works)):
            remaining_count = observed_count - observed[work_index]
            remaining_sum = effect_sum - np.nan_to_num(effects[work_index])
            remaining_mean = np.divide(
                remaining_sum,
                remaining_count,
                out=np.zeros(vocabulary_size),
                where=remaining_count > 0,
            )
            loo_support += observed[work_index] & (remaining_mean * direction > 0)
        loo_stability = np.divide(
            loo_support,
            observed_count,
            out=np.zeros(vocabulary_size),
            where=observed_count > 1,
        )

        focal_counts = term_counts[:, group_index]
        focal_totals = focal_counts.sum(axis=0)
        max_work_share = np.divide(
            focal_counts.max(axis=0),
            focal_totals,
            out=np.ones(vocabulary_size),
            where=focal_totals > 0,
        )
        strongest_work_indices = np.argmax(focal_counts, axis=0)
        coverage = observed_count / len(works)
        stability_score = directed_mean * support_rate * loo_stability * np.sqrt(coverage)
        eligible = (
            (directed_mean > 0)
            & (observed_count >= min_dialogues)
            & (support_rate >= min_support_rate)
            & (max_work_share <= max_work_share_allowed)
        )

        for category in VISIBLE_CATEGORIES:
            candidates = np.flatnonzero(eligible & (categories == category))
            best = candidates[np.argsort(stability_score[candidates])[::-1][:top_n]]
            for rank, term_index in enumerate(best, start=1):
                directed_se = effect_se[term_index]
                rows.append(
                    {
                        "language": language,
                        "comparison": "socrates_vs_others",
                        "group": group,
                        "term_category": category,
                        "term": terms[term_index],
                        "stability_score": float(stability_score[term_index]),
                        "mean_log2_ratio": float(directed_mean[term_index]),
                        "ci95_low": float(directed_mean[term_index] - 1.96 * directed_se),
                        "ci95_high": float(directed_mean[term_index] + 1.96 * directed_se),
                        "rank": rank,
                        "dialogues_observed": int(observed_count[term_index]),
                        "supporting_dialogues": int(supporting[term_index]),
                        "opposing_dialogues": int(opposing[term_index]),
                        "support_rate": float(support_rate[term_index]),
                        "dialogue_coverage": float(coverage[term_index]),
                        "loo_sign_stability": float(loo_stability[term_index]),
                        "max_work_share": float(max_work_share[term_index]),
                        "strongest_work_id": works[strongest_work_indices[term_index]],
                        "eligible_dialogues": len(works),
                        "utterance_count": utterance_counts[group],
                        "term_form": "surface",
                        "method": "within_dialogue_log_ratio",
                    }
                )
    return pl.DataFrame(rows)
