"""Dialogue-level structural metrics over complete NMF topic mixtures."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class StructuralAnalysis:
    """Tidy, inspectable tables produced by dialogue structural analysis."""

    passage_metrics: pl.DataFrame
    transitions: pl.DataFrame
    dialogue_symmetry: pl.DataFrame
    symmetry_pairs: pl.DataFrame
    trajectory_matches: pl.DataFrame


def normalize_topic_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return non-negative topic vectors normalized to sum to one."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("topic vectors must be a two-dimensional array with at least two topics")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("topic vectors must contain finite, non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every topic vector must have positive mass")
    return values / totals


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute base-2 Jensen-Shannon divergence, bounded between zero and one."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_values, right_values = np.broadcast_arrays(left_values, right_values)
    midpoint = 0.5 * (left_values + right_values)
    left_term = np.zeros_like(midpoint)
    right_term = np.zeros_like(midpoint)
    np.divide(left_values, midpoint, out=left_term, where=left_values > 0)
    np.divide(right_values, midpoint, out=right_term, where=right_values > 0)
    left_log = np.zeros_like(midpoint)
    right_log = np.zeros_like(midpoint)
    np.log2(left_term, out=left_log, where=left_values > 0)
    np.log2(right_term, out=right_log, where=right_values > 0)
    divergence = 0.5 * np.sum(
        np.where(left_values > 0, left_values * left_log, 0.0)
        + np.where(right_values > 0, right_values * right_log, 0.0),
        axis=-1,
    )
    return np.clip(divergence, 0.0, 1.0)


def normalized_topic_entropy(vectors: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy divided by its topic-count maximum."""
    values = normalize_topic_vectors(vectors)
    logs = np.zeros_like(values)
    np.log2(values, out=logs, where=values > 0)
    return -np.sum(np.where(values > 0, values * logs, 0.0), axis=1) / np.log2(values.shape[1])


def local_anomaly_scores(vectors: np.ndarray, *, neighborhood_size: int = 3) -> np.ndarray:
    """Compare each passage with the mean of up to ±k surrounding passages."""
    if neighborhood_size < 1:
        raise ValueError("neighborhood_size must be positive")
    values = normalize_topic_vectors(vectors)
    scores = np.full(values.shape[0], np.nan)
    for index in range(values.shape[0]):
        start = max(0, index - neighborhood_size)
        stop = min(values.shape[0], index + neighborhood_size + 1)
        neighbors = np.concatenate((values[start:index], values[index + 1 : stop]))
        if neighbors.size:
            scores[index] = float(jensen_shannon_divergence(values[index], neighbors.mean(axis=0)))
    return scores


def windowed_topic_vectors(vectors: np.ndarray, *, window_size: int = 3) -> np.ndarray:
    """Return centered, edge-truncated means for an odd passage window."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")
    values = normalize_topic_vectors(vectors)
    radius = window_size // 2
    return np.asarray(
        [
            values[max(0, index - radius) : min(len(values), index + radius + 1)].mean(axis=0)
            for index in range(len(values))
        ]
    )


def mirrored_symmetry(vectors: np.ndarray, *, window_size: int = 3) -> tuple[float, np.ndarray]:
    """Compare windowed topic mixtures at mirrored positions in a dialogue."""
    values = windowed_topic_vectors(vectors, window_size=window_size)
    pair_count = len(values) // 2
    if pair_count == 0:
        return float("nan"), np.asarray([], dtype=np.float64)
    similarities = 1.0 - jensen_shannon_divergence(
        values[:pair_count], values[: -pair_count - 1 : -1]
    )
    return float(similarities.mean()), similarities


def symmetry_null_distribution(
    vectors: np.ndarray,
    *,
    window_size: int = 3,
    iterations: int = 1_000,
    block_size: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Build a block-permutation null preserving vectors and within-block order."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    values = normalize_topic_vectors(vectors)
    if len(values) < 2:
        return np.full(iterations, np.nan)
    effective_block_size = min(block_size, max(1, len(values) // 2))
    blocks = [
        values[start : start + effective_block_size]
        for start in range(0, len(values), effective_block_size)
    ]
    generator = np.random.default_rng(random_state)
    scores = np.empty(iterations)
    for iteration in range(iterations):
        order = generator.permutation(len(blocks))
        permuted = np.concatenate([blocks[index] for index in order])
        scores[iteration] = mirrored_symmetry(permuted, window_size=window_size)[0]
    return scores


def _topic_lookup(passage_topics: pl.DataFrame) -> tuple[list[int], dict[str, np.ndarray]]:
    required = {"passage_id", "topic", "share"}
    if missing := required - set(passage_topics.columns):
        raise ValueError(f"Passage topics are missing columns: {', '.join(sorted(missing))}")
    topics = sorted(int(topic) for topic in passage_topics["topic"].unique().to_list())
    topic_index = {topic: index for index, topic in enumerate(topics)}
    lookup: dict[str, np.ndarray] = {}
    for frame in passage_topics.partition_by("passage_id", maintain_order=True):
        vector = np.zeros(len(topics), dtype=np.float64)
        for topic, share in frame.select("topic", "share").iter_rows():
            vector[topic_index[int(topic)]] = float(share)
        lookup[str(frame[0, "passage_id"])] = normalize_topic_vectors(vector[None, :])[0]
    return topics, lookup


def _ordered(frame: pl.DataFrame) -> pl.DataFrame:
    for column in ("passage_number", "sequence_start"):
        if column in frame.columns and frame[column].null_count() == 0:
            return frame.sort(column, maintain_order=True)
    return frame


def _positions(count: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, count) if count > 1 else np.asarray([0.5])


def _preview(value: Any, length: int = 220) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= length else f"{text[: length - 1]}…"


def _reference(row: dict[str, Any]) -> str:
    start = row.get("stephanus_start") or row.get("section_start")
    end = row.get("stephanus_end") or row.get("section_end")
    if start and end and end != start:
        return f"{start}–{end}"
    if start or end:
        return str(start or end)
    return str(row["passage_id"])


def _rank_columns(
    frame: pl.DataFrame,
    score: str,
    *,
    prefix: str,
    group: str | None = None,
) -> pl.DataFrame:
    rank = pl.col(score).rank("dense", descending=True)
    percentile = pl.col(score).rank("average") / pl.col(score).count()
    if group:
        rank = rank.over(group)
        percentile = percentile.over(group)
    return frame.with_columns(
        rank.alias(f"{prefix}_rank"),
        percentile.alias(f"{prefix}_percentile"),
    )


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    """Adjust a family of p-values while preserving their original order."""
    p_values = np.asarray(values, dtype=np.float64)
    adjusted = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not finite.size:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * len(finite) / np.arange(1, len(finite) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def _passage_and_transition_rows(
    passages: pl.DataFrame,
    lookup: dict[str, np.ndarray],
    *,
    neighborhood_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[pl.DataFrame, np.ndarray]]]:
    metric_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    dialogues: dict[str, tuple[pl.DataFrame, np.ndarray]] = {}
    for frame in passages.partition_by("work", maintain_order=True):
        frame = _ordered(frame)
        work = str(frame[0, "work"])
        vectors = np.asarray([lookup[str(identifier)] for identifier in frame["passage_id"]])
        vectors = normalize_topic_vectors(vectors)
        dialogues[work] = (frame, vectors)
        positions = _positions(len(frame))
        entropy = normalized_topic_entropy(vectors)
        anomalies = local_anomaly_scores(vectors, neighborhood_size=neighborhood_size)
        adjacent = (
            jensen_shannon_divergence(vectors[:-1], vectors[1:])
            if len(vectors) > 1
            else np.asarray([])
        )
        named_rows = list(frame.iter_rows(named=True))
        for index, (row, vector) in enumerate(zip(named_rows, vectors, strict=True)):
            incoming = float(adjacent[index - 1]) if index > 0 else None
            outgoing = float(adjacent[index]) if index < len(adjacent) else None
            available = [value for value in (incoming, outgoing) if value is not None]
            metric_row = {
                "work": work,
                "passage_id": str(row["passage_id"]),
                "reference": _reference(row),
                "passage_number": int(row.get("passage_number", index + 1)),
                "normalized_position": float(positions[index]),
                "dominant_topic": int(np.argmax(vector)),
                "dominant_topic_weight": float(vector.max()),
                "topic_entropy": float(entropy[index]),
                "incoming_transition_score": incoming,
                "outgoing_transition_score": outgoing,
                "transition_score": max(available) if available else None,
                "local_anomaly_score": (
                    float(anomalies[index]) if np.isfinite(anomalies[index]) else None
                ),
                "local_neighborhood_size": neighborhood_size,
                "local_neighbor_count": min(index, neighborhood_size)
                + min(len(frame) - index - 1, neighborhood_size),
                "has_full_local_neighborhood": (
                    index >= neighborhood_size and len(frame) - index - 1 >= neighborhood_size
                ),
                "is_dialogue_edge": index in {0, len(frame) - 1},
                "word_count": int(row.get("word_count", 0)),
                "text_preview": _preview(row.get("original_text", row.get("text", ""))),
                "transition_method": "base2_jensen_shannon_divergence",
                "anomaly_method": "js_to_mean_of_surrounding_passages",
                "entropy_method": "shannon_entropy_divided_by_log2_topic_count",
            }
            for optional in (
                "source_utterance_ids",
                "source_passage_ids",
                "utterance_id_start",
                "utterance_id_end",
                "section_start",
                "section_end",
                "stephanus_start",
                "stephanus_end",
                "source_path",
            ):
                if optional in row:
                    metric_row[optional] = row[optional]
            metric_rows.append(metric_row)
        for index, score in enumerate(adjacent):
            left = named_rows[index]
            right = named_rows[index + 1]
            transition_rows.append(
                {
                    "work": work,
                    "boundary_after_passage_id": str(left["passage_id"]),
                    "boundary_before_passage_id": str(right["passage_id"]),
                    "boundary_after_reference": _reference(left),
                    "boundary_before_reference": _reference(right),
                    "boundary_position": float((positions[index] + positions[index + 1]) / 2),
                    "transition_score": float(score),
                    "before_dominant_topic": int(np.argmax(vectors[index])),
                    "after_dominant_topic": int(np.argmax(vectors[index + 1])),
                    "before_text_preview": _preview(
                        left.get("original_text", left.get("text", ""))
                    ),
                    "after_text_preview": _preview(
                        right.get("original_text", right.get("text", ""))
                    ),
                    "method": "base2_jensen_shannon_divergence",
                }
            )
    return metric_rows, transition_rows, dialogues


def _symmetry_rows(
    dialogues: dict[str, tuple[pl.DataFrame, np.ndarray]],
    *,
    window_size: int,
    null_iterations: int,
    null_block_size: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for dialogue_number, (work, (frame, vectors)) in enumerate(dialogues.items()):
        score, similarities = mirrored_symmetry(vectors, window_size=window_size)
        null = symmetry_null_distribution(
            vectors,
            window_size=window_size,
            iterations=null_iterations,
            block_size=null_block_size,
            random_state=random_state + dialogue_number,
        )
        finite_null = null[np.isfinite(null)]
        null_mean = float(finite_null.mean()) if finite_null.size else float("nan")
        null_std = float(finite_null.std(ddof=1)) if finite_null.size > 1 else float("nan")
        effect = (score - null_mean) / null_std if null_std > 0 else float("nan")
        p_value = (
            float((1 + np.count_nonzero(finite_null >= score)) / (len(finite_null) + 1))
            if finite_null.size
            else float("nan")
        )
        percentile = (
            float((1 + np.count_nonzero(finite_null <= score)) / (len(finite_null) + 1))
            if finite_null.size
            else float("nan")
        )
        summaries.append(
            {
                "work": work,
                "passage_count": len(frame),
                "window_size": window_size,
                "symmetry_score": score,
                "null_iterations": null_iterations,
                "null_block_size": min(null_block_size, max(1, len(frame) // 2)),
                "null_mean": null_mean,
                "null_std": null_std,
                "effect_size_z": effect,
                "observed_minus_null": score - null_mean,
                "null_percentile": percentile,
                "p_value_one_sided": p_value,
                "null_method": "contiguous_block_permutation",
            }
        )
        positions = _positions(len(frame))
        named_rows = list(frame.iter_rows(named=True))
        for left_index, similarity in enumerate(similarities):
            right_index = len(frame) - left_index - 1
            left = named_rows[left_index]
            right = named_rows[right_index]
            pairs.append(
                {
                    "work": work,
                    "left_passage_id": str(left["passage_id"]),
                    "right_passage_id": str(right["passage_id"]),
                    "left_reference": _reference(left),
                    "right_reference": _reference(right),
                    "left_position": float(positions[left_index]),
                    "right_position": float(positions[right_index]),
                    "window_size": window_size,
                    "similarity": float(similarity),
                    "divergence": float(1.0 - similarity),
                }
            )
    return summaries, pairs


def recurring_trajectory_matches(
    dialogues: dict[str, tuple[pl.DataFrame, np.ndarray]],
    *,
    window_size: int = 5,
    min_movement: float = 0.02,
    max_matches: int = 100,
) -> pl.DataFrame:
    """Find high-similarity, non-flat topic trajectories in separate dialogues."""
    if window_size < 2:
        raise ValueError("trajectory window_size must be at least two")
    if not 0 <= min_movement <= 1:
        raise ValueError("min_movement must be between zero and one")
    if max_matches < 1:
        raise ValueError("max_matches must be positive")
    windows: dict[str, list[dict[str, Any]]] = {}
    for work, (frame, vectors) in dialogues.items():
        positions = _positions(len(frame))
        named_rows = list(frame.iter_rows(named=True))
        selected: list[dict[str, Any]] = []
        for start in range(len(frame) - window_size + 1):
            trajectory = vectors[start : start + window_size]
            movement = float(jensen_shannon_divergence(trajectory[:-1], trajectory[1:]).mean())
            if movement < min_movement:
                continue
            dominant_topics = np.argmax(trajectory, axis=1).tolist()
            if len(set(dominant_topics)) == 1:
                continue
            raw_signature = trajectory.ravel()
            raw_signature = raw_signature / np.linalg.norm(raw_signature)
            delta_signature = np.diff(trajectory, axis=0).ravel()
            delta_norm = np.linalg.norm(delta_signature)
            if delta_norm == 0:
                continue
            delta_signature = delta_signature / delta_norm
            selected.append(
                {
                    "start": start,
                    "end": start + window_size - 1,
                    "positions": (
                        float(positions[start]),
                        float(positions[start + window_size - 1]),
                    ),
                    "passage_ids": [
                        str(row["passage_id"]) for row in named_rows[start : start + window_size]
                    ],
                    "references": [
                        _reference(row) for row in named_rows[start : start + window_size]
                    ],
                    "dominant_topics": dominant_topics,
                    "trajectory": trajectory,
                    "movement": movement,
                    "raw_signature": raw_signature,
                    "delta_signature": delta_signature,
                }
            )
        windows[work] = selected

    heap: list[tuple[float, int, str, str, int, int]] = []
    serial = 0
    comparison_count = 0
    works = sorted(windows)
    for left_work_index, left_work in enumerate(works):
        left_windows = windows[left_work]
        if not left_windows:
            continue
        left_raw = np.asarray([window["raw_signature"] for window in left_windows])
        left_delta = np.asarray([window["delta_signature"] for window in left_windows])
        for right_work in works[left_work_index + 1 :]:
            right_windows = windows[right_work]
            if not right_windows:
                continue
            right_raw = np.asarray([window["raw_signature"] for window in right_windows])
            right_delta = np.asarray([window["delta_signature"] for window in right_windows])
            level_similarity = np.clip(left_raw @ right_raw.T, 0.0, 1.0)
            shape_similarity = np.clip((left_delta @ right_delta.T + 1.0) / 2.0, 0.0, 1.0)
            similarity = 0.5 * (level_similarity + shape_similarity)
            comparison_count += similarity.size
            take = min(max_matches, similarity.size)
            candidate_indices = np.argpartition(similarity.ravel(), -take)[-take:]
            for flat_index in candidate_indices:
                left_index, right_index = np.unravel_index(flat_index, similarity.shape)
                candidate = (
                    float(similarity[left_index, right_index]),
                    serial,
                    left_work,
                    right_work,
                    int(left_index),
                    int(right_index),
                )
                serial += 1
                if len(heap) < max_matches:
                    heapq.heappush(heap, candidate)
                elif candidate[0] > heap[0][0]:
                    heapq.heapreplace(heap, candidate)

    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(sorted(heap, reverse=True), start=1):
        similarity, _, source_work, match_work, source_index, match_index = candidate
        source = windows[source_work][source_index]
        match = windows[match_work][match_index]
        level_similarity = float(source["raw_signature"] @ match["raw_signature"])
        shape_similarity = float((source["delta_signature"] @ match["delta_signature"] + 1) / 2)
        rows.append(
            {
                "rank": rank,
                "similarity": similarity,
                "level_similarity": level_similarity,
                "shape_similarity": shape_similarity,
                "comparison_count": comparison_count,
                "corpus_percentile": (comparison_count - rank + 1) / comparison_count,
                "window_size": window_size,
                "source_work": source_work,
                "source_position_start": source["positions"][0],
                "source_position_end": source["positions"][1],
                "source_passage_ids": "|".join(source["passage_ids"]),
                "source_references": "|".join(source["references"]),
                "source_dominant_topics": ">".join(
                    str(topic) for topic in source["dominant_topics"]
                ),
                "source_topic_trajectory": json.dumps(source["trajectory"].round(6).tolist()),
                "source_movement": source["movement"],
                "match_work": match_work,
                "match_position_start": match["positions"][0],
                "match_position_end": match["positions"][1],
                "match_passage_ids": "|".join(match["passage_ids"]),
                "match_references": "|".join(match["references"]),
                "match_dominant_topics": ">".join(str(topic) for topic in match["dominant_topics"]),
                "match_topic_trajectory": json.dumps(match["trajectory"].round(6).tolist()),
                "match_movement": match["movement"],
                "method": "equal_window_level_and_first_difference_cosine",
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def analyze_structure(
    passages: pl.DataFrame,
    passage_topics: pl.DataFrame,
    *,
    neighborhood_size: int = 3,
    symmetry_window: int = 3,
    null_iterations: int = 1_000,
    null_block_size: int = 5,
    trajectory_window: int = 5,
    trajectory_min_movement: float = 0.02,
    max_trajectory_matches: int = 100,
    random_state: int = 42,
) -> StructuralAnalysis:
    """Analyze ordered dialogue trajectories without reducing them to one topic."""
    required = {"passage_id", "work", "text"}
    if missing := required - set(passages.columns):
        raise ValueError(f"Passages are missing columns: {', '.join(sorted(missing))}")
    if passages["passage_id"].n_unique() != passages.height:
        raise ValueError("passage_id values must be unique")
    _, lookup = _topic_lookup(passage_topics)
    if missing_ids := set(passages["passage_id"].to_list()) - set(lookup):
        raise ValueError(f"Missing topic mixtures for {len(missing_ids)} passages")

    metric_rows, transition_rows, dialogues = _passage_and_transition_rows(
        passages, lookup, neighborhood_size=neighborhood_size
    )
    passage_metrics = pl.DataFrame(metric_rows)
    passage_metrics = _rank_columns(
        passage_metrics, "local_anomaly_score", prefix="anomaly_dialogue", group="work"
    )
    passage_metrics = _rank_columns(passage_metrics, "local_anomaly_score", prefix="anomaly_corpus")
    passage_metrics = _rank_columns(
        passage_metrics, "transition_score", prefix="transition_dialogue", group="work"
    )
    passage_metrics = _rank_columns(passage_metrics, "transition_score", prefix="transition_corpus")

    transitions = pl.DataFrame(transition_rows)
    if not transitions.is_empty():
        transitions = _rank_columns(
            transitions, "transition_score", prefix="dialogue", group="work"
        )
        transitions = _rank_columns(transitions, "transition_score", prefix="corpus")

    symmetry_rows, pair_rows = _symmetry_rows(
        dialogues,
        window_size=symmetry_window,
        null_iterations=null_iterations,
        null_block_size=null_block_size,
        random_state=random_state,
    )
    dialogue_symmetry = pl.DataFrame(symmetry_rows)
    adjusted = _benjamini_hochberg(dialogue_symmetry["p_value_one_sided"].to_numpy())
    dialogue_symmetry = dialogue_symmetry.with_columns(
        pl.Series("q_value_bh", adjusted),
        pl.Series("survives_fdr_05", adjusted <= 0.05),
    ).sort("p_value_one_sided", nulls_last=True)
    symmetry_pairs = pl.DataFrame(pair_rows)
    trajectory_matches = recurring_trajectory_matches(
        dialogues,
        window_size=trajectory_window,
        min_movement=trajectory_min_movement,
        max_matches=max_trajectory_matches,
    )
    return StructuralAnalysis(
        passage_metrics=passage_metrics,
        transitions=transitions,
        dialogue_symmetry=dialogue_symmetry,
        symmetry_pairs=symmetry_pairs,
        trajectory_matches=trajectory_matches,
    )
