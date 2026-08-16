import numpy as np
import polars as pl

from allegorical_plato.structure import (
    analyze_structure,
    jensen_shannon_divergence,
    local_anomaly_scores,
    mirrored_symmetry,
    normalized_topic_entropy,
    symmetry_null_distribution,
)


def test_jensen_shannon_divergence_has_interpretable_bounds() -> None:
    identical = jensen_shannon_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    disjoint = jensen_shannon_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    assert identical == 0.0
    assert disjoint == 1.0


def test_entropy_and_local_anomaly_detect_a_mixed_outlier() -> None:
    vectors = np.array(
        [
            [0.95, 0.05],
            [0.9, 0.1],
            [0.05, 0.95],
            [0.9, 0.1],
            [0.95, 0.05],
        ]
    )

    entropy = normalized_topic_entropy(vectors)
    anomaly = local_anomaly_scores(vectors, neighborhood_size=1)

    assert 0 <= entropy.min() <= entropy.max() <= 1
    assert anomaly[2] == anomaly.max()


def test_mirrored_symmetry_and_null_are_reproducible() -> None:
    vectors = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.9, 0.1],
        ]
    )

    score, pairs = mirrored_symmetry(vectors, window_size=1)
    first = symmetry_null_distribution(
        vectors, window_size=1, iterations=30, block_size=2, random_state=7
    )
    second = symmetry_null_distribution(
        vectors, window_size=1, iterations=30, block_size=2, random_state=7
    )

    assert score == 1.0
    assert np.all(pairs == 1.0)
    assert np.array_equal(first, second)
    assert first.mean() < score


def test_structural_analysis_exposes_ranked_passages_and_cross_dialogue_windows() -> None:
    passage_ids = [f"a-{index}" for index in range(5)] + [f"b-{index}" for index in range(5)]
    passages = pl.DataFrame(
        {
            "passage_id": passage_ids,
            "work": ["dialogue-a"] * 5 + ["dialogue-b"] * 5,
            "passage_number": list(range(1, 6)) * 2,
            "text": [f"passage text {identifier}" for identifier in passage_ids],
            "word_count": [100] * 10,
            "section_start": [f"{index}a" for index in range(1, 6)] * 2,
            "section_end": [f"{index}b" for index in range(1, 6)] * 2,
        }
    )
    trajectories = np.array(
        [
            [0.9, 0.1],
            [0.65, 0.35],
            [0.2, 0.8],
            [0.65, 0.35],
            [0.9, 0.1],
        ]
        * 2
    )
    passage_topics = pl.DataFrame(
        {
            "passage_id": np.repeat(passage_ids, 2),
            "topic": np.tile([0, 1], 10),
            "share": trajectories.ravel(),
        }
    )

    result = analyze_structure(
        passages,
        passage_topics,
        neighborhood_size=1,
        symmetry_window=1,
        null_iterations=30,
        null_block_size=2,
        trajectory_window=3,
        trajectory_min_movement=0.0,
        max_trajectory_matches=5,
        random_state=9,
    )

    assert result.passage_metrics.height == 10
    assert {
        "reference",
        "topic_entropy",
        "transition_score",
        "local_anomaly_score",
        "anomaly_corpus_rank",
        "transition_corpus_percentile",
        "text_preview",
    }.issubset(result.passage_metrics.columns)
    assert result.transitions.height == 8
    assert result.transitions["corpus_rank"].min() == 1
    assert result.dialogue_symmetry["symmetry_score"].min() == 1.0
    assert {"q_value_bh", "survives_fdr_05"}.issubset(result.dialogue_symmetry.columns)
    assert result.trajectory_matches.height > 0
    assert result.trajectory_matches[0, "source_work"] != result.trajectory_matches[0, "match_work"]
