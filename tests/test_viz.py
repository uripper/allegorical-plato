from pathlib import Path

import polars as pl

from allegorical_plato.structure import analyze_structure
from allegorical_plato.viz import (
    render_dialogue_structure,
    render_dialogue_structures,
    render_topic_visuals,
)


def test_renders_static_topic_visuals(tmp_path: Path) -> None:
    topic_dir = tmp_path / "topics"
    topic_dir.mkdir()
    passages = pl.DataFrame(
        {
            "passage_id": ["one", "two", "three", "four"],
            "work": ["dialogue-a", "dialogue-a", "dialogue-b", "dialogue-b"],
            "passage_number": [1, 2, 1, 2],
            "text": ["justice", "virtue", "number", "harmony"],
            "word_count": [100, 120, 90, 110],
            "primary_topic": [0, 0, 1, 1],
            "topic_share": [0.8, 0.7, 0.75, 0.85],
            "x": [-0.5, -0.3, 0.4, 0.6],
            "y": [0.1, -0.1, 0.2, -0.2],
        }
    )
    terms = pl.DataFrame(
        {
            "topic": [0, 0, 1, 1],
            "rank": [1, 2, 1, 2],
            "term": ["justice", "virtue", "number", "harmony"],
            "term_category": ["content_word"] * 4,
            "weight": [0.8, 0.5, 0.9, 0.6],
        }
    )
    passage_topics = pl.DataFrame(
        {
            "passage_id": ["one", "one", "two", "two", "three", "three", "four", "four"],
            "topic": [0, 1] * 4,
            "share": [0.8, 0.2, 0.7, 0.3, 0.25, 0.75, 0.15, 0.85],
        }
    )
    cluster_topics = pl.DataFrame(
        {
            "cluster": [0, 0, 1, 1],
            "topic": [0, 1, 0, 1],
            "mean_share": [0.75, 0.25, 0.2, 0.8],
        }
    )
    for name, frame in {
        "passages": passages,
        "topic_terms": terms,
        "passage_topics": passage_topics,
        "cluster_topics": cluster_topics,
    }.items():
        frame.write_parquet(topic_dir / f"{name}.parquet")

    written = render_topic_visuals(topic_dir, tmp_path / "visuals", formats=("png",), dpi=72)

    assert len(written) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)

    structure = analyze_structure(
        passages,
        passage_topics,
        neighborhood_size=1,
        symmetry_window=1,
        null_iterations=10,
        null_block_size=1,
        trajectory_window=2,
        trajectory_min_movement=0.0,
        max_trajectory_matches=2,
    )
    structure_dir = tmp_path / "structure"
    structure_dir.mkdir()
    for name, frame in {
        "passage_metrics": structure.passage_metrics,
        "transitions": structure.transitions,
        "dialogue_symmetry": structure.dialogue_symmetry,
    }.items():
        frame.write_parquet(structure_dir / f"{name}.parquet")

    structural_written = render_dialogue_structure(
        topic_dir,
        structure_dir,
        tmp_path / "visuals",
        work="dialogue-a",
        formats=("png",),
        dpi=72,
    )

    assert len(structural_written) == 1
    assert structural_written[0].is_file() and structural_written[0].stat().st_size > 0

    gallery_written = render_dialogue_structures(
        topic_dir,
        structure_dir,
        tmp_path / "gallery",
        work_titles={"dialogue-a": "Alpha", "dialogue-b": "Beta"},
        formats=("png",),
        dpi=72,
    )

    assert len(gallery_written) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in gallery_written)
    index = gallery_written[-1].read_text()
    assert "Alpha" in index
    assert "Strongest transitions" in index
    assert "not evidence of an encoded doctrine" in index
