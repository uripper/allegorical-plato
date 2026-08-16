"""Static dialogue-structure visualizations."""

from __future__ import annotations

import html
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from allegorical_plato.structure import jensen_shannon_divergence, windowed_topic_vectors
from allegorical_plato.viz.static import (
    GOLD,
    GRID,
    INK,
    MUTED,
    PAPER,
    TOPIC_COLORS,
    _read_table,
    _short_work,
    _topic_labels,
)


def render_dialogue_structure(
    topic_dir: Path,
    structure_dir: Path,
    output_dir: Path,
    *,
    work: str | None = None,
    work_titles: dict[str, str] | None = None,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 220,
) -> list[Path]:
    """Render one dialogue's trajectory, surprise, anomaly, and symmetry matrix."""
    passages = _read_table(topic_dir / "passages")
    passage_topics = _read_table(topic_dir / "passage_topics")
    topic_terms = _read_table(topic_dir / "topic_terms")
    passage_metrics = _read_table(structure_dir / "passage_metrics")
    transitions = _read_table(structure_dir / "transitions")
    dialogue_symmetry = _read_table(structure_dir / "dialogue_symmetry")

    selected_work = _select_work(passages, work)
    figure = _dialogue_structure_figure(
        passages,
        passage_topics,
        passage_metrics,
        transitions,
        dialogue_symmetry,
        _topic_labels(topic_terms),
        selected_work,
        title=(work_titles or {}).get(selected_work),
    )
    selected_formats = tuple(dict.fromkeys(item.lower() for item in formats))
    if not selected_formats or any(item not in {"png", "svg"} for item in selected_formats):
        raise ValueError("formats must contain png, svg, or both")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dialogue_structure_{_short_work(selected_work)}"
    written: list[Path] = []
    for extension in selected_formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(
            path,
            dpi=dpi if extension == "png" else None,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        written.append(path)
    plt.close(figure)
    return written


def render_dialogue_structures(
    topic_dir: Path,
    structure_dir: Path,
    output_dir: Path,
    *,
    work_titles: dict[str, str] | None = None,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 220,
) -> list[Path]:
    """Render every modeled dialogue and write a passage-oriented HTML index."""
    passages = _read_table(topic_dir / "passages")
    passage_topics = _read_table(topic_dir / "passage_topics")
    topic_terms = _read_table(topic_dir / "topic_terms")
    passage_metrics = _read_table(structure_dir / "passage_metrics")
    transitions = _read_table(structure_dir / "transitions")
    dialogue_symmetry = _read_table(structure_dir / "dialogue_symmetry")
    selected_formats = tuple(dict.fromkeys(item.lower() for item in formats))
    if not selected_formats or any(item not in {"png", "svg"} for item in selected_formats):
        raise ValueError("formats must contain png, svg, or both")

    titles = work_titles or {}
    works = sorted(
        (str(value) for value in passages["work"].unique().to_list()),
        key=lambda work: (titles.get(work, _short_work(work)).casefold(), work),
    )
    labels = _topic_labels(topic_terms)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for work in works:
        figure = _dialogue_structure_figure(
            passages,
            passage_topics,
            passage_metrics,
            transitions,
            dialogue_symmetry,
            labels,
            work,
            title=titles.get(work),
        )
        stem = f"dialogue_structure_{_short_work(work)}"
        for extension in selected_formats:
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(
                path,
                dpi=dpi if extension == "png" else None,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
            )
            written.append(path)
        plt.close(figure)

    index = output_dir / "dialogue_structure_index.html"
    index.write_text(
        _structure_index_html(
            works,
            titles,
            passage_metrics,
            transitions,
            dialogue_symmetry,
            image_extension="png" if "png" in selected_formats else "svg",
        ),
        encoding="utf-8",
    )
    written.append(index)
    return written


def _select_work(passages: pl.DataFrame, requested: str | None) -> str:
    works = [str(value) for value in passages["work"].unique().to_list()]
    if requested is not None:
        matches = [work for work in works if work == requested or work.endswith(requested)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"No dialogue matches {requested!r}")
        raise ValueError(f"Dialogue selector {requested!r} matches more than one work")
    return str(
        passages.group_by("work")
        .agg(pl.col("word_count").sum().alias("words"))
        .sort("words", descending=True)[0, "work"]
    )


def _dialogue_topic_matrix(
    passages: pl.DataFrame, passage_topics: pl.DataFrame, work: str
) -> tuple[pl.DataFrame, list[int], np.ndarray]:
    frame = passages.filter(pl.col("work") == work)
    order_column = "passage_number" if "passage_number" in frame.columns else "sequence_start"
    frame = frame.sort(order_column, maintain_order=True)
    topics = sorted(int(value) for value in passage_topics["topic"].unique().to_list())
    lookup = {
        (str(passage_id), int(topic)): float(share)
        for passage_id, topic, share in passage_topics.select(
            "passage_id", "topic", "share"
        ).iter_rows()
    }
    matrix = np.asarray(
        [
            [lookup[(str(passage_id), topic)] for topic in topics]
            for passage_id in frame["passage_id"]
        ]
    )
    return frame, topics, matrix


def _dialogue_structure_figure(
    passages: pl.DataFrame,
    passage_topics: pl.DataFrame,
    passage_metrics: pl.DataFrame,
    transitions: pl.DataFrame,
    dialogue_symmetry: pl.DataFrame,
    labels: dict[int, str],
    work: str,
    *,
    title: str | None = None,
) -> plt.Figure:
    _, topics, matrix = _dialogue_topic_matrix(passages, passage_topics, work)
    metrics = passage_metrics.filter(pl.col("work") == work).sort("passage_number")
    boundaries = transitions.filter(pl.col("work") == work).sort("boundary_position")
    summary = dialogue_symmetry.filter(pl.col("work") == work).row(0, named=True)
    positions = metrics["normalized_position"].to_numpy()

    figure = plt.figure(figsize=(15, 14), facecolor=PAPER)
    grid = figure.add_gridspec(3, 1, height_ratios=(1.6, 0.8, 2.2), hspace=0.38)
    trajectory_axis = figure.add_subplot(grid[0])
    metric_axis = figure.add_subplot(grid[1], sharex=trajectory_axis)
    symmetry_axis = figure.add_subplot(grid[2])
    for axis in (trajectory_axis, metric_axis, symmetry_axis):
        axis.set_facecolor(PAPER)

    figure.suptitle(
        f"The conceptual trajectory of {title or _short_work(work)}",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.957,
        (
            f"{_short_work(work)} · raw passage-level NMF mixtures · "
            "normalized position is descriptive, not symbolic"
        ),
        ha="left",
        color=MUTED,
        fontsize=10,
    )

    for topic_index, topic in enumerate(topics):
        trajectory_axis.plot(
            positions,
            matrix[:, topic_index],
            color=TOPIC_COLORS[topic % len(TOPIC_COLORS)],
            linewidth=1.25,
            alpha=0.88,
            label=f"{topic + 1}  {labels.get(topic, f'Topic {topic + 1}')}",
        )
    trajectory_axis.set_ylabel("Topic share", color=INK, fontsize=9)
    trajectory_axis.set_ylim(0, max(0.65, float(matrix.max()) * 1.05))
    trajectory_axis.set_title(
        "Complete topic mixture through the dialogue",
        loc="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    trajectory_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    metric_axis.fill_between(
        boundaries["boundary_position"].to_numpy(),
        boundaries["transition_score"].to_numpy(),
        color=GOLD,
        alpha=0.25,
        linewidth=0,
    )
    metric_axis.plot(
        boundaries["boundary_position"].to_numpy(),
        boundaries["transition_score"].to_numpy(),
        color=GOLD,
        linewidth=1.5,
        label="Adjacent transition surprise",
    )
    metric_axis.plot(
        positions,
        metrics["local_anomaly_score"].to_numpy(),
        color=INK,
        linewidth=1.1,
        alpha=0.8,
        label="Local anomaly",
    )
    strongest = boundaries.sort("transition_score", descending=True).head(5)
    metric_axis.scatter(
        strongest["boundary_position"].to_numpy(),
        strongest["transition_score"].to_numpy(),
        color=GOLD,
        edgecolor=PAPER,
        linewidth=1,
        s=35,
        zorder=5,
    )
    metric_axis.set_ylabel("JS divergence", color=INK, fontsize=9)
    metric_axis.set_title(
        "Conceptual discontinuity",
        loc="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    metric_axis.legend(loc="upper right", frameon=False, fontsize=8)
    metric_axis.set_xlim(0, 1)
    metric_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    metric_axis.set_xlabel("Normalized dialogue position", color=INK, fontsize=9)

    smoothed = windowed_topic_vectors(matrix, window_size=int(summary["window_size"]))
    similarities = np.asarray(
        [1.0 - jensen_shannon_divergence(vector, smoothed) for vector in smoothed]
    )
    cmap = LinearSegmentedColormap.from_list("structural_similarity", ["#6E4936", GOLD, PAPER])
    image = symmetry_axis.imshow(
        similarities,
        origin="lower",
        extent=(0, 100, 0, 100),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        rasterized=True,
    )
    symmetry_axis.plot([0, 100], [100, 0], color=INK, linewidth=0.9, alpha=0.65)
    symmetry_axis.set_title(
        "Mirrored semantic similarity",
        loc="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=24,
    )
    symmetry_axis.text(
        0,
        1.015,
        (
            f"{summary['window_size']}-passage centered means · observed {summary['symmetry_score']:.3f} "
            f"vs block-null {summary['null_mean']:.3f} · Δ {summary['observed_minus_null']:+.3f} "
            f"· p={summary['p_value_one_sided']:.3f} · BH q={summary['q_value_bh']:.3f}"
        ),
        transform=symmetry_axis.transAxes,
        color=MUTED,
        fontsize=9,
        va="bottom",
    )
    symmetry_axis.set_xlabel("Position of comparison passage", color=INK, fontsize=9)
    symmetry_axis.set_ylabel("Position of focal passage", color=INK, fontsize=9)
    symmetry_axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    symmetry_axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    colorbar = figure.colorbar(image, ax=symmetry_axis, fraction=0.025, pad=0.025)
    colorbar.set_label("1 − JS divergence", color=MUTED, fontsize=8)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    for axis in (trajectory_axis, metric_axis):
        axis.grid(axis="both", color=GRID, linewidth=0.55, alpha=0.7)
        axis.tick_params(colors=MUTED, labelsize=8, length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    symmetry_axis.tick_params(colors=MUTED, labelsize=8, length=0)
    for spine in symmetry_axis.spines.values():
        spine.set_visible(False)
    figure.subplots_adjust(top=0.91, bottom=0.07, left=0.08, right=0.92)
    return figure


def _structure_index_html(
    works: list[str],
    titles: dict[str, str],
    passage_metrics: pl.DataFrame,
    transitions: pl.DataFrame,
    dialogue_symmetry: pl.DataFrame,
    *,
    image_extension: str,
) -> str:
    cards: list[str] = []
    for work in works:
        title = html.escape(titles.get(work, _short_work(work)))
        identifier = html.escape(_short_work(work))
        summary = dialogue_symmetry.filter(pl.col("work") == work).row(0, named=True)
        boundaries = transitions.filter(pl.col("work") == work).sort(
            "transition_score", descending=True
        )
        metrics = passage_metrics.filter(pl.col("work") == work)
        if "has_full_local_neighborhood" in metrics.columns:
            internal = metrics.filter(pl.col("has_full_local_neighborhood"))
            if not internal.is_empty():
                metrics = internal
        anomalies = metrics.sort("local_anomaly_score", descending=True)
        transition_items = "".join(
            (
                f"<li><b>{html.escape(str(row['boundary_after_reference']))} → "
                f"{html.escape(str(row['boundary_before_reference']))}</b> "
                f"<span>JS {row['transition_score']:.3f}</span></li>"
            )
            for row in boundaries.head(3).iter_rows(named=True)
        )
        anomaly_items = "".join(
            (
                f"<li><b>{html.escape(str(row['reference']))}</b> "
                f"<span>local JS {row['local_anomaly_score']:.3f}</span><br>"
                f"{html.escape(str(row['text_preview']))}</li>"
            )
            for row in anomalies.head(3).iter_rows(named=True)
        )
        filename = f"dialogue_structure_{_short_work(work)}.{image_extension}"
        cards.append(
            f"""
            <article>
              <h2>{title} <code>{identifier}</code></h2>
              <p class="stats">Mirrored similarity {summary["symmetry_score"]:.3f};
                block-null {summary["null_mean"]:.3f};
                Δ {summary["observed_minus_null"]:+.3f};
                p {summary["p_value_one_sided"]:.3f}; q {summary["q_value_bh"]:.3f}.</p>
              <div class="candidates">
                <section><h3>Strongest transitions</h3><ol>{transition_items}</ol></section>
                <section><h3>Strongest internal anomalies</h3><ol>{anomaly_items}</ol></section>
              </div>
              <a href="{html.escape(filename)}"><img loading="lazy" src="{html.escape(filename)}"
                alt="Structural topic profile for {title}"></a>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plato dialogue structural candidates</title>
<style>
:root {{ color-scheme: light; --paper:#F6F0E4; --ink:#202A35; --muted:#69737D;
  --grid:#D8CDBB; --gold:#C58B36; }}
* {{ box-sizing:border-box; }}
body {{ margin:0 auto; max-width:1200px; padding:3rem 2rem; background:var(--paper);
  color:var(--ink); font:16px/1.5 system-ui,sans-serif; }}
h1 {{ margin-bottom:.4rem; font-size:2.4rem; }}
.intro {{ max-width:850px; color:var(--muted); }}
.tables a {{ color:#76521e; margin-right:1rem; }}
article {{ margin:3rem 0; padding-top:2rem; border-top:1px solid var(--grid); }}
h2 code {{ color:var(--muted); font-size:.75rem; font-weight:400; }}
.stats {{ color:var(--muted); }}
.candidates {{ display:grid; grid-template-columns:1fr 1fr; gap:2rem; }}
li {{ margin:.55rem 0; }} li span {{ color:var(--muted); }}
img {{ width:100%; height:auto; margin-top:1rem; border:1px solid var(--grid); }}
@media (max-width:700px) {{ .candidates {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>Dialogue structural candidates</h1>
<p class="intro">This is a reading index, not evidence of an encoded doctrine. It identifies where
the topic model sees abrupt conceptual changes, locally unusual passages, and possible mirrored
similarity. Use the linked references to return to the text. Symmetry claims should be ignored when
the block-null q-value is not small.</p>
<p class="tables"><a href="../structure/passage_metrics.csv">All ranked passages</a>
<a href="../structure/transitions.csv">All boundaries</a>
<a href="../structure/dialogue_symmetry.csv">All symmetry tests</a>
<a href="../structure/trajectory_matches.csv">Recurring trajectories</a></p>
{"".join(cards)}
</body>
</html>
"""
