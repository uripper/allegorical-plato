"""Aesthetic, publication-ready static charts for discovered topics."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "allegorical-plato-matplotlib")
)

import matplotlib
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
from matplotlib import pyplot as plt

INK = "#202A35"
MUTED = "#69737D"
PAPER = "#F6F0E4"
GRID = "#D8CDBB"
GOLD = "#C58B36"
TOPIC_COLORS = (
    "#294C60",
    "#B05C45",
    "#637A53",
    "#8C6A9D",
    "#C58B36",
    "#3E7C78",
    "#A4475B",
    "#657EAA",
    "#9A744E",
    "#536D66",
    "#7C526F",
    "#B27436",
)


def render_topic_visuals(
    topic_dir: Path,
    output_dir: Path,
    *,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 220,
) -> list[Path]:
    """Render a coordinated suite of topic charts from exported analysis tables."""
    tables = {
        name: _read_table(topic_dir / name)
        for name in ("passages", "topic_terms", "passage_topics", "cluster_topics")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_formats = tuple(dict.fromkeys(item.lower() for item in formats))
    if not selected_formats or any(item not in {"png", "svg"} for item in selected_formats):
        raise ValueError("formats must contain png, svg, or both")

    labels = _topic_labels(tables["topic_terms"])
    figures = {
        "topic_constellation": _topic_constellation(tables["passages"], labels),
        "topic_terms": _topic_term_panels(tables["topic_terms"], labels),
        "cluster_topic_map": _cluster_topic_map(tables["cluster_topics"], labels),
        "dialogue_topic_profiles": _dialogue_topic_profiles(
            tables["passages"], tables["passage_topics"], labels
        ),
    }
    written: list[Path] = []
    for name, figure in figures.items():
        for extension in selected_formats:
            path = output_dir / f"{name}.{extension}"
            figure.savefig(
                path,
                dpi=dpi if extension == "png" else None,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
            )
            written.append(path)
        plt.close(figure)
    return written


def _read_table(stem: Path) -> pl.DataFrame:
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.is_file():
        return pl.read_parquet(parquet)
    if csv.is_file():
        return pl.read_csv(csv)
    raise FileNotFoundError(f"Missing topic table: {parquet} or {csv}")


def _topic_labels(terms: pl.DataFrame, count: int = 3) -> dict[int, str]:
    content = terms.filter(pl.col("term_category") == "content_word")
    if content.is_empty():
        content = terms
    labels: dict[int, str] = {}
    for topic_frame in content.sort("rank").partition_by("topic", maintain_order=True):
        topic = int(topic_frame[0, "topic"])
        labels[topic] = " · ".join(topic_frame["term"].head(count).to_list())
    for topic in terms["topic"].unique().to_list():
        labels.setdefault(int(topic), f"Topic {topic}")
    return labels


def _figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(width, height), facecolor=PAPER)
    axis.set_facecolor(PAPER)
    return figure, axis


def _title(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", fontsize=20, fontweight="bold", color=INK, pad=28)
    axis.text(
        0,
        1.025,
        subtitle,
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )


def _topic_constellation(passages: pl.DataFrame, labels: dict[int, str]) -> plt.Figure:
    figure, axis = _figure(12, 8)
    topics = sorted(int(value) for value in passages["primary_topic"].unique().to_list())
    for topic in topics:
        frame = passages.filter(pl.col("primary_topic") == topic)
        shares = frame["topic_share"].to_numpy()
        axis.scatter(
            frame["x"].to_numpy(),
            frame["y"].to_numpy(),
            s=10 + 38 * shares**2,
            c=TOPIC_COLORS[topic % len(TOPIC_COLORS)],
            alpha=0.2 + 0.55 * shares,
            linewidths=0,
            rasterized=True,
            label=f"{topic + 1}  {labels[topic]}",
        )
        axis.text(
            float(frame["x"].median()),
            float(frame["y"].median()),
            str(topic + 1),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=PAPER,
            bbox={
                "boxstyle": "circle,pad=0.38",
                "facecolor": TOPIC_COLORS[topic % len(TOPIC_COLORS)],
                "edgecolor": PAPER,
                "linewidth": 1.5,
            },
        )
    _title(
        axis,
        "A constellation of Plato’s passages",
        "Nearby passages have similar topic mixtures · larger, darker points have a clearer primary topic",
    )
    axis.axhline(0, color=GRID, linewidth=0.6, zorder=0)
    axis.axvline(0, color=GRID, linewidth=0.6, zorder=0)
    axis.set(xticks=[], yticks=[], xlabel="", ylabel="")
    for spine in axis.spines.values():
        spine.set_visible(False)
    legend = axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        labelspacing=1.1,
        markerscale=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    figure.subplots_adjust(right=0.73)
    return figure


def _topic_term_panels(terms: pl.DataFrame, labels: dict[int, str]) -> plt.Figure:
    topics = sorted(int(value) for value in terms["topic"].unique().to_list())
    columns = min(4, len(topics))
    rows = math.ceil(len(topics) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.2 * rows + 1),
        facecolor=PAPER,
        squeeze=False,
    )
    figure.suptitle(
        "The vocabulary of each discovered topic",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.04,
        0.925,
        "Top content words and phrases by NMF weight",
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    for panel, topic in enumerate(topics):
        axis = axes.flat[panel]
        axis.set_facecolor(PAPER)
        frame = (
            terms.filter((pl.col("topic") == topic) & (pl.col("term_category") == "content_word"))
            .sort("weight", descending=True)
            .head(8)
            .sort("weight")
        )
        if frame.is_empty():
            frame = terms.filter(pl.col("topic") == topic).sort("weight", descending=True).head(8)
        axis.barh(
            frame["term"].to_list(),
            frame["weight"].to_numpy(),
            color=TOPIC_COLORS[topic % len(TOPIC_COLORS)],
            alpha=0.9,
            height=0.68,
        )
        axis.set_title(
            f"{topic + 1}  {labels[topic]}",
            loc="left",
            fontsize=10,
            fontweight="bold",
            color=INK,
            pad=10,
        )
        axis.tick_params(axis="y", colors=INK, labelsize=9, length=0)
        axis.tick_params(axis="x", colors=MUTED, labelsize=7, length=0)
        axis.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in axes.flat[len(topics) :]:
        axis.set_visible(False)
    figure.subplots_adjust(top=0.84, hspace=0.55, wspace=0.65)
    return figure


def _cluster_topic_map(cluster_topics: pl.DataFrame, labels: dict[int, str]) -> plt.Figure:
    topics = sorted(labels)
    clusters = sorted(int(value) for value in cluster_topics["cluster"].unique().to_list())
    lookup = {
        (int(cluster), int(topic)): float(share)
        for cluster, topic, share in cluster_topics.select(
            "cluster", "topic", "mean_share"
        ).iter_rows()
    }
    values = np.asarray(
        [[lookup.get((cluster, topic), 0) for topic in topics] for cluster in clusters]
    )
    figure, axis = _figure(max(10, len(topics) * 1.25), max(6, len(clusters) * 0.75))
    cmap = LinearSegmentedColormap.from_list("plato", [PAPER, "#D4B477", GOLD, "#70452B"])
    image = axis.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=max(0.5, values.max()))
    for row in range(len(clusters)):
        for column in range(len(topics)):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color=PAPER if value > 0.32 else INK,
            )
    _title(
        axis,
        "How passage clusters combine the topics",
        "Each cell is the mean share of a topic among passages assigned to that cluster",
    )
    axis.set_xticks(range(len(topics)), [f"{topic + 1}\n{labels[topic]}" for topic in topics])
    axis.set_yticks(range(len(clusters)), [f"Cluster {cluster + 1}" for cluster in clusters])
    axis.tick_params(axis="x", labelsize=8, colors=INK, length=0, pad=10)
    axis.tick_params(axis="y", labelsize=9, colors=INK, length=0)
    for label in axis.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.03)
    colorbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    colorbar.outline.set_visible(False)
    return figure


def _dialogue_topic_profiles(
    passages: pl.DataFrame,
    passage_topics: pl.DataFrame,
    labels: dict[int, str],
    max_dialogues: int = 24,
) -> plt.Figure:
    largest = (
        passages.group_by("work")
        .agg(pl.col("word_count").sum().alias("words"))
        .sort("words", descending=True)
        .head(max_dialogues)
    )
    selected = largest["work"].to_list()
    profiles = (
        passage_topics.join(passages.select("passage_id", "work", "word_count"), on="passage_id")
        .filter(pl.col("work").is_in(selected))
        .with_columns((pl.col("share") * pl.col("word_count")).alias("weighted_share"))
        .group_by("work", "topic")
        .agg((pl.col("weighted_share").sum() / pl.col("word_count").sum()).alias("mean_share"))
    )
    topics = sorted(labels)
    lookup = {(str(work), int(topic)): float(share) for work, topic, share in profiles.iter_rows()}
    works = largest["work"].to_list()[::-1]
    figure, axis = _figure(13, max(7, len(works) * 0.38))
    left = np.zeros(len(works))
    for topic in topics:
        values = np.asarray([lookup.get((work, topic), 0) for work in works])
        axis.barh(
            [_short_work(work) for work in works],
            values,
            left=left,
            color=TOPIC_COLORS[topic % len(TOPIC_COLORS)],
            height=0.72,
            label=f"{topic + 1}  {labels[topic]}",
        )
        left += values
    _title(
        axis,
        "Topic fingerprints across the largest dialogues",
        "Word-weighted topic mixtures · the original work identifier is shortened for display",
    )
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axis.tick_params(axis="both", colors=INK, labelsize=8, length=0)
    axis.grid(axis="x", color=PAPER, linewidth=1, alpha=0.8)
    axis.set_axisbelow(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    return figure


def _short_work(work: str) -> str:
    return work.rsplit(":", maxsplit=1)[-1].replace(".perseus-eng", "")
