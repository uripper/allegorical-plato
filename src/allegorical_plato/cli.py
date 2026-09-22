"""Command-line interface for corpus exploration."""

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from allegorical_plato.corpus import load_corpus, profile
from allegorical_plato.features import contrast_terms, distinctive_terms, group_socrates
from allegorical_plato.structure import StructuralAnalysis, analyze_structure
from allegorical_plato.topics import analyze_topics, build_passages
from allegorical_plato.viz import (
    render_dialogue_structure,
    render_dialogue_structures,
    render_topic_visuals,
)

app = typer.Typer(no_args_is_help=True, help="Explore linguistic patterns in Plato's corpus.")
DEFAULT_DATA_PATH = Path(os.environ.get("ALLEGORICAL_PLATO_DATA", "data/corpus/utterances.parquet"))
DataPath = Annotated[Path, typer.Option("--data", "-d", help="Canonical utterance Parquet.")]


class Language(StrEnum):
    ENGLISH = "eng"
    GREEK = "grc"


class Comparison(StrEnum):
    SPEAKERS = "speakers"
    SOCRATES = "socrates"


def _languages(language: Language | None) -> list[Language]:
    return [language] if language else list(Language)


def _read_analysis_table(directory: Path, name: str) -> pl.DataFrame:
    for suffix, reader in ((".parquet", pl.read_parquet), (".csv", pl.read_csv)):
        path = directory / f"{name}{suffix}"
        if path.is_file():
            return reader(path)
    raise FileNotFoundError(f"Missing analysis table: {directory / name}.parquet or .csv")


def _write_structure_tables(
    result: StructuralAnalysis, structure_dir: Path, *, csv: bool = True
) -> None:
    structure_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "passage_metrics": result.passage_metrics,
        "transitions": result.transitions,
        "dialogue_symmetry": result.dialogue_symmetry,
        "symmetry_pairs": result.symmetry_pairs,
        "trajectory_matches": result.trajectory_matches,
    }
    for name, table in tables.items():
        output = structure_dir / f"{name}.parquet"
        table.write_parquet(output)
        if csv:
            table.write_csv(output.with_suffix(".csv"))


def _work_titles() -> dict[str, str]:
    works_path = DEFAULT_DATA_PATH.parent / "works.parquet"
    if not works_path.is_file():
        return {}
    works = pl.read_parquet(works_path)
    if not {"work_id", "canonical_title"}.issubset(works.columns):
        return {}
    return {
        str(work): str(title)
        for work, title in works.select("work_id", "canonical_title").iter_rows()
        if title
    }


@app.command()
def inspect(
    data: DataPath = DEFAULT_DATA_PATH,
    text_column: Annotated[str | None, typer.Option()] = None,
    speaker_column: Annotated[str | None, typer.Option()] = None,
    work_column: Annotated[str | None, typer.Option()] = None,
    language: Annotated[Language | None, typer.Option(help="Inspect only one language.")] = None,
) -> None:
    """Print separate corpus profiles for English and Ancient Greek."""
    for selected in _languages(language):
        corpus = load_corpus(
            data,
            text_column=text_column,
            speaker_column=speaker_column,
            work_column=work_column,
            language=selected.value,
        )
        typer.echo(f"\n[{selected.value}] Rows: {corpus.frame.height:,}")
        typer.echo(f"Analysis speech rows: {corpus.utterances.height:,}")
        typer.echo(f"Canonical schema: {'yes' if corpus.is_canonical else 'no'}")
        typer.echo(f"Columns: {', '.join(corpus.frame.columns)}")
        typer.echo(
            f"Using work={corpus.work_column!r}, speaker={corpus.speaker_column!r}, "
            f"text={corpus.text_column!r}\n"
        )
        typer.echo(profile(corpus))


@app.command("distinctive-terms")
def terms(
    data: DataPath = DEFAULT_DATA_PATH,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("outputs"),
    top_n: Annotated[int, typer.Option(min=1)] = 15,
    min_utterances: Annotated[int, typer.Option(min=1)] = 5,
    text_column: Annotated[str | None, typer.Option()] = None,
    speaker_column: Annotated[str | None, typer.Option()] = None,
    work_column: Annotated[str | None, typer.Option()] = None,
    language: Annotated[Language | None, typer.Option(help="Analyze only one language.")] = None,
    csv: Annotated[bool, typer.Option(help="Also write a CSV for manual browsing.")] = False,
    comparison: Annotated[
        Comparison,
        typer.Option(help="Compare every speaker or Socrates versus all other speakers."),
    ] = Comparison.SPEAKERS,
    min_dialogues: Annotated[
        int,
        typer.Option(min=2, help="Minimum dialogue coverage for a contrast term."),
    ] = 5,
    min_support_rate: Annotated[
        float,
        typer.Option(min=0.5, max=1.0, help="Minimum share of dialogues agreeing in sign."),
    ] = 0.6,
    max_work_share: Annotated[
        float,
        typer.Option(
            min=0.1,
            max=1.0,
            help="Maximum share of a group's term uses contributed by one dialogue.",
        ),
    ] = 0.6,
) -> None:
    """Write separate lexical results for English and Ancient Greek."""
    for selected in _languages(language):
        corpus = load_corpus(
            data,
            text_column=text_column,
            speaker_column=speaker_column,
            work_column=work_column,
            language=selected.value,
        )
        utterances = corpus.utterances
        if comparison is Comparison.SOCRATES:
            utterances = group_socrates(utterances, language=selected.value)
            result = contrast_terms(
                utterances,
                top_n=top_n,
                min_utterances=min_utterances,
                min_dialogues=min_dialogues,
                min_support_rate=min_support_rate,
                max_work_share_allowed=max_work_share,
                language=selected.value,
            )
        else:
            result = distinctive_terms(
                utterances,
                top_n=top_n,
                min_utterances=min_utterances,
                language=selected.value,
            )
        language_dir = output_dir / selected.value
        language_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if comparison is Comparison.SPEAKERS else "_socrates_vs_others"
        output = language_dir / f"distinctive_terms{suffix}.parquet"
        result.write_parquet(output)
        typer.echo(f"Wrote {result.height:,} {selected.value} ranked terms to {output}")
        if csv:
            csv_output = output.with_suffix(".csv")
            result.write_csv(csv_output)
            typer.echo(f"Wrote browsing copy to {csv_output}")


@app.command("discover-topics")
def topics(
    data: DataPath = DEFAULT_DATA_PATH,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("outputs"),
    language: Annotated[Language | None, typer.Option(help="Analyze only one language.")] = None,
    target_words: Annotated[
        int,
        typer.Option(min=25, help="Approximate passage size; dialogue boundaries are preserved."),
    ] = 200,
    n_topics: Annotated[int, typer.Option(min=2)] = 8,
    n_clusters: Annotated[int, typer.Option(min=2)] = 8,
    terms_per_topic: Annotated[int, typer.Option(min=1)] = 15,
    min_df: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option(help="Random seed for reproducible results.")] = 42,
    text_column: Annotated[str | None, typer.Option()] = None,
    speaker_column: Annotated[str | None, typer.Option()] = None,
    work_column: Annotated[str | None, typer.Option()] = None,
    csv: Annotated[
        bool, typer.Option(help="Also write CSV tables for websites and manual browsing.")
    ] = True,
) -> None:
    """Discover passage-level NMF topics and clusters for visualization."""
    for selected in _languages(language):
        corpus = load_corpus(
            data,
            text_column=text_column,
            speaker_column=speaker_column,
            work_column=work_column,
            language=selected.value,
        )
        passages = build_passages(corpus.topic_utterances, target_words=target_words)
        result = analyze_topics(
            passages,
            language=selected.value,
            n_topics=n_topics,
            n_clusters=n_clusters,
            terms_per_topic=terms_per_topic,
            min_df=min_df,
            random_state=seed,
        )
        language_dir = output_dir / selected.value / "topics"
        language_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "passages": result.passages,
            "topic_terms": result.topic_terms,
            "passage_topics": result.passage_topics,
            "cluster_topics": result.cluster_topics,
        }
        for name, table in tables.items():
            output = language_dir / f"{name}.parquet"
            table.write_parquet(output)
            if csv:
                table.write_csv(output.with_suffix(".csv"))
        typer.echo(
            f"Wrote {passages.height:,} {selected.value} passages, {n_topics} topics, "
            f"and {n_clusters} clusters to {language_dir}"
        )


@app.command("render-visuals")
def visuals(
    input_dir: Annotated[
        Path, typer.Option("--input-dir", "-i", help="Root containing language/topic tables.")
    ] = Path("outputs"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Root for rendered visualizations.")
    ] = Path("outputs"),
    language: Annotated[Language | None, typer.Option(help="Render only one language.")] = None,
    png: Annotated[bool, typer.Option(help="Write high-resolution PNG images.")] = True,
    svg: Annotated[bool, typer.Option(help="Write scalable SVG images.")] = True,
    dpi: Annotated[int, typer.Option(min=72, max=600)] = 220,
) -> None:
    """Render corpus topic charts and every dialogue's structural reading profile."""
    formats = [name for name, enabled in (("png", png), ("svg", svg)) if enabled]
    if not formats:
        raise typer.BadParameter("Enable at least one of --png or --svg")
    for selected in _languages(language):
        topic_dir = input_dir / selected.value / "topics"
        visual_dir = output_dir / selected.value / "visuals"
        topic_written = render_topic_visuals(topic_dir, visual_dir, formats=formats, dpi=dpi)
        passages = _read_analysis_table(topic_dir, "passages")
        structural = analyze_structure(
            passages,
            _read_analysis_table(topic_dir, "passage_topics"),
        )
        structure_dir = output_dir / selected.value / "structure"
        _write_structure_tables(structural, structure_dir)
        structure_written = render_dialogue_structures(
            topic_dir,
            structure_dir,
            visual_dir,
            work_titles=_work_titles(),
            formats=formats,
            dpi=dpi,
        )
        dialogue_count = passages["work"].n_unique()
        typer.echo(
            f"Wrote {len(topic_written)} corpus charts, structural tables, "
            f"and profiles for all {dialogue_count} {selected.value} dialogues to {visual_dir}"
        )
        typer.echo(f"Open {structure_written[-1]} to browse the candidates and source references")


@app.command("analyze-structure")
def structure_analysis(
    input_dir: Annotated[
        Path, typer.Option("--input-dir", "-i", help="Root containing language/topic tables.")
    ] = Path("outputs"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Root for structural-analysis tables.")
    ] = Path("outputs"),
    language: Annotated[Language | None, typer.Option(help="Analyze only one language.")] = None,
    neighborhood: Annotated[
        int, typer.Option(min=1, help="Passages on each side used by local anomaly.")
    ] = 3,
    symmetry_window: Annotated[
        int, typer.Option(min=1, help="Odd centered window for mirrored comparisons.")
    ] = 3,
    null_iterations: Annotated[
        int, typer.Option(min=1, help="Block-permutation samples per dialogue.")
    ] = 1_000,
    null_block_size: Annotated[
        int, typer.Option(min=1, help="Contiguous passages retained within each null block.")
    ] = 5,
    trajectory_window: Annotated[
        int, typer.Option(min=2, help="Passages in a recurring-trajectory window.")
    ] = 5,
    trajectory_min_movement: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Minimum mean adjacent JS divergence for trajectory matching.",
        ),
    ] = 0.02,
    max_trajectory_matches: Annotated[int, typer.Option(min=1)] = 100,
    seed: Annotated[int, typer.Option(help="Random seed for reproducible null models.")] = 42,
    csv: Annotated[bool, typer.Option(help="Also write CSV tables for manual browsing.")] = True,
) -> None:
    """Measure transitions, anomalies, symmetry, and recurring topic trajectories."""
    if symmetry_window % 2 == 0:
        raise typer.BadParameter("--symmetry-window must be odd")
    for selected in _languages(language):
        topic_dir = input_dir / selected.value / "topics"
        passages = _read_analysis_table(topic_dir, "passages")
        passage_topics = _read_analysis_table(topic_dir, "passage_topics")
        result = analyze_structure(
            passages,
            passage_topics,
            neighborhood_size=neighborhood,
            symmetry_window=symmetry_window,
            null_iterations=null_iterations,
            null_block_size=null_block_size,
            trajectory_window=trajectory_window,
            trajectory_min_movement=trajectory_min_movement,
            max_trajectory_matches=max_trajectory_matches,
            random_state=seed,
        )
        structure_dir = output_dir / selected.value / "structure"
        _write_structure_tables(result, structure_dir, csv=csv)
        typer.echo(
            f"Wrote {result.passage_metrics.height:,} {selected.value} passage metrics, "
            f"{result.transitions.height:,} boundaries, and "
            f"{result.trajectory_matches.height:,} cross-dialogue matches to {structure_dir}"
        )


@app.command("render-structure")
def structural_visuals(
    input_dir: Annotated[
        Path, typer.Option("--input-dir", "-i", help="Root containing analysis tables.")
    ] = Path("outputs"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Root for rendered visualizations.")
    ] = Path("outputs"),
    language: Annotated[Language | None, typer.Option(help="Render only one language.")] = None,
    work: Annotated[
        str | None,
        typer.Option(help="Exact work ID or unique suffix; defaults to the largest dialogue."),
    ] = None,
    png: Annotated[bool, typer.Option(help="Write high-resolution PNG images.")] = True,
    svg: Annotated[bool, typer.Option(help="Write scalable SVG images.")] = True,
    dpi: Annotated[int, typer.Option(min=72, max=600)] = 220,
) -> None:
    """Render a structural profile for one dialogue in each selected language."""
    formats = [name for name, enabled in (("png", png), ("svg", svg)) if enabled]
    if not formats:
        raise typer.BadParameter("Enable at least one of --png or --svg")
    for selected in _languages(language):
        language_dir = input_dir / selected.value
        visual_dir = output_dir / selected.value / "visuals"
        if work:
            written = render_dialogue_structure(
                language_dir / "topics",
                language_dir / "structure",
                visual_dir,
                work=work,
                work_titles=_work_titles(),
                formats=formats,
                dpi=dpi,
            )
        else:
            written = render_dialogue_structures(
                language_dir / "topics",
                language_dir / "structure",
                visual_dir,
                work_titles=_work_titles(),
                formats=formats,
                dpi=dpi,
            )
        typer.echo(f"Wrote {len(written)} {selected.value} structural files to {visual_dir}")


if __name__ == "__main__":
    app()
