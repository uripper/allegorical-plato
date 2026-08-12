"""Command-line interface for corpus exploration."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from allegorical_plato.corpus import load_corpus, profile
from allegorical_plato.features import contrast_terms, distinctive_terms, group_socrates

app = typer.Typer(no_args_is_help=True, help="Explore linguistic patterns in Plato's corpus.")
DEFAULT_DATA_PATH = Path("data/corpus/utterances.parquet")
DataPath = Annotated[Path, typer.Option("--data", "-d", help="Canonical utterance Parquet.")]


class Language(StrEnum):
    ENGLISH = "eng"
    GREEK = "grc"


class Comparison(StrEnum):
    SPEAKERS = "speakers"
    SOCRATES = "socrates"


def _languages(language: Language | None) -> list[Language]:
    return [language] if language else list(Language)


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


if __name__ == "__main__":
    app()
