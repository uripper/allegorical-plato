"""Corpus loading, validation, and summary statistics."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

TEXT_CANDIDATES = (
    "text_topic",
    "text_clean",
    "utterance",
    "text",
    "text_normalized",
    "content",
    "sentence",
    "dialogue",
)
SPEAKER_CANDIDATES = (
    "speaker_id",
    "speaker",
    "speaker_label",
    "speaker_name",
    "character",
    "author",
)
WORK_CANDIDATES = ("work_id", "dialogue_id", "work", "dialogue")
LANGUAGES = ("eng", "grc")


@dataclass(frozen=True)
class Corpus:
    frame: pl.DataFrame
    text_column: str
    speaker_column: str
    work_column: str
    language: str

    @property
    def is_canonical(self) -> bool:
        """Whether the corpus exposes the provenance-preserving analysis schema."""
        return {
            "text_clean",
            "speaker_id",
            "segment_type",
            "work_id",
        }.issubset(self.frame.columns)

    @property
    def utterances(self) -> pl.DataFrame:
        """Return normalized, non-empty speaker/text pairs for one language."""
        columns = [
            pl.col(self.work_column).cast(pl.String).alias("work"),
            pl.col(self.speaker_column).cast(pl.String).str.strip_chars().alias("speaker"),
            pl.col(self.text_column).cast(pl.String).str.strip_chars().alias("text"),
        ]
        columns.extend(
            pl.col(optional)
            for optional in ("utterance_id", "sequence", "source_path")
            if optional in self.frame.columns
        )
        frame = self.frame
        if "segment_type" in frame.columns:
            frame = frame.filter(pl.col("segment_type") == "speech")
        return frame.select(columns).filter(
            pl.col("work").is_not_null()
            & pl.col("speaker").is_not_null()
            & pl.col("text").is_not_null()
            & (pl.col("work") != "")
            & (pl.col("speaker") != "")
            & (pl.col("text") != "")
        )


def _detect_column(columns: list[str], candidates: tuple[str, ...], kind: str) -> str:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Could not detect the {kind} column. Available columns: {', '.join(columns)}. "
        f"Pass --{kind}-column explicitly."
    )


def load_corpus(
    path: Path,
    *,
    text_column: str | None = None,
    speaker_column: str | None = None,
    work_column: str | None = None,
    language: str,
) -> Corpus:
    """Load and validate a single-language slice of a parquet corpus."""
    if not path.is_file():
        raise FileNotFoundError(f"Corpus does not exist: {path}")
    frame = pl.read_parquet(path)
    if frame.is_empty():
        raise ValueError(f"Corpus is empty: {path}")

    columns = frame.columns
    text_column = text_column or _detect_column(columns, TEXT_CANDIDATES, "text")
    speaker_column = speaker_column or _detect_column(columns, SPEAKER_CANDIDATES, "speaker")
    work_column = work_column or _detect_column(columns, WORK_CANDIDATES, "work")
    if missing := [
        column for column in (text_column, speaker_column, work_column) if column not in columns
    ]:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if "language" not in columns:
        raise ValueError("Corpus must have a language column")
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; choose one of {', '.join(LANGUAGES)}")
    frame = frame.filter(pl.col("language") == language)
    if frame.is_empty():
        raise ValueError(f"Corpus has no rows for language {language!r}")
    return Corpus(frame, text_column, speaker_column, work_column, language)


def profile(corpus: Corpus) -> pl.DataFrame:
    """Compute utterance and word counts for each speaker."""
    return (
        corpus.utterances.with_columns(pl.col("text").str.count_matches(r"\b\w+\b").alias("words"))
        .group_by("speaker")
        .agg(pl.len().alias("utterances"), pl.col("words").sum())
        .sort("words", descending=True)
    )
