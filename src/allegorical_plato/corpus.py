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
        return self._utterances(topic_fallback=False)

    @property
    def topic_utterances(self) -> pl.DataFrame:
        """Return ordered topic text, falling back for works without speech markup."""
        return self._utterances(topic_fallback=True)

    def _utterances(self, *, topic_fallback: bool) -> pl.DataFrame:
        frame = self.frame
        if "segment_type" in frame.columns:
            if topic_fallback:
                speech_works = frame.filter(pl.col("segment_type") == "speech")[
                    self.work_column
                ].unique()
                frame = frame.filter(
                    (pl.col("segment_type") == "speech")
                    | (
                        (~pl.col(self.work_column).is_in(speech_works.implode()))
                        & pl.col("segment_type").is_in(["narration", "unattributed_speech"])
                    )
                )
            else:
                frame = frame.filter(pl.col("segment_type") == "speech")
        speaker = pl.col(self.speaker_column).cast(pl.String).str.strip_chars()
        if topic_fallback:
            speaker = speaker.fill_null("unattributed").replace("", "unattributed")
        columns = [
            pl.col(self.work_column).cast(pl.String).alias("work"),
            speaker.alias("speaker"),
            pl.col(self.text_column).cast(pl.String).str.strip_chars().alias("text"),
        ]
        if "text_clean" in self.frame.columns:
            columns.append(pl.col("text_clean").cast(pl.String).alias("original_text"))
        passthrough = (
            "utterance_id",
            "sequence",
            "source_path",
            "section_start",
            "section_end",
            "stephanus_markers",
        )
        columns.extend(
            pl.col(optional) for optional in passthrough if optional in self.frame.columns
        )
        if "passage_id" in self.frame.columns:
            columns.append(pl.col("passage_id").alias("source_passage_id"))
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
