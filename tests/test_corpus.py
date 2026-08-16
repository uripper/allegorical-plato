from pathlib import Path

import polars as pl

from allegorical_plato.corpus import load_corpus, profile
from allegorical_plato.features import contrast_terms, distinctive_terms, group_socrates
from allegorical_plato.preprocessing import classify_term, tokenize, topic_features
from allegorical_plato.topics import analyze_topics, build_passages


def test_load_and_profile(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "speaker": ["Socrates", "Socrates", "Glaucon"],
            "utterance": ["Know thyself", "The good", "Yes"],
            "language": ["eng"] * 3,
            "work_id": ["dialogue"] * 3,
        }
    ).write_parquet(path)

    corpus = load_corpus(path, language="eng")
    result = profile(corpus)

    assert corpus.text_column == "utterance"
    assert result.filter(pl.col("speaker") == "Socrates")["utterances"].item() == 2


def test_detects_project_schema(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "speaker_label": ["Socrates"],
            "text_normalized": ["Know thyself"],
            "language": ["eng"],
            "work_id": ["dialogue"],
        }
    ).write_parquet(path)

    corpus = load_corpus(path, language="eng")

    assert corpus.speaker_column == "speaker_label"
    assert corpus.text_column == "text_normalized"


def test_prefers_canonical_clean_speech_schema(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "work_id": ["dialogue"] * 3,
            "language": ["eng"] * 3,
            "segment_type": ["speech", "narration", "heading"],
            "speaker_id": ["socrates", None, None],
            "speaker_local_id": ["Socrates", None, None],
            "text_clean": ["Clean speech", "Narrator", "Title"],
            "text_normalized": ["Contaminated speech", "Narrator", "Title"],
        }
    ).write_parquet(path)

    corpus = load_corpus(path, language="eng")

    assert corpus.is_canonical
    assert corpus.speaker_column == "speaker_id"
    assert corpus.text_column == "text_clean"
    assert corpus.utterances.select("speaker", "text").rows() == [("socrates", "Clean speech")]


def test_prefers_analysis_text_without_changing_canonical_text(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "work_id": ["dialogue"],
            "language": ["grc"],
            "segment_type": ["speech"],
            "speaker_id": ["socrates"],
            "text_clean": ["τοίνυν σώματος"],
            "text_topic": ["σῶμα"],
        }
    ).write_parquet(path)

    corpus = load_corpus(path, language="grc")

    assert corpus.text_column == "text_topic"
    assert corpus.frame["text_clean"].item() == "τοίνυν σώματος"
    assert corpus.utterances["text"].item() == "σῶμα"


def test_distinctive_terms() -> None:
    frame = pl.DataFrame(
        {
            "speaker": ["A"] * 3 + ["B"] * 3,
            "text": ["number harmony"] * 3 + ["justice city"] * 3,
        }
    )

    result = distinctive_terms(frame, top_n=3, min_utterances=2, language="eng")

    assert "number" in result.filter(pl.col("group") == "A")["term"].to_list()
    assert "justice" in result.filter(pl.col("group") == "B")["term"].to_list()


def test_language_slices_never_mix(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "speaker": ["Socrates", "ΣΩ."],
            "utterance": ["The good", "τὸ ἀγαθόν"],
            "language": ["eng", "grc"],
            "work_id": ["dialogue", "dialogue"],
        }
    ).write_parquet(path)

    corpus = load_corpus(path, language="grc")

    assert corpus.frame["language"].unique().to_list() == ["grc"]
    assert corpus.utterances["text"].to_list() == ["τὸ ἀγαθόν"]


def test_groups_only_exact_socrates_labels() -> None:
    utterances = pl.DataFrame(
        {
            "speaker": ["Soc.", "Socrates.", "Alcibiades’ praise of Socrates", "Glaucon."],
            "text": ["one", "two", "three", "four"],
            "work": ["dialogue"] * 4,
        }
    )

    result = group_socrates(utterances, language="eng")

    assert result["speaker"].to_list() == [
        "socrates",
        "socrates",
        "non_socrates",
        "non_socrates",
    ]


def test_contrast_terms_ranks_group_differences() -> None:
    utterances = pl.DataFrame(
        {
            "speaker": (["socrates"] * 2 + ["non_socrates"] * 2) * 3,
            "text": (["shared dialectic"] * 2 + ["shared rhetoric"] * 2) * 3,
            "work": ["one"] * 4 + ["two"] * 4 + ["three"] * 4,
        }
    )

    result = contrast_terms(utterances, language="eng", top_n=1, min_utterances=1, min_dialogues=3)

    assert any(
        "dialectic" in term
        for term in result.filter(pl.col("group") == "socrates")["term"].to_list()
    )
    assert any(
        "rhetoric" in term
        for term in result.filter(pl.col("group") == "non_socrates")["term"].to_list()
    )
    assert result["dialogues_observed"].min() == 3
    assert result["loo_sign_stability"].min() == 1.0


def test_ngrams_do_not_cross_utterance_boundaries() -> None:
    utterances = pl.DataFrame(
        {
            "speaker": ["A", "A", "B", "B"],
            "text": ["alpha ending", "starting beta", "gamma ending", "starting delta"],
            "work": ["dialogue"] * 4,
        }
    )

    result = distinctive_terms(utterances, language="eng", top_n=20, min_utterances=2)

    assert "ending starting" not in result["term"].to_list()


def test_unmatched_dialogues_do_not_enter_contrast() -> None:
    utterances = pl.DataFrame(
        {
            "speaker": (["socrates", "non_socrates"] * 3) + ["non_socrates"] * 3,
            "text": (["dialectic", "rhetoric"] * 3) + ["lawcode"] * 3,
            "work": ["one"] * 2 + ["two"] * 2 + ["three"] * 2 + ["laws"] * 3,
        }
    )

    result = contrast_terms(utterances, language="eng", top_n=10, min_utterances=1, min_dialogues=3)

    assert "lawcode" not in result["term"].to_list()
    assert result["eligible_dialogues"].unique().to_list() == [3]


def test_lexical_categories_and_editorial_filtering() -> None:
    proper_names = {"socrates"}

    assert classify_term("the", language="eng", proper_name_tokens=proper_names) == "function_word"
    assert (
        classify_term("Socrates", language="eng", proper_name_tokens=proper_names) == "proper_name"
    )
    assert (
        classify_term("certainly", language="eng", proper_name_tokens=proper_names)
        == "dialogue_formula"
    )
    assert (
        classify_term("plat laws", language="eng", proper_name_tokens=proper_names)
        == "editorial_artifact"
    )
    assert tokenize("Cp. Plat. Laws") == ["laws"]


def test_greek_topic_features_match_folded_stopwords():
    features = topic_features(
        "τοίνυν εἶπον πάλιν ναί οὐκοῦν ἄρα γάρ μέν δέ σῶμα",
        language="grc",
    )

    assert features == ["σῶμα"]


def test_topic_bigrams_do_not_cross_utterance_boundaries_or_repeat():
    features = topic_features("beautiful\nbeautiful things", language="eng")

    assert "beautiful beautiful" not in features
    assert "beautiful things" in features


def test_build_passages_preserves_dialogue_boundaries_and_sequence() -> None:
    utterances = pl.DataFrame(
        {
            "work": ["one", "one", "one", "two"],
            "speaker": ["B", "A", "A", "C"],
            "text": ["third ending", "first beginning", "second middle", "separate work"],
            "sequence": [3, 1, 2, 1],
        }
    )

    passages = build_passages(utterances, target_words=4)

    assert passages["passage_id"].to_list() == ["one__0001", "one__0002", "two__0001"]
    assert passages[0, "text"] == "first beginning second middle"
    assert passages[0, "speaker"] == "A"
    assert passages[1, "sequence_start"] == 3
    assert passages[2, "work"] == "two"


def test_topic_analysis_produces_tidy_visualization_tables() -> None:
    passages = pl.DataFrame(
        {
            "passage_id": [f"work__{index:04d}" for index in range(1, 7)],
            "work": ["work"] * 6,
            "passage_number": list(range(1, 7)),
            "speaker": ["A", "A", "A", "B", "B", "B"],
            "dominant_speaker": ["A", "A", "A", "B", "B", "B"],
            "dominant_speaker_share": [1.0] * 6,
            "speakers": ["A", "A", "A", "B", "B", "B"],
            "text": [
                "number harmony ratio music",
                "number ratio measure harmony",
                "music harmony number measure",
                "justice city law guardian",
                "justice law ruler city",
                "guardian city justice ruler",
            ],
            "word_count": [4] * 6,
            "utterance_count": [1] * 6,
        }
    )

    result = analyze_topics(
        passages,
        language="eng",
        n_topics=2,
        n_clusters=2,
        terms_per_topic=3,
        min_df=1,
    )

    assert result.passages.height == 6
    assert {"primary_topic", "topic_share", "cluster", "x", "y"}.issubset(result.passages.columns)
    assert result.topic_terms.height == 6
    assert "term_category" in result.topic_terms.columns
    assert result.passage_topics.height == 12
    assert result.cluster_topics["cluster"].n_unique() == 2
    shares = result.passage_topics.group_by("passage_id").agg(pl.col("share").sum())
    assert all(abs(value - 1.0) < 1e-9 for value in shares["share"])
