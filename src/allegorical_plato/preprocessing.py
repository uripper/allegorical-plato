"""Conservative text cleanup and lexical classification."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# The source XML's notes have already been flattened into the text. These markers
# can be removed safely; deleting the prose around them would require source markup.
EDITORIAL_MARKER_RE = re.compile(
    r"(?i)(?<!\w)(?:cf|cp|plat|ff|fr|ibid|vol|pp?|ed)\.(?!\w)|"
    r"(?<!\w)(?:e\.g|i\.e)\.(?!\w)"
)
TOKEN_RE = re.compile(r"(?<!\w)[^\W\d_]+(?:[’'ʼ][^\W\d_]+)?(?!\w)", re.UNICODE)

EDITORIAL_TOKENS = frozenset(
    {
        "cf",
        "cp",
        "plat",
        "ff",
        "fr",
        "ibid",
        "vol",
        "p",
        "pp",
        "ed",
        "hom",
        "il",
        "od",
        "rep",
        "leg",
        "arist",
        "quint",
        "class",
        "phil",
    }
)

# Accent-insensitive Ancient Greek function-word inventory. This intentionally
# favors precision over exhaustive morphological coverage.
GREEK_FUNCTION_WORDS = frozenset(
    {
        "α",
        "αι",
        "αλλ",
        "αλλα",
        "αν",
        "ανα",
        "απο",
        "αρα",
        "αραγε",
        "αυ",
        "αυτος",
        "αυτοι",
        "αυτον",
        "αυτου",
        "αυτω",
        "αυτη",
        "αυτην",
        "αυτης",
        "αυτοις",
        "αυταις",
        "αυτους",
        "αυτας",
        "αυτων",
        "γαρ",
        "γε",
        "γουν",
        "δ",
        "δε",
        "δη",
        "δια",
        "διο",
        "εαν",
        "εαυτου",
        "εγω",
        "ει",
        "ειμι",
        "ειναι",
        "εις",
        "ειτε",
        "εισιν",
        "εκ",
        "εν",
        "επι",
        "εστι",
        "εστιν",
        "ετι",
        "εως",
        "η",
        "ηδη",
        "ημεις",
        "ημων",
        "ημας",
        "ημιν",
        "υμεις",
        "υμων",
        "υμιν",
        "υμας",
        "ην",
        "ινα",
        "και",
        "κατα",
        "μα",
        "με",
        "μεν",
        "μετα",
        "μη",
        "μηδε",
        "μητε",
        "μοι",
        "μου",
        "εμοι",
        "ναι",
        "νυν",
        "ο",
        "οδε",
        "οι",
        "ομως",
        "οις",
        "ον",
        "οπως",
        "ος",
        "οστις",
        "οταν",
        "οτι",
        "αλλο",
        "ου",
        "ουδε",
        "ουκ",
        "ουχ",
        "ουν",
        "ουτε",
        "ουτος",
        "ουτω",
        "μονον",
        "μεντοι",
        "παλιν",
        "πρωτον",
        "παρα",
        "πανυ",
        "περι",
        "πως",
        "προς",
        "που",
        "ποτε",
        "συ",
        "σου",
        "σοι",
        "σε",
        "σαυτου",
        "συν",
        "τε",
        "την",
        "της",
        "τη",
        "τηι",
        "τι",
        "τινα",
        "τινος",
        "τις",
        "το",
        "τοδε",
        "τον",
        "του",
        "των",
        "τω",
        "τωι",
        "τα",
        "ταδε",
        "ταυτα",
        "τας",
        "τοις",
        "ταις",
        "τοινυν",
        "τους",
        "υπερ",
        "υπο",
        "ως",
        "ω",
        "ωδε",
        "ωστε",
    }
)

ENGLISH_FUNCTION_WORDS = frozenset(ENGLISH_STOP_WORDS) | {
    "yes",
    "no",
    "nay",
    "well",
    "indeed",
    "oh",
}

ENGLISH_DIALOGUE_WORDS = frozenset(
    {
        "agree",
        "agreed",
        "answer",
        "answered",
        "answering",
        "apparently",
        "assuredly",
        "ask",
        "asked",
        "certainly",
        "course",
        "conversation",
        "conversed",
        "discussions",
        "fact",
        "feel",
        "follow",
        "hear",
        "hearing",
        "hope",
        "opinion",
        "inevitable",
        "likely",
        "mean",
        "means",
        "perhaps",
        "question",
        "questions",
        "reflecting",
        "remember",
        "quite",
        "replied",
        "reply",
        "say",
        "says",
        "seem",
        "seems",
        "speaking",
        "stop",
        "sure",
        "think",
        "understand",
        "true",
        "truly",
        "word",
        "words",
        "worth",
        "joke",
        "joking",
        "lead",
        "obvious",
        "obviously",
        "assure",
        "assured",
    }
)

GREEK_DIALOGUE_WORDS = frozenset(
    {
        "αληθη",
        "αληθεστατα",
        "αριστε",
        "αρ",
        "αρα",
        "δοκει",
        "δοκω",
        "δοκεις",
        "δηπου",
        "δητα",
        "δι",
        "εγωγε",
        "ελεγε",
        "ελεγον",
        "ειπον",
        "ειπε",
        "εοικε",
        "εοικεν",
        "εμοιγε",
        "εταιρε",
        "εφη",
        "εφην",
        "λεγει",
        "λεγε",
        "λεγεις",
        "λεγω",
        "μην",
        "μηδαμως",
        "μαλα",
        "μαλιστα",
        "οιμαι",
        "οισθα",
        "ουδαμως",
        "ουκουν",
        "αναγκη",
        "δηλον",
        "εχοιμι",
        "κομιδη",
        "κωλυει",
        "μενταν",
        "νυνδη",
        "φαινεται",
        "ποτερον",
        "πανταπασι",
        "παντως",
        "ποιον",
        "πως",
        "φιλε",
        "φερε",
        "καλεις",
        "εικος",
        "σφοδρα",
        "ορθοτατα",
        "συγχωρω",
        "φημι",
    }
)


def fold_token(token: str) -> str:
    """Lowercase a token and remove accents for dictionary lookup only."""
    decomposed = unicodedata.normalize("NFD", token.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def clean_text(text: str) -> str:
    """Normalize Unicode and remove high-confidence editorial citation markers."""
    normalized = unicodedata.normalize("NFC", text).replace("’", "'").replace("ʼ", "'")
    return re.sub(r"\s+", " ", EDITORIAL_MARKER_RE.sub(" ", normalized)).strip()


def tokenize(text: str) -> list[str]:
    """Return normalized Unicode word tokens without crossing text boundaries."""
    return [
        unicodedata.normalize("NFC", token.lower()) for token in TOKEN_RE.findall(clean_text(text))
    ]


def topic_tokens(text: str, *, language: str) -> list[str]:
    """Return topic tokens, matching stoplists through accent-folded forms."""
    stops = function_words(language) | dialogue_words(language)
    return [token for token in tokenize(text) if fold_token(token) not in stops]


def topic_features(document: str, *, language: str) -> list[str]:
    """Build unigram/bigram features without crossing newline utterance boundaries."""
    features: list[str] = []
    for utterance in document.splitlines() or [document]:
        tokens = topic_tokens(utterance, language=language)
        features.extend(tokens)
        features.extend(f"{left} {right}" for left, right in pairwise(tokens) if left != right)
    return features


def function_words(language: str) -> frozenset[str]:
    if language == "eng":
        return ENGLISH_FUNCTION_WORDS
    if language == "grc":
        return GREEK_FUNCTION_WORDS
    raise ValueError(f"No function-word inventory for language {language!r}")


def dialogue_words(language: str) -> frozenset[str]:
    if language == "eng":
        return ENGLISH_DIALOGUE_WORDS
    if language == "grc":
        return GREEK_DIALOGUE_WORDS
    raise ValueError(f"No dialogue-word inventory for language {language!r}")


def detect_proper_name_tokens(texts: Sequence[str]) -> frozenset[str]:
    """Infer likely names from capitalization away from utterance starts."""
    occurrences: Counter[str] = Counter()
    capitalized: Counter[str] = Counter()
    for text in texts:
        raw_tokens = TOKEN_RE.findall(clean_text(text))
        for position, token in enumerate(raw_tokens):
            folded = fold_token(token)
            occurrences[folded] += 1
            if position > 0 and token[0].isupper():
                capitalized[folded] += 1
    return frozenset(
        token
        for token, count in occurrences.items()
        if capitalized[token] >= 2 and capitalized[token] / count >= 0.6
    )


def classify_term(
    term: str,
    *,
    language: str,
    proper_name_tokens: Iterable[str],
) -> str:
    """Classify an n-gram without pretending to perform full POS tagging."""
    tokens = [fold_token(token) for token in TOKEN_RE.findall(term)]
    if not tokens or any(token in EDITORIAL_TOKENS for token in tokens):
        return "editorial_artifact"
    stops = function_words(language)
    if all(token in stops for token in tokens):
        return "function_word"
    proper = set(proper_name_tokens)
    if any(token in proper for token in tokens):
        return "proper_name"
    dialogue = dialogue_words(language)
    if any(token in dialogue for token in tokens) and all(
        token in dialogue or token in stops for token in tokens
    ):
        return "dialogue_formula"
    return "content_word"
