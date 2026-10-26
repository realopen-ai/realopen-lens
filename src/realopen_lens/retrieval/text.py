"""Unicode-preserving OCR terms with bounded Han/Kana substring support."""

import re
import unicodedata

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\U00020000-\U0002fa1f]+")


def normalize(text: str) -> str:
    # Keep legacy Latin accent/Arabic vowel folding, but preserve meaningful marks
    # in other scripts (e.g. Hindi/Bengali vowels) and compose Hangul syllables.
    source = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text).casefold())
    result = []
    fold_marks = False
    for char in source:
        category = unicodedata.category(char)
        if category.startswith("M"):
            if not (fold_marks and category == "Mn"):
                result.append(char)
        else:
            name = unicodedata.name(char, "")
            fold_marks = "LATIN" in name or "ARABIC" in name
            result.append(char)
    value = unicodedata.normalize("NFC", "".join(result))
    # OCR commonly inserts spaces between Han/Kana glyphs. Join only these scripts;
    # retain meaningful Latin, Indic and Korean word boundaries.
    chars = r"\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\U00020000-\U0002fa1f"
    return re.sub(rf"(?<=[{chars}])\s+(?=[{chars}])", "", value)


def words(text: str) -> set[str]:
    terms: set[str] = set()
    current: list[str] = []
    for char in text:
        if unicodedata.category(char)[0] in {"L", "N", "M"} or char == "_":
            current.append(char)
        elif current:
            terms.add("".join(current))
            current = []
    if current:
        terms.add("".join(current))
    return terms


def grams(text: str, *, singles: bool = False) -> set[str]:
    terms: set[str] = set()
    for match in _CJK.finditer(text):
        value = match.group()
        if singles or len(value) == 1:
            terms.update(value)
        terms.update(value[i : i + 2] for i in range(len(value) - 1))
    return terms


def tokens(text: str) -> set[str]:
    value = normalize(text)
    return words(value) | grams(value)
