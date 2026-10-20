"""Conservative typed numeric intent and query aliases without an index rebuild."""

import ipaddress
import re
import unicodedata


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text).casefold())
    return "".join(
        str(unicodedata.decimal(c)) if c.isdecimal() else c
        for c in text
        if unicodedata.category(c) != "Mn"
    )


_MARKERS = {
    "form": {"form", "formulaire", "نموذج"},
    "invoice": {"invoice", "facture", "فاتورة"},
    "error": {"error", "erreur", "خطأ"},
}
_MARKERS = {kind: {_normalized(word) for word in words} for kind, words in _MARKERS.items()}
_QUALIFIERS = {
    _normalized(word)
    for word in ("no", "number", "numero", "n", "id", "code", "reference", "ref", "رقم")
}
_AMBIGUOUS = {
    _normalized(word)
    for word in (
        "year",
        "date",
        "dated",
        "quantity",
        "qty",
        "total",
        "amount",
        "price",
        "dollars",
        "annee",
        "quantite",
        "montant",
        "prix",
        "عام",
        "سنة",
        "تاريخ",
        "كمية",
        "مبلغ",
        "اجمالي",
    )
}


def _field_pattern(markers: set[str], qualifiers: set[str]) -> re.Pattern[str]:
    return re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(word) for word in sorted(markers))
        + r")(?=\W|\d)[\s#:°.]*"
        + r"(?:(?:"
        + "|".join(re.escape(word) for word in sorted(qualifiers))
        + r")(?!\w)[\s#:°.]*){0,2}(?P<value>\d\w*(?:[-_/.:]\w+)*)"
    )


_FIELD_PATTERNS = {kind: _field_pattern(markers, _QUALIFIERS) for kind, markers in _MARKERS.items()}
_QUERY_PATTERNS = {
    kind: _field_pattern(
        markers,
        _QUALIFIERS
        | ({_normalized(w) for w in ("IRS", "مصلحة", "الضرائب")} if kind == "form" else set()),
    )
    for kind, markers in _MARKERS.items()
}


def numeric_intent(query: str) -> tuple[str, str] | None:
    """Require one standalone number and one compatible typed intent, no quantity/date cues."""
    normalized = _normalized(query)
    words = re.findall(r"\w+", normalized)
    if set(words) & _AMBIGUOUS:
        return None
    numbers = [word for word in words if word.isdecimal()]
    if len(numbers) != 1 or not 2 <= len(numbers[0]) <= 24:
        return None
    kinds = [kind for kind, markers in _MARKERS.items() if set(words) & markers]
    if len(kinds) != 1:
        return None
    field = _QUERY_PATTERNS[kinds[0]].search(normalized)
    if field is None or field.group("value") != numbers[0]:
        return None
    # A suffix/prefix belongs to the identifier: 1099-MISC is not bare 1099.
    # Only a direct typed field such as Error:1045 may use a numeric separator.
    for compound in re.findall(r"\b\w+(?:[-_/.:]\w+)+\b", normalized):
        if any(char.isdecimal() for char in compound):
            field = _FIELD_PATTERNS[kinds[0]].fullmatch(compound)
            if field is None or field.group("value") != numbers[0]:
                return None
    return kinds[0], numbers[0]


def numeric_identifier_match(query: str, text: str) -> bool:
    intent = numeric_intent(query)
    if intent is None:
        return False
    kind, number = intent
    field = _FIELD_PATTERNS[kind].search(_normalized(text))
    return field is not None and field.group("value") == number


def numeric_query_aliases(query: str) -> set[str]:
    """Search existing Unicode-digit FTS terms without changing stored tokenization."""
    intent = numeric_intent(query)
    if intent is None:
        return set()
    number = intent[1]
    return {
        number,
        number.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")),
        number.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")),
    }


def code_identifier(value: str) -> bool:
    """Punctuation alone does not identify a code; retain valid numeric IP literals."""
    if any(char.isalpha() for char in value):
        return True
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        host, separator, port = value.rpartition(":")
        if separator and port.isascii() and port.isdecimal() and 1 <= int(port) <= 65535:
            try:
                ipaddress.ip_address(host.strip("[]"))
                return True
            except ValueError:
                pass
        return False
