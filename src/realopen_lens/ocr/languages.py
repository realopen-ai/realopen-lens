"""Language data shipped by desktop builds; selection remains explicit."""

LANGUAGES = {
    "eng": "English",
    "fra": "French",
    "ara": "Arabic",
    "spa": "Spanish",
    "por": "Portuguese",
    "deu": "German",
    "ita": "Italian",
    "ind": "Indonesian",
    "tur": "Turkish",
    "vie": "Vietnamese",
    "rus": "Russian",
    "hin": "Hindi",
    "ben": "Bengali",
    "urd": "Urdu",
    "fas": "Persian",
    "chi_sim": "Chinese (simplified)",
    "chi_tra": "Chinese (traditional)",
    "jpn": "Japanese",
    "kor": "Korean",
}
BUNDLED_PACKS = frozenset(LANGUAGES) | {"osd"}


def selection(payload: dict, configured: str | None) -> dict:
    """Freeze actual routing beside the transcript; legacy automatic routing is unknown."""
    return {
        "mode": payload.get("ocr_language_mode", "auto" if configured == "auto" else "manual"),
        "selected_languages": payload.get(
            "selected_languages", None if configured == "auto" else configured
        ),
        "script_detection": payload.get("script_detection"),
    }
