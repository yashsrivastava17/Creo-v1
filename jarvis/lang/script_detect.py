from __future__ import annotations

import regex as re

DEVANAGARI = re.compile(r"\p{Script=Devanagari}")
ALNUM_OR_MARK = re.compile(r"[\p{L}\p{M}]+", re.UNICODE)


def detect_lang_from_text(text: str | None) -> str:
    """Return 'hi' if *text* contains any Devanagari characters, else 'en'."""
    if not text or not isinstance(text, str):
        return "en"

    matches = ALNUM_OR_MARK.findall(text)
    if not matches:
        return "en"

    letters = "".join(matches)
    if DEVANAGARI.search(letters):
        return "hi"
    return "en"


__all__ = ["detect_lang_from_text"]
