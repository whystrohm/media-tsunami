"""Shared filter for prescriptive word lists.

Removes contractions, URL fragments, and very short tokens from word lists that
are surfaced prescriptively to LLMs or humans. Raw unfiltered data still lives
in voice-fingerprint.json for research.
"""
from __future__ import annotations

URL_NOISE = {"https", "http", "www", "com", "org", "net", "io"}


def is_display_noise(token: str) -> bool:
    """True if the token should be hidden from prescriptive outputs."""
    if "'" in token:
        return True
    if token in URL_NOISE:
        return True
    if len(token) < 4:
        return True
    return False


def filter_signature_tokens(signature_words: list[dict], limit: int = 15) -> list[str]:
    """Pick the top N signature tokens suitable for prescriptive display."""
    return [w["token"] for w in signature_words if not is_display_noise(w["token"])][:limit]
