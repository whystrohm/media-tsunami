"""Tests for tsunami.engine.voice_fingerprinter."""

from __future__ import annotations

import numpy as np

from tsunami.engine.voice_fingerprinter import (
    _is_usable_sentence,
    fingerprint_voice,
    render_length_histogram,
)


def test_is_usable_sentence_accepts_normal():
    assert _is_usable_sentence("This is a perfectly normal prose sentence about voice.")


def test_is_usable_sentence_rejects_newline_break():
    bad = "For founders who are the voice of their company\n\nEverything in your head."
    assert not _is_usable_sentence(bad)


def test_is_usable_sentence_rejects_long():
    s = " ".join(["word"] * 70)
    assert not _is_usable_sentence(s)


def test_is_usable_sentence_rejects_too_long_chars():
    s = "a" * 600
    assert not _is_usable_sentence(s)


def test_is_usable_sentence_rejects_fragment():
    assert not _is_usable_sentence("Too short.")


def test_fingerprint_empty_docs():
    fp = fingerprint_voice([])
    assert fp["sentence_count"] == 0
    assert fp["closest_to_centroid"] == []
    assert fp["farthest_from_centroid"] == []
    assert fp["top_vocab"] == []
    assert fp["embedding_dim"] == 0
    assert isinstance(fp["centroid_distances"], np.ndarray)
    assert fp["centroid_distances"].size == 0


def _synthetic_docs() -> list[dict]:
    d1 = {
        "text": (
            "Your brand voice is the signature of your company. "
            "Founders know this instinctively. "
            "Every sentence you write should sound like you, not a template. "
            "The tone you use sets the standard for every hire that follows."
        ),
        "source": "doc1",
    }
    d2 = {
        "text": (
            "Content systems fail when the voice drifts. "
            "A founder writes one landing page and it sings. "
            "Then a junior marketer writes the next one and it dies. "
            "The gap is voice, not skill."
        ),
        "source": "doc2",
    }
    d3 = {
        "text": (
            "Voice is a compounding asset. "
            "Every piece published either reinforces the signature or erodes it. "
            "Most companies erode. "
            "A few compound."
        ),
        "source": "doc3",
    }
    return [d1, d2, d3]


def test_fingerprint_returns_required_keys():
    docs = _synthetic_docs()
    fp = fingerprint_voice(docs, sample_size=3)
    required = {
        "sentence_count",
        "word_count",
        "cadence",
        "top_vocab",
        "centroid_distances",
        "closest_to_centroid",
        "farthest_from_centroid",
        "embedding_dim",
        "embed_model",
    }
    assert required.issubset(fp.keys())
    assert fp["sentence_count"] > 0
    assert fp["embedding_dim"] > 0


def test_fingerprint_centroid_distances_sorted_extremes():
    docs = _synthetic_docs()
    fp = fingerprint_voice(docs, sample_size=3)
    closest = fp["closest_to_centroid"]
    farthest = fp["farthest_from_centroid"]
    assert len(closest) > 0
    assert len(farthest) > 0
    max_closest = max(s["distance"] for s in closest)
    min_farthest = min(s["distance"] for s in farthest)
    assert max_closest <= min_farthest + 1e-9


def test_fingerprint_closest_sentences_are_clean():
    docs = _synthetic_docs()
    # Inject a junk block: header + broken cross-block text without terminal punct.
    junk_doc = {
        "text": "Header:\n\nbad broken sentence without punctuation across the block.",
        "source": "junk",
    }
    fp = fingerprint_voice(docs + [junk_doc], sample_size=5)
    for s in fp["closest_to_centroid"]:
        assert "\n\n" not in s["text"]


def test_render_length_histogram():
    out = render_length_histogram([3, 5, 7, 9, 11, 13, 15, 4, 6, 8, 10])
    assert isinstance(out, str)
    assert len(out) > 0
    assert "█" in out
