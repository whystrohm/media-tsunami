"""Tests for tsunami.engine.cadence_analyzer."""

from __future__ import annotations

import pytest

from tsunami.engine.cadence_analyzer import analyze_cadence, analyze_corpus


def test_empty_string():
    result = analyze_cadence("")
    assert result["sentence_count"] == 0
    assert result["word_count"] == 0
    assert result["sentence_lengths"] == []
    assert result["mean_length"] == 0.0
    assert result["pronoun_ratio_first"] == 0.0
    assert result["punct_density"] == 0.0
    assert result["fragment_rate"] == 0.0


def test_simple_paragraph():
    text = "I love coding. You should too. It's great!"
    result = analyze_cadence(text)
    assert result["sentence_count"] == 3
    assert 7 <= result["word_count"] <= 10
    assert result["pronoun_ratio_first"] > 0
    assert result["pronoun_ratio_second"] > 0
    assert result["exclamation_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["question_rate"] == 0.0


def test_fragment_heavy():
    result = analyze_cadence("Short. Punchy. To the point.")
    assert result["fragment_rate"] == 1.0
    assert result["sentence_count"] == 3


def test_long_prose():
    text = (
        "The quiet rhythm of the morning settled into the kitchen as the coffee brewed slowly on the stove. "
        "Sunlight slipped through the half-open blinds and painted the counters in warm, slanted stripes of gold. "
        "She lingered by the window for a while, watching the neighbors walk their dog in the dew-wet grass."
    )
    result = analyze_cadence(text)
    assert result["sentence_count"] == 3
    assert result["mean_length"] > 15
    assert result["fragment_rate"] == 0.0


def test_analyze_corpus():
    doc_a = {"path": "a", "title": "A", "text": "I write. You read."}
    doc_b = {"path": "b", "title": "B", "text": "We ship. Fast. Always."}

    sep = analyze_cadence(doc_a["text"])["sentence_count"] + analyze_cadence(doc_b["text"])["sentence_count"]
    combined = analyze_corpus([doc_a, doc_b])
    assert combined["sentence_count"] == sep
    assert combined["word_count"] > 0
    assert combined["fragment_rate"] == 1.0  # all sentences tiny


@pytest.mark.parametrize(
    "text,min_density",
    [
        ("Hello, world! How are you? Fine, thanks.", 10.0),
        ("One, two, three, four, five.", 10.0),
    ],
)
def test_punctuation_density(text, min_density):
    result = analyze_cadence(text)
    assert result["punct_density"] > min_density


def test_empty_corpus():
    assert analyze_corpus([])["sentence_count"] == 0
