"""Tests for tone_classifier."""
from __future__ import annotations

import re

import pytest

from tsunami.engine.cadence_analyzer import _ZERO_RESULT, analyze_corpus
from tsunami.engine.forbidden_detector import detect_forbidden_and_signature
from tsunami.engine.tone_classifier import classify_tone
from tsunami.inputs.folder_reader import read_folder


def _base_signals(**overrides) -> dict:
    """Start from the zero cadence dict, override signal keys, set non-empty counts."""
    out = dict(_ZERO_RESULT)
    out["sentence_count"] = 100
    out["word_count"] = 1200
    out.update(overrides)
    return out


def test_empty_cadence():
    cadence = dict(_ZERO_RESULT)
    cadence["sentence_count"] = 0
    cadence["word_count"] = 0
    r = classify_tone(cadence)
    assert r["primary_tone"] == "unknown"
    assert r["uncertain"] is True
    assert r["confidence"] == 0.0
    assert r["rationale"] == ["empty input"]


def test_direct_punchy_signal():
    cadence = _base_signals(
        mean_length=8.0,
        fragment_rate=0.35,
        p75_length=13.0,
        p50_length=6.0,
        std_length=5.0,
        punct_density=12.0,
        pronoun_ratio_first=0.002,
        pronoun_ratio_second=0.002,
        question_rate=0.0,
        exclamation_rate=0.0,
    )
    r = classify_tone(cadence)
    assert r["primary_tone"] in {"direct", "punchy"}
    assert r["confidence"] >= 0.75


def test_formal_signal():
    cadence = _base_signals(
        mean_length=22.0,
        pronoun_ratio_second=0.001,
        pronoun_ratio_first=0.001,
        fragment_rate=0.05,
        std_length=5.0,
        p50_length=22.0,
        p75_length=26.0,
        exclamation_rate=0.0,
        question_rate=0.0,
        punct_density=14.0,
    )
    r = classify_tone(cadence)
    assert r["primary_tone"] in {"formal", "analytical", "authoritative"}
    assert r["confidence"] >= 0.75


def test_conversational_signal():
    cadence = _base_signals(
        mean_length=14.0,
        pronoun_ratio_second=0.04,
        pronoun_ratio_first=0.01,
        fragment_rate=0.15,
        std_length=6.0,
        p50_length=13.0,
        p75_length=18.0,
        exclamation_rate=0.005,
        question_rate=0.02,
        punct_density=14.0,
    )
    sig_words = [
        {"token": "you're", "brand_rate": 500.0, "baseline_rate": 10.0, "ratio": 50.0, "brand_count": 20},
        {"token": "it's", "brand_rate": 400.0, "baseline_rate": 50.0, "ratio": 8.0, "brand_count": 16},
    ]
    r = classify_tone(cadence, signature_words=sig_words)
    top_two = {r["primary_tone"], r["secondary_tone"]}
    assert "conversational" in top_two


def test_whystrohm_corpus():
    docs = read_folder("test_corpus/whystrohm")
    cadence = analyze_corpus(docs)
    fpout = detect_forbidden_and_signature(docs)
    r = classify_tone(cadence, signature_words=fpout["signature_words"])
    assert r["primary_tone"] in {"direct", "punchy", "conversational", "energetic"}
    assert r["confidence"] >= 0.6
    assert len(r["rationale"]) >= 2


def test_rationale_strings():
    cadence = _base_signals(
        mean_length=8.0,
        fragment_rate=0.35,
        p75_length=13.0,
        p50_length=6.0,
        std_length=5.0,
        punct_density=12.0,
        pronoun_ratio_first=0.002,
        pronoun_ratio_second=0.002,
    )
    r = classify_tone(cadence)
    assert len(r["rationale"]) >= 1
    for line in r["rationale"]:
        assert isinstance(line, str)
        assert line.strip()
        assert re.search(r"\d", line), f"rationale should mention a numeric signal value: {line!r}"


def test_deterministic():
    cadence = _base_signals(
        mean_length=11.0,
        fragment_rate=0.22,
        p75_length=16.0,
        p50_length=10.0,
        std_length=6.0,
        punct_density=14.0,
        pronoun_ratio_first=0.01,
        pronoun_ratio_second=0.02,
        question_rate=0.015,
        exclamation_rate=0.005,
    )
    r1 = classify_tone(cadence)
    r2 = classify_tone(cadence)
    assert r1 == r2
