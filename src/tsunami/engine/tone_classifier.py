"""Tone classifier — heuristic rule-based mapping from cadence signals to tone labels.

Transparent rule table. No LLM calls. Fast (<5ms), deterministic, free.
Each tone has a set of triggering conditions; its score = fraction of triggers that fired.
"""
from __future__ import annotations

# Tone taxonomy with ordered trigger evaluations.
# Each trigger is (name, predicate) where predicate takes the signals dict and returns bool.
# Order matters only for tie-breaking (labels earlier in this list win ties).
_TONE_RULES = [
    ("direct", [
        ("short mean length", lambda s: s["mean_length"] < 12),
        ("high fragment rate", lambda s: s["fragment_rate"] > 0.25),
        ("tight p75 length", lambda s: s["p75_length"] < 18),
        ("lean punctuation", lambda s: s["punct_density"] < 15),
    ]),
    ("conversational", [
        ("second-person pronouns", lambda s: s["pronoun_ratio_second"] > 0.015),
        ("first-person pronouns", lambda s: s["pronoun_ratio_first"] > 0.005),
        ("medium mean length", lambda s: 10 <= s["mean_length"] <= 18),
        ("contractions present", lambda s: s.get("has_contractions", False)),
    ]),
    ("formal", [
        ("long mean length", lambda s: s["mean_length"] > 18),
        ("rare second-person", lambda s: s["pronoun_ratio_second"] < 0.005),
        ("rare first-person", lambda s: s["pronoun_ratio_first"] < 0.005),
        ("low fragment rate", lambda s: s["fragment_rate"] < 0.1),
    ]),
    ("energetic", [
        ("exclamations", lambda s: s["exclamation_rate"] > 0.02),
        ("high length variance", lambda s: s["std_length"] > 7),
        ("some fragments", lambda s: s["fragment_rate"] > 0.2),
        ("dense punctuation", lambda s: s["punct_density"] > 18),
    ]),
    ("instructive", [
        ("questions", lambda s: s["question_rate"] > 0.03),
        ("strong second-person", lambda s: s["pronoun_ratio_second"] > 0.02),
        ("teaching length", lambda s: 10 <= s["mean_length"] <= 16),
    ]),
    ("analytical", [
        ("long-ish mean length", lambda s: 14 <= s["mean_length"] <= 22),
        ("consistent length", lambda s: s["std_length"] < 8),
        ("rare exclamations", lambda s: s["exclamation_rate"] < 0.01),
        ("few fragments", lambda s: s["fragment_rate"] < 0.15),
    ]),
    ("punchy", [
        ("very short mean length", lambda s: s["mean_length"] < 10),
        ("very high fragment rate", lambda s: s["fragment_rate"] > 0.3),
        ("tight median", lambda s: s["p50_length"] <= 7),
        ("dense punctuation", lambda s: s["punct_density"] > 15),
    ]),
    ("authoritative", [
        ("medium-long mean length", lambda s: s["mean_length"] > 14),
        ("rare first-person", lambda s: s["pronoun_ratio_first"] < 0.01),
        ("rare questions", lambda s: s["question_rate"] < 0.02),
        ("rare exclamations", lambda s: s["exclamation_rate"] < 0.01),
    ]),
]

_SIGNAL_KEYS = (
    "mean_length", "std_length", "p25_length", "p50_length", "p75_length",
    "pronoun_ratio_first", "pronoun_ratio_second",
    "punct_density", "question_rate", "exclamation_rate", "fragment_rate",
)


def _signals_from_cadence(cadence: dict, signature_words: list[dict] | None) -> dict:
    """Pull the numeric signals from the cadence dict + derive contraction flag."""
    signals = {k: float(cadence.get(k, 0.0)) for k in _SIGNAL_KEYS}
    if signature_words:
        signals["has_contractions"] = any("'" in (w.get("token") or "") for w in signature_words)
    else:
        signals["has_contractions"] = False
    return signals


def _format_signal_value(trigger_name: str, signals: dict) -> str:
    """Small helper — pick the likely signal value to mention in a rationale line."""
    # Map trigger descriptions to the signal they reference so rationale strings show a number.
    if "mean length" in trigger_name:
        return f"mean={signals['mean_length']:.1f} tokens"
    if "fragment" in trigger_name:
        return f"fragment={signals['fragment_rate'] * 100:.0f}%"
    if "p75" in trigger_name:
        return f"p75={signals['p75_length']:.1f}"
    if "median" in trigger_name or "p50" in trigger_name:
        return f"p50={signals['p50_length']:.1f}"
    if "punctuation" in trigger_name:
        return f"punct_density={signals['punct_density']:.1f}"
    if "variance" in trigger_name:
        return f"std={signals['std_length']:.1f}"
    if "second-person" in trigger_name:
        return f"you-ratio={signals['pronoun_ratio_second'] * 100:.2f}%"
    if "first-person" in trigger_name:
        return f"i-ratio={signals['pronoun_ratio_first'] * 100:.2f}%"
    if "exclamation" in trigger_name:
        return f"excl_rate={signals['exclamation_rate'] * 100:.1f}%"
    if "question" in trigger_name:
        return f"q_rate={signals['question_rate'] * 100:.1f}%"
    if "contraction" in trigger_name:
        return "contractions=1" if signals.get("has_contractions") else "contractions=0"
    return "signal present"


def _rationale_for(tone: str, signals: dict, max_items: int = 3) -> list[str]:
    """Build 1-3 rationale strings for a tone, listing the triggers that fired."""
    triggers = dict(_TONE_RULES)[tone]
    lines: list[str] = []
    for name, pred in triggers:
        try:
            fired = bool(pred(signals))
        except Exception:
            fired = False
        if fired:
            lines.append(f"{tone}: {name} ({_format_signal_value(name, signals)})")
            if len(lines) >= max_items:
                break
    return lines


def classify_tone(
    cadence: dict,
    signature_words: list[dict] | None = None,
) -> dict:
    """Classify brand tone from cadence and vocabulary signals."""
    empty_labels = {label: 0.0 for label, _ in _TONE_RULES}

    if not cadence or cadence.get("sentence_count", 0) == 0 or cadence.get("word_count", 0) == 0:
        return {
            "primary_tone": "unknown",
            "secondary_tone": None,
            "tone_scores": empty_labels,
            "signals": {},
            "confidence": 0.0,
            "uncertain": True,
            "rationale": ["empty input"],
        }

    signals = _signals_from_cadence(cadence, signature_words)

    # Score each tone as fraction of triggers that fired.
    scores: dict[str, float] = {}
    for label, triggers in _TONE_RULES:
        fired = 0
        for _, pred in triggers:
            try:
                if pred(signals):
                    fired += 1
            except Exception:
                continue
        scores[label] = fired / len(triggers) if triggers else 0.0

    # Rank preserving declaration order for deterministic tie-breaking.
    label_order = {label: i for i, (label, _) in enumerate(_TONE_RULES)}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], label_order[kv[0]]))

    top_label, top_score = ranked[0]
    if top_score == 0.0:
        return {
            "primary_tone": "mixed",
            "secondary_tone": None,
            "tone_scores": scores,
            "signals": signals,
            "confidence": 0.0,
            "uncertain": True,
            "rationale": ["no tone triggers fired"],
        }

    primary_tone = top_label
    confidence = top_score

    secondary_tone: str | None = None
    if len(ranked) > 1:
        second_label, second_score = ranked[1]
        if second_score >= 0.5 and second_score >= confidence - 0.25:
            secondary_tone = second_label

    rationale = _rationale_for(primary_tone, signals)
    if secondary_tone:
        rationale.extend(_rationale_for(secondary_tone, signals, max_items=2))
    # Clamp to 3–6 items.
    if len(rationale) > 6:
        rationale = rationale[:6]
    if not rationale:
        rationale = [f"{primary_tone}: no per-trigger detail available"]

    return {
        "primary_tone": primary_tone,
        "secondary_tone": secondary_tone,
        "tone_scores": scores,
        "signals": signals,
        "confidence": confidence,
        "uncertain": confidence < 0.6,
        "rationale": rationale,
    }
