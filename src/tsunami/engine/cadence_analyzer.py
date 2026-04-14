"""Cadence analyzer: sentence rhythm, pronoun patterns, punctuation signature."""

from __future__ import annotations

import numpy as np
import spacy

_FIRST_PERSON = {
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "i'm", "i've", "i'd", "i'll",
    "we're", "we've", "we'd", "we'll",
}
_SECOND_PERSON = {
    "you", "your", "yours",
    "you're", "you've", "you'd", "you'll",
}

_ZERO_RESULT = {
    "sentence_count": 0,
    "word_count": 0,
    "sentence_lengths": [],
    "mean_length": 0.0,
    "std_length": 0.0,
    "p25_length": 0.0,
    "p50_length": 0.0,
    "p75_length": 0.0,
    "pronoun_ratio_first": 0.0,
    "pronoun_ratio_second": 0.0,
    "punct_density": 0.0,
    "question_rate": 0.0,
    "exclamation_rate": 0.0,
    "fragment_rate": 0.0,
}

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _NLP = nlp
    return _NLP


def analyze_cadence(text: str) -> dict:
    """Return cadence statistics for the input text."""
    if not text or not text.strip():
        return dict(_ZERO_RESULT, sentence_lengths=[])

    nlp = _get_nlp()
    doc = nlp(text)

    sentence_lengths: list[int] = []
    question_count = 0
    exclamation_count = 0
    fragment_count = 0
    word_count = 0
    punct_count = 0
    first_count = 0
    second_count = 0

    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if not tokens:
            continue
        word_tokens = [t for t in tokens if not t.is_punct]
        n_words = len(word_tokens)
        if n_words == 0:
            # punctuation-only "sentence" — skip
            continue

        sentence_lengths.append(n_words)
        word_count += n_words

        for t in tokens:
            if t.is_punct:
                punct_count += 1
                continue
            lower = t.text.lower()
            if lower in _FIRST_PERSON:
                first_count += 1
            elif lower in _SECOND_PERSON:
                second_count += 1

        # last non-space token to detect terminal punctuation
        last = tokens[-1].text
        if last.endswith("?"):
            question_count += 1
        elif last.endswith("!"):
            exclamation_count += 1

        if n_words < 5:
            fragment_count += 1

    sentence_count = len(sentence_lengths)
    if sentence_count == 0 or word_count == 0:
        return dict(_ZERO_RESULT, sentence_lengths=[])

    lengths_arr = np.array(sentence_lengths, dtype=float)
    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "sentence_lengths": sentence_lengths,
        "mean_length": float(lengths_arr.mean()),
        "std_length": float(lengths_arr.std()),
        "p25_length": float(np.percentile(lengths_arr, 25)),
        "p50_length": float(np.percentile(lengths_arr, 50)),
        "p75_length": float(np.percentile(lengths_arr, 75)),
        "pronoun_ratio_first": first_count / word_count,
        "pronoun_ratio_second": second_count / word_count,
        "punct_density": 100.0 * punct_count / word_count,
        "question_rate": question_count / sentence_count,
        "exclamation_rate": exclamation_count / sentence_count,
        "fragment_rate": fragment_count / sentence_count,
    }


def analyze_corpus(docs: list[dict]) -> dict:
    """Run analyze_cadence over concatenated document texts."""
    if not docs:
        return dict(_ZERO_RESULT, sentence_lengths=[])
    combined = "\n\n".join(d.get("text", "") for d in docs)
    return analyze_cadence(combined)
