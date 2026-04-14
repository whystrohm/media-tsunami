"""Tests for tsunami.engine.vocabulary_clusterer."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tsunami.engine.vocabulary_clusterer import cluster_vocabulary
from tsunami.inputs.folder_reader import read_folder

CORPUS_PATH = Path(__file__).resolve().parent.parent / "test_corpus" / "whystrohm"


def _make_doc(path: str, text: str) -> dict:
    return {"path": path, "title": path, "text": text}


def test_empty_input():
    import numpy as np

    result = cluster_vocabulary([])
    assert result["clusters"] == []
    assert result["n_clusters"] == 0
    assert result["dropped_clusters"] == 0
    assert result["total_unique_tokens"] == 0
    assert result["total_tokens_clustered"] == 0
    assert isinstance(result["centroid_embeddings"], np.ndarray)
    assert result["centroid_embeddings"].shape == (0, 384)


def test_tiny_corpus():
    docs = [
        _make_doc(
            "a",
            "The founder shipped the brand system quickly. The brand system guides writing. "
            "Writing ships when the system holds. The founder holds the system steady.",
        ),
        _make_doc(
            "b",
            "Content infrastructure matters. The system ships content every day. "
            "Every founder needs infrastructure. The brand keeps shipping content anyway.",
        ),
    ]
    result = cluster_vocabulary(docs, n_clusters=3, min_token_freq=2)
    # With a tiny corpus we may get fewer clusters than requested (including
    # post-hoc drops) but must not crash.
    assert result["n_clusters"] <= 3
    assert result["n_clusters"] >= 1
    assert result["dropped_clusters"] >= 0
    assert result["total_unique_tokens"] > 0
    assert len(result["clusters"]) == result["n_clusters"]
    for c in result["clusters"]:
        assert "id" in c
        assert "label" in c
        assert "top_tokens" in c
        assert "size" in c
        assert "centroid_token" in c


def test_semantic_coherence():
    animals = ["dog", "cat", "bird", "fish", "rabbit"]
    tech = ["software", "hardware", "database", "server", "network"]
    food = ["bread", "pasta", "rice", "salad", "soup"]

    def make_topic_doc(words: list[str], context: str) -> str:
        # Each topic word appears 5+ times across the corpus.
        lines = []
        for w in words:
            for _ in range(6):
                lines.append(f"The {w} about {context} was notable today here.")
        return " ".join(lines)

    docs = [
        _make_doc("animals", make_topic_doc(animals, "pets")),
        _make_doc("tech", make_topic_doc(tech, "systems")),
        _make_doc("food", make_topic_doc(food, "kitchen")),
    ]
    result = cluster_vocabulary(docs, n_clusters=3, min_token_freq=3)
    assert result["n_clusters"] == 3

    # Map tokens to their assigned cluster id.
    tok_to_cluster: dict[str, int] = {}
    for c in result["clusters"]:
        for tok, _ in c["top_tokens"]:
            tok_to_cluster[tok] = c["id"]

    # For each topic, the 5 seed words should land mostly in the same cluster.
    for topic_words in (animals, tech, food):
        present = [tok_to_cluster[w] for w in topic_words if w in tok_to_cluster]
        assert len(present) >= 3, f"Missing seed words for topic: {topic_words}"
        # Dominant cluster among present seeds
        from collections import Counter as _C
        dominant_count = _C(present).most_common(1)[0][1]
        assert dominant_count >= 3, (
            f"Topic {topic_words} did not cluster coherently: assignments={present}"
        )


def test_against_whystrohm_corpus():
    assert CORPUS_PATH.exists(), f"Missing corpus at {CORPUS_PATH}"
    docs = read_folder(CORPUS_PATH)
    assert len(docs) > 0
    result = cluster_vocabulary(docs, n_clusters=5)
    # Post-hoc filter may drop stopword-heavy clusters.
    assert 3 <= result["n_clusters"] <= 6
    assert len(result["clusters"]) == result["n_clusters"]
    assert result["dropped_clusters"] >= 0

    all_top_tokens = set()
    for c in result["clusters"]:
        assert c["label"], f"Cluster {c['id']} has empty label"
        assert len(c["top_tokens"]) >= 1, f"Cluster {c['id']} has no top_tokens"
        for tok, weight in c["top_tokens"]:
            all_top_tokens.add(tok)

    assert any(t in all_top_tokens for t in ("content", "brand", "voice")), (
        f"Expected core brand tokens in result; got: {sorted(all_top_tokens)[:40]}"
    )


def test_contractions_filtered_pre_cluster():
    # Synthetic corpus where "it's" appears 20+ times alongside real topic tokens.
    its_line = "it's it's it's it's it's it's it's it's it's it's "
    topic_line = (
        "The brand system ships content. The founder guides content strategy. "
        "Content infrastructure matters. Strategy keeps shipping. System holds. "
    )
    docs = [
        _make_doc("a", its_line + topic_line * 5),
        _make_doc("b", its_line + topic_line * 5),
    ]
    result = cluster_vocabulary(docs, n_clusters=3, min_token_freq=2)
    for c in result["clusters"]:
        top_toks = [tok for tok, _ in c["top_tokens"]]
        assert "it's" not in top_toks, (
            f"Contraction 'it's' leaked into cluster {c['id']} top_tokens: {top_toks}"
        )


def test_post_hoc_drops_stopword_clusters():
    # Engineer a corpus with one obviously stopword-heavy semantic region
    # (determiner/quantifier family) alongside two real topics.
    stopword_blob = (
        "some every most many several much more such each some every most many "
        "such much more each several some every most many such more each "
    ) * 4
    tech_blob = (
        "software hardware database server network software hardware database server "
        "network software hardware database server network software hardware database "
    ) * 3
    food_blob = (
        "bread pasta rice salad soup bread pasta rice salad soup bread pasta rice "
        "salad soup bread pasta rice salad soup bread pasta rice salad soup "
    ) * 3
    docs = [
        _make_doc("sw", stopword_blob),
        _make_doc("tech", tech_blob),
        _make_doc("food", food_blob),
    ]
    result = cluster_vocabulary(docs, n_clusters=3, min_token_freq=3)
    assert result["dropped_clusters"] > 0, (
        f"Expected stopword-heavy cluster to be dropped; result={result}"
    )


def test_whystrohm_no_function_word_cluster():
    assert CORPUS_PATH.exists(), f"Missing corpus at {CORPUS_PATH}"
    docs = read_folder(CORPUS_PATH)
    result = cluster_vocabulary(docs, n_clusters=6)

    # Pre-filter guarantee: contractions must never appear in any cluster label
    # or top_tokens. These are the worst offenders from the original bad cluster
    # ("that's, it's, you're, doesn't, isn't" were 5 of 8 tokens).
    contractions = {"that's", "it's", "you're", "doesn't", "isn't", "don't", "can't"}
    for c in result["clusters"]:
        top_toks = {tok for tok, _ in c["top_tokens"]}
        leaked = contractions & top_toks
        assert not leaked, (
            f"Contractions leaked into cluster {c['id']} label={c['label']!r}: {leaked}"
        )
        for tok in top_toks:
            assert "'" not in tok, (
                f"Contraction-shaped token {tok!r} leaked into cluster {c['id']}"
            )
        # And no cluster label should be anchored by the original bad label head.
        assert not c["label"].startswith("that's"), c["label"]
        assert not c["label"].startswith("it's"), c["label"]
        assert not c["label"].startswith("you're"), c["label"]


def test_performance():
    assert CORPUS_PATH.exists()
    docs = read_folder(CORPUS_PATH)
    # Warm the embedder singleton once so we measure steady-state.
    _ = cluster_vocabulary(docs, n_clusters=3)
    start = time.perf_counter()
    cluster_vocabulary(docs)
    elapsed = time.perf_counter() - start
    assert elapsed < 8.0, f"cluster_vocabulary too slow: {elapsed:.2f}s"
