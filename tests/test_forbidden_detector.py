"""Tests for tsunami.engine.forbidden_detector."""
from __future__ import annotations

from pathlib import Path

import pytest

from tsunami.engine.forbidden_detector import (
    _get_baseline,
    detect_forbidden_and_signature,
)
from tsunami.inputs.folder_reader import read_folder

CORPUS_PATH = Path(__file__).resolve().parent.parent / "test_corpus" / "whystrohm"

_REQUIRED_KEYS = {"token", "brand_rate", "baseline_rate", "ratio", "brand_count"}


def _make_doc(path: str, text: str) -> dict:
    return {"path": path, "title": path, "text": text}


def test_baseline_loads_or_builds():
    baseline = _get_baseline()
    assert isinstance(baseline, dict)
    assert "rates" in baseline
    rates = baseline["rates"]
    # Common, non-stopword English tokens that SHOULD be in a wikitext baseline.
    for tok in ("time", "people", "make", "year"):
        assert tok in rates, f"expected {tok!r} in wikitext baseline"
        assert rates[tok] > 0


def test_empty_corpus():
    result = detect_forbidden_and_signature([])
    assert result["signature_words"] == []
    assert result["forbidden_words"] == []
    assert result["baseline"] == "wikitext-2"
    assert result["brand_total_tokens"] == 0


def test_signature_detection():
    text_a = "Our infrastructure holds. The infrastructure ships. Infrastructure wins today."
    text_b = (
        "Brand infrastructure is the moat. Infrastructure compounds over time. "
        "We built infrastructure to last. Infrastructure beats tactics every time."
    )
    text_c = (
        "Infrastructure scales. Infrastructure is quiet. Infrastructure carries the load. "
        "When infrastructure fails, everything fails. Ship infrastructure first, always. "
        "Good infrastructure is invisible. Infrastructure is the real asset here today."
    )
    docs = [
        _make_doc("a", text_a),
        _make_doc("b", text_b),
        _make_doc("c", text_c),
    ]
    result = detect_forbidden_and_signature(docs, min_brand_count=3, top_signature=10)
    sig_tokens = [w["token"] for w in result["signature_words"]]
    assert "infrastructure" in sig_tokens, f"got: {sig_tokens}"
    infra = next(w for w in result["signature_words"] if w["token"] == "infrastructure")
    assert infra["ratio"] > 5, f"expected ratio > 5, got {infra['ratio']}"
    assert infra["brand_count"] >= 10
    # Infrastructure should rank in the top 10 signature slots for this contrived corpus.
    sig_top10 = [w["token"] for w in result["signature_words"][:10]]
    assert "infrastructure" in sig_top10, f"infrastructure not in top 10: {sig_top10}"


def test_forbidden_detection():
    bball_vocab = [
        "dribble", "layup", "rebound", "shot", "player", "team", "game",
        "coach", "assist", "court", "jumper", "defender", "pass",
    ]
    # ~500 tokens, all basketball-specific. No "time", "world", "part", "years".
    lines = []
    for i in range(40):
        for w in bball_vocab:
            lines.append(w)
    text = " ".join(lines)
    docs = [_make_doc("hoops", text)]
    result = detect_forbidden_and_signature(docs, top_forbidden=30)
    forbidden = {w["token"] for w in result["forbidden_words"]}
    # At least 3 of these common wikitext words should show up as forbidden.
    # Note: numeric/generic words like "time", "years", "made", "new", "first"
    # are filtered by _GENERIC_NOISE (topically empty, not voice signal).
    # We expect the remaining genuinely stylistic/topical words.
    expected_pool = {"world", "system", "used", "however", "including",
                     "although", "several", "known", "music", "war", "film",
                     "south", "north"}
    hits = forbidden & expected_pool
    assert len(hits) >= 3, f"expected >=3 genuine wikitext words in forbidden; got hits={hits}, list={forbidden}"


def test_against_whystrohm_corpus():
    assert CORPUS_PATH.exists(), f"Missing corpus at {CORPUS_PATH}"
    docs = read_folder(CORPUS_PATH)
    assert len(docs) > 0
    result = detect_forbidden_and_signature(docs, top_signature=30, top_forbidden=30)

    sig_tokens = {w["token"] for w in result["signature_words"]}
    forbidden_tokens = {w["token"] for w in result["forbidden_words"]}

    brand_pool = {"content", "brand", "voice", "infrastructure", "founder", "guardrails", "whystrohm"}
    assert sig_tokens & brand_pool, (
        f"expected brand-flavored signature words; got: {sorted(sig_tokens)}"
    )

    wikitext_pool = {"government", "war", "century", "army", "king", "city", "state"}
    assert forbidden_tokens & wikitext_pool, (
        f"expected wikitext-flavored forbidden words; got: {sorted(forbidden_tokens)}"
    )

    assert len(result["signature_words"]) > 0
    assert len(result["forbidden_words"]) > 0
    for entry in result["signature_words"] + result["forbidden_words"]:
        assert _REQUIRED_KEYS.issubset(entry.keys())


def test_result_schema():
    docs = [
        _make_doc("a", "The founder ships the brand system quickly. " * 5),
        _make_doc("b", "Content infrastructure matters. The system ships content every day. " * 5),
    ]
    result = detect_forbidden_and_signature(docs, min_brand_count=2)
    assert set(result.keys()) >= {
        "signature_words", "forbidden_words", "baseline", "brand_total_tokens", "stats"
    }
    for entry in result["signature_words"] + result["forbidden_words"]:
        assert _REQUIRED_KEYS.issubset(entry.keys()), f"missing keys in {entry}"
        assert isinstance(entry["token"], str)
        assert isinstance(entry["brand_count"], int)
        assert isinstance(entry["ratio"], float)


# ---------------------------------------------------------------------------
# Semantic filter tests (FIX 5)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _whystrohm_clusters():
    """Build vocabulary clusters on the WhyStrohm corpus once per module run."""
    from tsunami.engine.vocabulary_clusterer import cluster_vocabulary

    docs = read_folder(CORPUS_PATH)
    clusters = cluster_vocabulary(docs, n_clusters=6, top_n_per_cluster=12)
    return docs, clusters


def test_semantic_filter_keeps_stylistic_words(_whystrohm_clusters):
    """Common English hedges inside the brand's SaaS/marketing semantic territory
    should survive the filter and land in forbidden_words."""
    docs, clusters = _whystrohm_clusters
    centroids = clusters["centroid_embeddings"]
    assert centroids.shape[0] > 0

    result = detect_forbidden_and_signature(
        docs,
        top_forbidden=25,
        top_signature=25,
        cluster_centroids=centroids,
    )
    forbidden_tokens = {w["token"] for w in result["forbidden_words"]}
    assert forbidden_tokens, "forbidden_words should be non-empty after filter"

    stylistic_pool = {"however", "although", "several", "including", "known"}
    hits = forbidden_tokens & stylistic_pool
    assert hits, (
        f"expected at least one stylistic hedge to survive the filter; "
        f"got forbidden_tokens={sorted(forbidden_tokens)}"
    )
    # Every stylistic hit must carry its min_cluster_distance.
    for entry in result["forbidden_words"]:
        assert "min_cluster_distance" in entry
        assert isinstance(entry["min_cluster_distance"], float)
        assert entry["min_cluster_distance"] <= result["relevance_threshold"]


def test_semantic_filter_demotes_topical_words(_whystrohm_clusters):
    """Wikipedia-topical nouns (geography/places) should be demoted out of
    forbidden_words into forbidden_irrelevant.

    Empirical note: on the WhyStrohm corpus (a content/media brand), tokens
    like ``album``, ``film``, ``music``, ``song`` are actually semantically
    adjacent to the brand's ``voice``/``video``/``content`` clusters — MiniLM
    cannot distinguish them from abstract stylistic hedges on token-level
    centroid distance. The filter's reliable value shows up on clearly
    off-topic Wikipedia noise (place names, geography) whose min-cluster
    distance sits higher. At the default 0.65 threshold very few candidates
    clear the bar on this corpus; we use a slightly tighter 0.60 here to
    exercise the partition mechanism on reliable geographic noise.
    """
    docs, clusters = _whystrohm_clusters
    centroids = clusters["centroid_embeddings"]

    result = detect_forbidden_and_signature(
        docs,
        top_forbidden=25,
        top_signature=25,
        cluster_centroids=centroids,
        relevance_threshold=0.60,
    )
    forbidden_tokens = {w["token"] for w in result["forbidden_words"]}
    irrelevant_tokens = {w["token"] for w in result["forbidden_irrelevant"]}

    # Geographic/topical Wikipedia noise whose min-cluster distance on this
    # corpus is reliably > 0.60 (measured empirically).
    topical_pool = {"north", "south", "east", "west", "england", "york",
                    "tropical", "species"}

    demoted = irrelevant_tokens & topical_pool
    assert len(demoted) >= 3, (
        f"expected >=3 of {topical_pool} demoted to forbidden_irrelevant; "
        f"got demoted={demoted}, irrelevant={sorted(irrelevant_tokens)[:30]}"
    )
    # These specific demoted items must not have leaked into forbidden_words.
    leaked = forbidden_tokens & topical_pool
    assert len(leaked) < len(demoted), (
        f"more topical words leaked ({leaked}) than demoted ({demoted}) — filter broken"
    )

    # Every irrelevant entry must carry its min_cluster_distance above threshold.
    for entry in result["forbidden_irrelevant"]:
        assert "min_cluster_distance" in entry
        assert entry["min_cluster_distance"] > result["relevance_threshold"]


def test_backward_compat_no_clusters():
    """Calling without cluster_centroids preserves pre-FIX-5 behavior."""
    docs = read_folder(CORPUS_PATH)
    result = detect_forbidden_and_signature(docs, top_forbidden=25, top_signature=25)
    assert result["forbidden_irrelevant"] == []
    assert len(result["forbidden_words"]) > 0
    # No semantic filter ran, so min_cluster_distance should NOT appear on entries.
    for entry in result["forbidden_words"]:
        assert "min_cluster_distance" not in entry
    assert result["stats"].get("semantic_filter_applied") is False


def test_schema_has_relevance_threshold():
    docs = [
        _make_doc("a", "The founder ships the brand system quickly. " * 5),
        _make_doc("b", "Content infrastructure matters. The system ships content every day. " * 5),
    ]
    result = detect_forbidden_and_signature(docs, min_brand_count=2)
    assert "relevance_threshold" in result
    assert 0.0 < result["relevance_threshold"] < 1.0
    assert "forbidden_irrelevant" in result
