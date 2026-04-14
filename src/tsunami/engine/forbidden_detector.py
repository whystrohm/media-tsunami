"""Forbidden + signature word detector.

Compares the brand corpus against a fixed generic-English baseline (wikitext-2)
to surface two lists:
  - signature_words: tokens the brand uses far more than generic English
  - forbidden_words: common generic-English tokens the brand avoids

The baseline is built once on first run and cached as JSON under .cache/.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from tsunami.engine.voice_fingerprinter import _STOPWORDS, _get_embedder

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")
_MIN_LEN = 3
_BASELINE_NAME = "wikitext-2"
_BASELINE_TOP_N = 20_000
_GENERIC_COMMON_POOL = 500  # how many of the top baseline tokens to scan for forbidden
_NOISE_FLOOR = 5  # min count in baseline OR brand to keep a token in signature list
_FORBIDDEN_CANDIDATE_POOL = 100  # top-N candidates that go through the semantic filter
# NOTE: 0.55 tuned against whystrohm.com. Higher thresholds (0.65+) leave domain-topical
# noise (film/album/music) in the forbidden list because MiniLM token embeddings conflate
# semantic domain with stylistic choice. Lower thresholds (<0.5) start removing genuine
# stylistic avoidance words (however/although/several). 0.55 is a compromise that catches
# the worst encyclopedic noise (geographic, administrative) without killing real signal.
_DEFAULT_RELEVANCE_THRESHOLD = 0.55

_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"
_BASELINE_PATH = _CACHE_DIR / "wikitext2_freqs.json"


def _tokenize(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) >= _MIN_LEN
    ]


def _build_baseline() -> dict:
    """Download wikitext-2, tokenize, build rate-per-million frequencies. Cached to JSON."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - dep is in pyproject
        raise RuntimeError(
            "datasets package is required to build the wikitext-2 baseline. "
            "Install with: pip install datasets"
        ) from e

    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load wikitext-2 baseline ({e}). "
            f"Check network connectivity or pre-populate {_BASELINE_PATH}."
        ) from e

    counter: Counter[str] = Counter()
    total = 0
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        toks = _tokenize(text)
        counter.update(toks)
        total += len(toks)

    if total == 0:
        raise RuntimeError("wikitext-2 produced zero tokens after filtering — aborting.")

    most_common = counter.most_common(_BASELINE_TOP_N)
    rates = {tok: (count / total) * 1_000_000 for tok, count in most_common}
    counts = {tok: count for tok, count in most_common}
    payload = {"total_tokens": total, "rates": rates, "counts": counts}

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _BASELINE_PATH.write_text(json.dumps(payload))
    return payload


@lru_cache(maxsize=1)
def _get_baseline() -> dict:
    if _BASELINE_PATH.exists():
        try:
            return json.loads(_BASELINE_PATH.read_text())
        except json.JSONDecodeError:
            pass  # fall through to rebuild
    return _build_baseline()


def _empty_result(relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD) -> dict:
    return {
        "signature_words": [],
        "forbidden_words": [],
        "forbidden_irrelevant": [],
        "baseline": _BASELINE_NAME,
        "brand_total_tokens": 0,
        "stats": {"note": "empty input"},
        "relevance_threshold": relevance_threshold,
    }


def detect_forbidden_and_signature(
    docs: list[dict],
    min_brand_count: int = 3,
    top_forbidden: int = 25,
    top_signature: int = 25,
    cluster_centroids: "np.ndarray | None" = None,
    relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
) -> dict:
    """Compare brand corpus to the generic-English baseline (wikitext-2).

    Returns signature_words (over-represented in brand) + forbidden_words
    (common in generic English but rare/absent in brand).

    If ``cluster_centroids`` is provided (shape ``(n_clusters, 384)``, normalized
    MiniLM embeddings), candidate forbidden words are partitioned by cosine
    distance to the nearest cluster centroid:

      - min distance <= relevance_threshold  -> forbidden_words (stylistic avoid)
      - min distance  > relevance_threshold  -> forbidden_irrelevant (topical noise)

    When ``cluster_centroids`` is None, ``forbidden_irrelevant`` is ``[]`` and
    ``forbidden_words`` is the raw top-suppression list (backward compat).
    """
    if not docs:
        return _empty_result(relevance_threshold)

    full_text = "\n\n".join(d.get("text", "") for d in docs)
    brand_tokens = _tokenize(full_text)
    if not brand_tokens:
        return _empty_result(relevance_threshold)

    brand_counts = Counter(brand_tokens)
    brand_total = len(brand_tokens)
    brand_rate = {tok: (c / brand_total) * 1_000_000 for tok, c in brand_counts.items()}

    baseline = _get_baseline()
    base_rates: dict[str, float] = baseline["rates"]
    base_counts: dict[str, int] = baseline.get("counts", {})

    # --- Signature words: over-represented in brand ---
    sig_candidates = []
    for tok, bcount in brand_counts.items():
        if bcount < min_brand_count:
            continue
        b_rate = brand_rate[tok]
        base_rate = base_rates.get(tok, 0.0)
        base_count = base_counts.get(tok, 0)
        # Noise floor: skip tokens that are rare in both corpora (typos / rare names).
        if base_count < _NOISE_FLOOR and bcount < _NOISE_FLOOR:
            continue
        ratio = b_rate / (base_rate + 1.0)
        sig_candidates.append({
            "token": tok,
            "brand_rate": b_rate,
            "baseline_rate": base_rate,
            "ratio": ratio,
            "brand_count": bcount,
        })
    sig_candidates.sort(key=lambda x: x["ratio"], reverse=True)
    signature_words = sig_candidates[:top_signature]

    # --- Forbidden words: common in generic English, under-represented in brand ---
    # Rank baseline tokens by rate, take top pool, then sort by suppression.
    top_generic = sorted(base_rates.items(), key=lambda kv: kv[1], reverse=True)[:_GENERIC_COMMON_POOL]
    forbidden_candidates = []
    for tok, base_rate in top_generic:
        bcount = brand_counts.get(tok, 0)
        if bcount >= min_brand_count:
            continue  # brand actually uses this — not forbidden
        b_rate = brand_rate.get(tok, 0.0)
        suppression = base_rate / (b_rate + 1.0)
        forbidden_candidates.append({
            "token": tok,
            "brand_rate": b_rate,
            "baseline_rate": base_rate,
            "ratio": b_rate / (base_rate + 1e-6),
            "brand_count": bcount,
            "_suppression": suppression,
        })
    forbidden_candidates.sort(key=lambda x: x["_suppression"], reverse=True)

    def _strip(entry: dict) -> dict:
        return {k: v for k, v in entry.items() if k != "_suppression"}

    forbidden_words: list[dict]
    forbidden_irrelevant: list[dict] = []

    if cluster_centroids is None or (
        isinstance(cluster_centroids, np.ndarray) and cluster_centroids.shape[0] == 0
    ):
        # Backward-compat path: no semantic filter.
        forbidden_words = [_strip(entry) for entry in forbidden_candidates[:top_forbidden]]
    else:
        # Semantic filter — embed top-N candidates and partition by min cluster distance.
        pool = forbidden_candidates[:_FORBIDDEN_CANDIDATE_POOL]
        if not pool:
            forbidden_words = []
        else:
            centroids = np.asarray(cluster_centroids, dtype=np.float32)
            candidate_tokens = [entry["token"] for entry in pool]
            embedder = _get_embedder()
            cand_embeds = embedder.encode(
                candidate_tokens,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            # Cosine similarity against every centroid; distance = 1 - max_sim.
            sims = cand_embeds @ centroids.T  # shape (n_candidates, n_clusters)
            max_sims = sims.max(axis=1)
            min_dists = 1.0 - max_sims

            stylistic: list[dict] = []
            irrelevant: list[dict] = []
            for entry, dist in zip(pool, min_dists):
                record = _strip(entry)
                record["min_cluster_distance"] = float(dist)
                if float(dist) <= relevance_threshold:
                    stylistic.append(record)
                else:
                    irrelevant.append(record)

            forbidden_words = stylistic[:top_forbidden]
            forbidden_irrelevant = irrelevant

    stats = {
        "brand_unique_tokens": len(brand_counts),
        "baseline_unique_tokens": len(base_rates),
        "baseline_total_tokens": baseline.get("total_tokens", 0),
        "min_brand_count": min_brand_count,
        "semantic_filter_applied": cluster_centroids is not None
        and isinstance(cluster_centroids, np.ndarray)
        and cluster_centroids.shape[0] > 0,
    }

    return {
        "signature_words": signature_words,
        "forbidden_words": forbidden_words,
        "forbidden_irrelevant": forbidden_irrelevant,
        "baseline": _BASELINE_NAME,
        "brand_total_tokens": brand_total,
        "stats": stats,
        "relevance_threshold": relevance_threshold,
    }
