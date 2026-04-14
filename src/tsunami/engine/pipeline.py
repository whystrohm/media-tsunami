"""Top-level pipeline: docs → full brand fingerprint."""
from __future__ import annotations

import time

from tsunami.engine.cadence_analyzer import analyze_corpus
from tsunami.engine.forbidden_detector import detect_forbidden_and_signature
from tsunami.engine.tone_classifier import classify_tone
from tsunami.engine.vocabulary_clusterer import cluster_vocabulary
from tsunami.engine.voice_fingerprinter import fingerprint_voice


def run_pipeline(docs: list[dict], brand_name: str = "Unknown", source: str = "") -> dict:
    """Run the full engine over a document corpus.

    Returns an aggregated fingerprint dict combining cadence, vocab, tone,
    forbidden/signature words, clusters, and centroid-representative sentences.
    """
    if not docs:
        return {
            "brand_name": brand_name,
            "source": source,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "doc_count": 0,
            "empty": True,
        }

    t0 = time.time()
    cadence = analyze_corpus(docs)
    t_cadence = time.time() - t0

    t0 = time.time()
    voice = fingerprint_voice(docs, top_n_vocab=30, sample_size=8)
    t_voice = time.time() - t0

    t0 = time.time()
    clusters = cluster_vocabulary(docs, n_clusters=6, top_n_per_cluster=12)
    t_clusters = time.time() - t0

    t0 = time.time()
    centroid_embeddings = clusters.get("centroid_embeddings")
    forbidden_sig = detect_forbidden_and_signature(
        docs,
        top_forbidden=25,
        top_signature=25,
        cluster_centroids=centroid_embeddings,
    )
    t_forbidden = time.time() - t0

    t0 = time.time()
    tone = classify_tone(cadence, signature_words=forbidden_sig["signature_words"])
    t_tone = time.time() - t0

    return {
        "brand_name": brand_name,
        "source": source,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "doc_count": len(docs),
        "cadence": cadence,
        "voice": {
            "top_vocab": voice["top_vocab"],
            "closest_to_centroid": voice["closest_to_centroid"],
            "farthest_from_centroid": voice["farthest_from_centroid"],
            "embedding_dim": voice["embedding_dim"],
            "embed_model": voice["embed_model"],
        },
        "clusters": clusters,
        "signature_words": forbidden_sig["signature_words"],
        "forbidden_words": forbidden_sig["forbidden_words"],
        "forbidden_irrelevant": forbidden_sig.get("forbidden_irrelevant", []),
        "relevance_threshold": forbidden_sig.get("relevance_threshold"),
        "baseline": forbidden_sig["baseline"],
        "tone": tone,
        "timings_ms": {
            "cadence": round(t_cadence * 1000, 1),
            "voice": round(t_voice * 1000, 1),
            "clusters": round(t_clusters * 1000, 1),
            "forbidden": round(t_forbidden * 1000, 1),
            "tone": round(t_tone * 1000, 1),
        },
    }
