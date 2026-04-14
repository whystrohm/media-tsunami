"""Writes brand-config.json — machine-readable brand rules."""
from __future__ import annotations

import json
from pathlib import Path

from tsunami.outputs._display_filter import is_display_noise


def build_brand_config(fingerprint: dict) -> dict:
    """Build the brand-config.json structure from a full pipeline fingerprint."""
    if fingerprint.get("empty"):
        return {
            "schema_version": "0.1",
            "brand_name": fingerprint.get("brand_name", "Unknown"),
            "source": fingerprint.get("source", ""),
            "generated_at": fingerprint.get("generated_at", ""),
            "empty": True,
        }

    cadence = fingerprint["cadence"]
    tone = fingerprint["tone"]
    clusters = fingerprint["clusters"]

    mean = cadence["mean_length"]
    std = cadence["std_length"]
    target_lo = max(3, int(mean - std))
    target_hi = max(target_lo + 2, int(mean + std))

    filtered_signature = [w for w in fingerprint["signature_words"] if not is_display_noise(w["token"])]

    return {
        "schema_version": "0.1",
        "brand_name": fingerprint["brand_name"],
        "source": fingerprint["source"],
        "generated_at": fingerprint["generated_at"],
        "tone": {
            "primary": tone["primary_tone"],
            "secondary": tone["secondary_tone"],
            "confidence": round(tone["confidence"], 3),
            "uncertain": tone["uncertain"],
        },
        "cadence": {
            "mean_sentence_length": round(mean, 1),
            "target_range": [target_lo, target_hi],
            "fragment_rate": round(cadence["fragment_rate"], 3),
            "punct_density": round(cadence["punct_density"], 1),
            "first_person_ratio": round(cadence["pronoun_ratio_first"], 4),
            "second_person_ratio": round(cadence["pronoun_ratio_second"], 4),
            "question_rate": round(cadence["question_rate"], 3),
            "exclamation_rate": round(cadence["exclamation_rate"], 3),
        },
        "signature_words": [
            {"token": w["token"], "ratio": round(w["ratio"], 2), "count": w["brand_count"]}
            for w in filtered_signature[:20]
        ],
        "forbidden_words": [w["token"] for w in fingerprint["forbidden_words"][:20]],
        "vocabulary_clusters": [
            {
                "label": c["label"],
                "exemplar": c["centroid_token"],
                "top_tokens": [t for t, _ in c["top_tokens"][:8]],
                "size": c["size"],
            }
            for c in clusters["clusters"]
        ],
        "exemplar_sentences": [
            s["text"] for s in fingerprint["voice"]["closest_to_centroid"][:5]
        ],
        "baseline_corpus": fingerprint["baseline"],
        "embedding_model": fingerprint["voice"]["embed_model"],
    }


def write_brand_config(fingerprint: dict, output_path: str | Path) -> Path:
    """Serialize and write brand-config.json. Returns the absolute output path."""
    config = build_brand_config(fingerprint)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path.resolve()
