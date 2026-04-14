"""GATE (step 4): run the engine end-to-end on a local corpus and print the three gate artifacts.

Usage:
    python scripts/gate_check.py [corpus_path]

Default corpus: ./test_corpus/whystrohm/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from tsunami.inputs.folder_reader import read_folder
from tsunami.engine.voice_fingerprinter import fingerprint_voice, render_length_histogram


def main():
    corpus_path = Path(sys.argv[1] if len(sys.argv) > 1 else "test_corpus/whystrohm")
    if not corpus_path.exists():
        print(f"ERROR: corpus path does not exist: {corpus_path}")
        sys.exit(1)

    print(f"Reading corpus: {corpus_path}")
    t0 = time.time()
    docs = read_folder(corpus_path)
    t_read = time.time() - t0
    print(f"  {len(docs)} documents loaded in {t_read*1000:.0f}ms")

    print("\nFingerprinting voice (loading MiniLM, embedding sentences)...")
    t0 = time.time()
    fp = fingerprint_voice(docs, top_n_vocab=20, sample_size=5)
    t_fp = time.time() - t0
    print(f"  Done in {t_fp:.2f}s\n")

    print("=" * 72)
    print("GATE ARTIFACT 1 — Sentence-length histogram")
    print("=" * 72)
    print(f"  sentences analyzed: {fp['sentence_count']}")
    print(f"  words in corpus:    {fp['word_count']}")
    cadence = fp["cadence"]
    print(f"  mean length:        {cadence['mean_length']:.1f} tokens")
    print(f"  std dev:            {cadence['std_length']:.1f}")
    print(f"  p25 / p50 / p75:    {cadence['p25_length']:.0f} / {cadence['p50_length']:.0f} / {cadence['p75_length']:.0f}")
    print(f"  fragment rate:      {cadence['fragment_rate']:.1%}")
    print()
    print(render_length_histogram(cadence["sentence_lengths"], bins=12, width=40))

    print()
    print("=" * 72)
    print("GATE ARTIFACT 2 — Top 20 vocabulary tokens")
    print("=" * 72)
    for rank, (tok, count) in enumerate(fp["top_vocab"], 1):
        print(f"  {rank:>2}. {tok:<20} {count:>4}×")

    print()
    print("=" * 72)
    print("GATE ARTIFACT 3 — Centroid distance for 5 sample sentences")
    print("=" * 72)
    print("\nClosest to centroid (most voice-representative):")
    for i, s in enumerate(fp["closest_to_centroid"], 1):
        text = s["text"].replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        print(f"  [{i}] dist={s['distance']:.4f}")
        print(f"      {text}")

    print("\nFarthest from centroid (outliers — topical drift or off-voice):")
    for i, s in enumerate(fp["farthest_from_centroid"], 1):
        text = s["text"].replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        print(f"  [{i}] dist={s['distance']:.4f}")
        print(f"      {text}")

    print()
    print("=" * 72)
    print("Pronoun signature")
    print("=" * 72)
    print(f"  1st-person ratio: {cadence['pronoun_ratio_first']:.1%}")
    print(f"  2nd-person ratio: {cadence['pronoun_ratio_second']:.1%}")
    print(f"  punct density:    {cadence['punct_density']:.1f} per 100 words")
    print(f"  question rate:    {cadence['question_rate']:.1%}")
    print(f"  exclamation rate: {cadence['exclamation_rate']:.1%}")

    print()
    print("=" * 72)
    print("Runtime")
    print("=" * 72)
    print(f"  corpus read:      {t_read*1000:>6.0f}ms")
    print(f"  fingerprint:      {t_fp*1000:>6.0f}ms")
    print(f"  total:            {(t_read+t_fp)*1000:>6.0f}ms")
    print(f"  embedding model:  {fp['embed_model']} ({fp['embedding_dim']}d)")


if __name__ == "__main__":
    main()
