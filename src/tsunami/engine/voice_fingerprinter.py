"""Voice fingerprinting — sentence embeddings, centroid distance, vocab signature."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from tsunami.engine.cadence_analyzer import analyze_corpus

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "than", "too", "very", "can", "will", "just",
    "don", "should", "now", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "should", "may", "might", "must", "shall", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "am", "what", "which", "who", "whom",
    "whose", "because", "while", "until", "s", "t", "re", "ve", "ll", "d", "m",
}

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")


@lru_cache(maxsize=1)
def _get_sentencizer():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def _split_sentences(text: str) -> list[str]:
    nlp = _get_sentencizer()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 0]


def _is_usable_sentence(s: str) -> bool:
    """Filter out junk sentences from the embedding step.

    Drops:
      - fragments below 3 whitespace-split tokens
      - sentences containing '\\n\\n' (paragraph break escaped into sentence body =
        sentencizer failure on a block with no terminal punctuation)
      - sentences longer than 60 whitespace-split tokens (paragraphs masquerading
        as sentences)
      - sentences longer than 500 characters (another paragraph-ish guard)
    """
    tokens = s.split()
    if len(tokens) < 3:
        return False
    if "\n\n" in s:
        return False
    if len(tokens) > 60:
        return False
    if len(s) > 500:
        return False
    return True


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _top_vocab(text: str, n: int = 20, min_len: int = 3) -> list[tuple[str, int]]:
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) >= min_len]
    return Counter(tokens).most_common(n)


def fingerprint_voice(docs: list[dict], top_n_vocab: int = 20, sample_size: int = 5) -> dict:
    """Compute the voice fingerprint over a corpus of Documents.

    Returns a dict with:
      sentence_count, word_count
      cadence: dict (from analyze_corpus)
      top_vocab: list[(token, count)]
      centroid_distances: np.ndarray, cosine distances of each sentence to corpus centroid
      closest_to_centroid: list[dict {text, distance}]  (the most voice-representative sentences)
      farthest_from_centroid: list[dict {text, distance}] (outliers)
      embedding_dim: int
      embed_model: str
    """
    full_text = "\n\n".join(d["text"] for d in docs)
    sentences = _split_sentences(full_text)
    # Filter unusable sentences from embedding step — see _is_usable_sentence.
    sentences = [s for s in sentences if _is_usable_sentence(s)]

    cadence = analyze_corpus(docs)

    if len(sentences) == 0:
        return {
            "sentence_count": 0,
            "word_count": cadence["word_count"],
            "cadence": cadence,
            "top_vocab": [],
            "centroid_distances": np.array([]),
            "closest_to_centroid": [],
            "farthest_from_centroid": [],
            "embedding_dim": 0,
            "embed_model": EMBED_MODEL_NAME,
        }

    model = _get_embedder()
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    # Cosine distance (since embeddings are normalized): 1 - dot product.
    distances = 1.0 - embeddings @ centroid

    order_asc = np.argsort(distances)
    closest_idx = order_asc[:sample_size]
    farthest_idx = order_asc[-sample_size:][::-1]

    closest = [{"text": sentences[i], "distance": float(distances[i])} for i in closest_idx]
    farthest = [{"text": sentences[i], "distance": float(distances[i])} for i in farthest_idx]

    return {
        "sentence_count": len(sentences),
        "word_count": cadence["word_count"],
        "cadence": cadence,
        "top_vocab": _top_vocab(full_text, n=top_n_vocab),
        "centroid_distances": distances,
        "closest_to_centroid": closest,
        "farthest_from_centroid": farthest,
        "embedding_dim": embeddings.shape[1],
        "embed_model": EMBED_MODEL_NAME,
    }


def render_length_histogram(sentence_lengths: list[int], bins: int = 10, width: int = 40) -> str:
    """ASCII histogram of sentence lengths."""
    if not sentence_lengths:
        return "(no sentences)"
    arr = np.array(sentence_lengths)
    hist, edges = np.histogram(arr, bins=bins)
    peak = hist.max()
    lines = []
    for i, count in enumerate(hist):
        lo, hi = int(edges[i]), int(edges[i + 1])
        bar = "█" * int((count / peak) * width) if peak else ""
        lines.append(f"  {lo:>3}-{hi:<3} | {bar} {count}")
    return "\n".join(lines)
