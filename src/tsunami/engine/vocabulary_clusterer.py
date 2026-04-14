"""Vocabulary clusterer: group the brand's most-used words into semantic clusters."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

from tsunami.engine.voice_fingerprinter import _STOPWORDS, _get_embedder
from tsunami.outputs._display_filter import is_display_noise

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")
_TOKEN_PATTERN = r"[a-z][a-z'\-]+"
_MIN_LEN = 3
_LABEL_TOP_K = 3
_STOPWORD_CLUSTER_FRACTION = 0.6
_STOPWORD_CHECK_TOP_K = 8

_EMPTY_RESULT = {
    "clusters": [],
    "n_clusters": 0,
    "dropped_clusters": 0,
    "total_unique_tokens": 0,
    "total_tokens_clustered": 0,
    "centroid_embeddings": np.zeros((0, 384), dtype=np.float32),
}


def cluster_vocabulary(
    docs: list[dict],
    n_clusters: int = 6,
    top_n_per_cluster: int = 12,
    min_token_freq: int = 3,
) -> dict:
    """Build semantic clusters of the brand's vocabulary.

    Input: list of Document dicts {"path", "title", "text"} from folder_reader.
    Returns a dict with clusters list + corpus-level stats.
    """
    if not docs:
        return dict(_EMPTY_RESULT)

    doc_texts = [d.get("text", "") for d in docs]
    full_text = "\n\n".join(doc_texts).lower()

    # Tokenize + filter stopwords + min length.
    all_tokens = _TOKEN_RE.findall(full_text)
    counts = Counter(t for t in all_tokens if t not in _STOPWORDS and len(t) >= _MIN_LEN)

    # Filter by frequency.
    vocab = [tok for tok, c in counts.items() if c >= min_token_freq]

    # Pre-cluster filter: drop contractions, URL fragments, and ultra-short tokens so
    # function words like "that's"/"it's"/"you're" never reach KMeans centroids.
    vocab = [tok for tok in vocab if not is_display_noise(tok)]

    if not vocab:
        return dict(_EMPTY_RESULT)

    total_unique = len(vocab)

    # Clamp n_clusters to available vocab size.
    effective_k = max(1, min(n_clusters, total_unique))

    # TF-IDF weights: sum across documents.
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
    )
    tfidf_matrix = vectorizer.fit_transform(doc_texts)  # shape (n_docs, n_vocab)
    tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).ravel()  # shape (n_vocab,)
    weight_by_token = {tok: float(tfidf_sum[i]) for i, tok in enumerate(vocab)}

    # Embed all tokens in one batch (normalized -> cosine = dot).
    embedder = _get_embedder()
    embeddings = embedder.encode(
        vocab,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Cluster.
    kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Build per-cluster outputs.
    clusters: list[dict] = []
    for cid in range(effective_k):
        member_idx = np.where(labels == cid)[0]
        if len(member_idx) == 0:
            clusters.append({
                "id": cid,
                "label": "",
                "top_tokens": [],
                "size": 0,
                "centroid_token": "",
                "centroid_embedding": [],
            })
            continue

        member_tokens = [vocab[i] for i in member_idx]
        member_embeds = embeddings[member_idx]

        # Sort members by TF-IDF weight descending.
        member_weighted = sorted(
            ((tok, weight_by_token[tok]) for tok in member_tokens),
            key=lambda x: x[1],
            reverse=True,
        )
        top_tokens = member_weighted[:top_n_per_cluster]

        # Label from top 3 tokens.
        label = ", ".join(tok for tok, _ in top_tokens[:_LABEL_TOP_K])

        # Centroid-closest token (cosine, embeddings already normalized).
        mean_vec = member_embeds.mean(axis=0)
        mean_norm = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
        sims = member_embeds @ mean_norm
        exemplar_token = member_tokens[int(np.argmax(sims))]

        clusters.append({
            "id": cid,
            "label": label,
            "top_tokens": top_tokens,
            "size": int(len(member_idx)),
            "centroid_token": exemplar_token,
            # Normalized centroid embedding — JSON-serializable plain list.
            "centroid_embedding": mean_norm.astype(np.float32).tolist(),
        })

    # Post-cluster drop: filter out clusters whose top tokens are mostly spaCy stopwords.
    surviving: list[dict] = []
    n_dropped = 0
    for c in clusters:
        top_slice = [tok for tok, _ in c["top_tokens"][:_STOPWORD_CHECK_TOP_K]]
        if not top_slice:
            surviving.append(c)
            continue
        sw_count = sum(1 for tok in top_slice if tok in SPACY_STOP_WORDS)
        sw_fraction = sw_count / len(top_slice)
        if sw_fraction > _STOPWORD_CLUSTER_FRACTION:
            n_dropped += 1
            continue
        surviving.append(c)

    # Stacked centroid matrix over surviving clusters — convenience handle for
    # downstream consumers (forbidden_detector semantic filter).
    surviving_centroids = [
        c["centroid_embedding"] for c in surviving if c.get("centroid_embedding")
    ]
    if surviving_centroids:
        centroid_embeddings = np.asarray(surviving_centroids, dtype=np.float32)
    else:
        centroid_embeddings = np.zeros((0, 384), dtype=np.float32)

    return {
        "clusters": surviving,
        "n_clusters": len(surviving),
        "dropped_clusters": n_dropped,
        "total_unique_tokens": total_unique,
        "total_tokens_clustered": int(sum(c["size"] for c in surviving)),
        "centroid_embeddings": centroid_embeddings,
    }
