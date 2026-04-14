# Changelog

All notable changes to media-tsunami are documented here.

## [0.1.1] — 2026-04-14

Thin-corpus hygiene. Caught during a fresh-install test run on stripe.com.

### Fixed

- **Forbidden list noise on thin corpora.** Added `_GENERIC_NOISE` set in `forbidden_detector.py` to exclude topically-empty English words (numbers: `one, two, three…`; generic adverbs: `also, still, even`; time/amount noise: `time, years, part, way`; filler verbs: `made, get, put, take`; filler adjectives: `new, old, big, small`) from forbidden-candidate consideration. These had been leaking into the top of the forbidden list when the brand corpus was small enough that their suppression-ratio vs. wikitext baseline spiked. Does NOT affect `_STOPWORDS` — signature detection and cluster analysis still see these tokens, which is correct behavior.
- **Degenerate clusters.** Added `_MIN_CLUSTER_SIZE = 4` to `vocabulary_clusterer.py`. Previously a 2-token "cluster" like `integrating, integration` could surface as a vocabulary territory.
- **Thin-corpus warning.** CLI now prints a yellow warning when `word_count < 3,000`. CLAUDE.md caps the forbidden list at 5 (vs. 15) on thin corpora to reduce false-positive surface area. Pipeline result carries a `thin_corpus: bool` flag so downstream tools can branch.

### Unchanged

All core signals (cadence, signature vocabulary, tone classification, exemplar selection) are unchanged — the v0.1.0 test suite passes without modification, and the WhyStrohm fingerprint is byte-identical apart from the CLAUDE.md forbidden-list section.

## [0.1.0] — 2026-04-14

Initial release. Text-only brand voice fingerprint.

### Added

- **Folder reader** — recursively reads `.md` / `.html` / `.txt` from a folder into a normalized corpus.
- **Web scraper** — polite same-origin crawler with priority-ordered frontier, robots.txt respect, and a content-density filter that drops breadcrumb/nav/link-soup blocks.
- **Cadence analyzer** — sentence-length distribution, pronoun ratios, punctuation density, question/exclamation/fragment rates.
- **Voice fingerprinter** — sentence embeddings via `all-MiniLM-L6-v2`, corpus centroid, closest/farthest sentence selection with unusable-sentence filter (no `\n\n` artifacts, no 60+ token paragraphs).
- **Vocabulary clusterer** — TF-IDF + k-means on MiniLM token embeddings, contraction pre-filter, post-hoc drop of clusters that are >60% spaCy stopwords.
- **Forbidden / signature detector** — brand corpus vs wikitext-2 baseline, with optional semantic-relevance filter against cluster centroids that separates stylistic avoidance from topical noise.
- **Tone classifier** — 8-label heuristic rule table (direct, punchy, conversational, formal, energetic, instructive, analytical, authoritative) with confidence and rationale.
- **Writers** — `voice-fingerprint.json` (raw signals), `brand-config.json` (machine-readable rules), `CLAUDE.md` (LLM system prompt).
- **CLI** — `tsunami --url <site>` or `tsunami --folder <path>` — one command, three outputs.
- **Full pipeline** — end-to-end in ~3s on a 15K-word corpus, zero paid API calls.

### Test suite

59 tests across 7 modules. All green.

### Known limitations

- MiniLM token-level embeddings conflate semantic domain with stylistic avoidance. For media-adjacent brands, domain-topical words may leak into the forbidden list. Threshold is exposed as a tuning parameter.
- Static HTML only — sites requiring JavaScript to render body content are not yet supported.
- English only.
