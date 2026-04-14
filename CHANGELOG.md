# Changelog

All notable changes to media-tsunami are documented here.

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
