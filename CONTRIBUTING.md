# Contributing to media-tsunami

Thanks for considering a contribution. This project is small, focused, and intentional — I'd rather ship a tight v0.x than a sprawling v1.

## Scope

**In scope:**
- Bug fixes in the extraction pipeline (tokenization, sentence splitting, embedding edge cases)
- Quality improvements to any of the 6 engine modules (cadence, voice, vocab clusters, forbidden, tone, pipeline)
- Better defaults for thresholds that aren't tunable today
- Additional test coverage
- Docs + example outputs for brands different from WhyStrohm
- Scraper robustness (particular sites that fail)

**Out of scope for v0.1.x:**
- Visual fingerprinting (palette, typography) — coming in v0.2
- Motion / video fingerprinting — coming in v0.3
- Non-English language support — tracked but not actively worked on
- Cloud-hosted / web-UI versions — the engine stays a CLI
- Paid-API integrations (OpenAI, Anthropic) — core stays local-only

If you're not sure, open an issue before writing code.

## Development setup

```bash
git clone https://github.com/whystrohm/media-tsunami
cd media-tsunami
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

Run tests:
```bash
pytest tests/ -v
```

Regenerate the WhyStrohm fingerprint (sanity check that your changes didn't regress core quality):
```bash
tsunami --folder test_corpus/whystrohm --brand-name WhyStrohm -o ./fingerprint-check
```

## Pull request checklist

Before opening a PR:

- [ ] `pytest tests/` passes (all 59+ tests)
- [ ] New behavior has a test
- [ ] No paid-API imports (no `openai`, no `anthropic`, no provider SDKs)
- [ ] No new runtime dependencies without discussion (check pyproject.toml)
- [ ] No home-directory paths or PII in example outputs or tests
- [ ] CHANGELOG.md has an entry under the next unreleased version
- [ ] Runs end-to-end on the whystrohm test corpus in under 10s

## Coding principles

Read a few existing engine modules before writing new ones. The house style:

- **Plain dicts over dataclasses.** Only use a class if you need identity or mutation patterns.
- **`lru_cache` for heavy singletons** (spaCy pipeline, embedding model, baseline corpus).
- **No premature abstraction.** If two files need the same filter, inline both copies until a third shows up — then extract.
- **No configuration systems.** Module-level constants over user config. Only take a parameter if a test has already needed to vary it.
- **Tests live in `tests/`**, mirror the src path. Each engine module gets its own test file.

## What I'll merge fast

Clear, small PRs that fix one thing or add one capability, with a test, and a CHANGELOG line. A 50-line PR with 5 lines of tests beats a 500-line PR with none.

## What I'll push back on

- Adding pandas, lightning, wandb, or other heavy ML framework dependencies
- Refactoring that "improves architecture" without improving output quality
- New features gated on closed APIs
- Changes to `_STOPWORDS` — that set is load-bearing for forbidden detection; propose additions in an issue first

## License

By submitting a PR, you agree your contribution is licensed under the MIT license (same as the project).
