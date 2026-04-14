# Security policy

## Scope

media-tsunami runs entirely locally. It does not transmit scraped content, extracted voice profiles, or any other data to external services outside of:

- Downloading the `all-MiniLM-L6-v2` model from Hugging Face on first run (one-time, ~80 MB)
- Downloading the wikitext-2 dataset from Hugging Face on first run (one-time, ~15 MB)
- Any URL the user explicitly passes to `tsunami --url <url>` (fetched for analysis)

No API keys required. No telemetry. No analytics.

## Reporting a vulnerability

If you discover a security issue, please do NOT open a public GitHub issue.

Email: **security@whystrohm.com**

Include:
- A description of the vulnerability
- Reproduction steps if possible
- The affected version (`tsunami --version`)

I aim to respond within 72 hours and publish a fix within 14 days for confirmed issues.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |
