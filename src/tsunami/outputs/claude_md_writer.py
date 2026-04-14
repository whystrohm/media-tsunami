"""Writes CLAUDE.md — the LLM system prompt that makes any model write like the brand.

This is the crown-jewel output. It should be prescriptive, concrete, and usable
as-is when loaded into a Claude / LLM system prompt, memory, or CLAUDE.md.
"""
from __future__ import annotations

from pathlib import Path

from tsunami.outputs._display_filter import filter_signature_tokens


def _describe_pronoun_policy(first: float, second: float) -> str:
    """Convert first/second person ratios to a prescriptive one-liner."""
    lines = []
    if second > 0.02:
        lines.append("Address the reader directly. Use 'you' and 'your' liberally.")
    elif second > 0.008:
        lines.append("Use 'you' / 'your' in moderation — every 3-5 sentences.")
    else:
        lines.append("Avoid second-person address. Write in the third person or passive voice.")

    if first > 0.015:
        lines.append("Use first-person ('I', 'we', 'our') freely — this is founder-voice territory.")
    elif first > 0.005:
        lines.append("First-person is allowed but sparingly — save 'I' and 'we' for emphasis.")
    else:
        lines.append("Minimize first-person pronouns. The brand speaks about things, not about itself.")
    return "\n".join("- " + s for s in lines)


def _tone_directive(tone: dict) -> str:
    """Convert tone classification into a prescriptive directive."""
    primary = tone.get("primary_tone", "mixed")
    secondary = tone.get("secondary_tone")
    confidence = tone.get("confidence", 0.0)

    directives = {
        "direct": "Write with directness. Short declarations. Skip qualifiers. Make the claim.",
        "punchy": "Write punchy. Fragments are fine. One thought per sentence. Hit hard, move on.",
        "conversational": "Write like you're talking to one person. Contractions welcome. Plain language, not corporate.",
        "formal": "Write with formal register. Complete sentences. No contractions. Measured phrasing.",
        "energetic": "Write with energy. Vary sentence length dramatically. Use exclamation sparingly but deliberately.",
        "instructive": "Write to teach. Lead with the 'why'. Use questions to surface the reader's thinking.",
        "analytical": "Write analytically. Structured arguments. Sentences that carry weight. Few fragments.",
        "authoritative": "Write with authority. Declarative statements. Avoid hedging. Own the claim.",
    }

    lines = [directives.get(primary, f"Write in a {primary} voice.")]
    if secondary and secondary in directives:
        lines.append(f"Blend in {secondary} elements: {directives[secondary]}")
    if confidence < 0.6:
        lines.append(f"(Tone signal is mixed — confidence {confidence:.0%}. Default to the primary directive above.)")
    return "\n".join(lines)


def _cadence_block(cadence: dict) -> str:
    """Concrete cadence rules with numbers."""
    mean = cadence["mean_length"]
    std = cadence["std_length"]
    target_lo = max(3, int(mean - std))
    target_hi = max(target_lo + 2, int(mean + std))
    fragment = cadence["fragment_rate"]
    punct = cadence["punct_density"]

    lines = [
        f"- **Target sentence length: {target_lo}–{target_hi} tokens** (mean is {mean:.1f}).",
        f"- **Fragment rate: {fragment:.0%}.** Roughly {'every 3rd' if fragment > 0.3 else 'every 4th-5th' if fragment > 0.2 else 'occasional'} sentence should be under 5 tokens.",
        f"- **Punctuation density: {punct:.0f} marks per 100 words** ({'punctuation-heavy' if punct > 18 else 'moderate' if punct > 12 else 'sparse'}).",
    ]
    if cadence["question_rate"] > 0.03:
        lines.append(f"- Use questions — about {cadence['question_rate']:.0%} of sentences are questions.")
    if cadence["exclamation_rate"] > 0.02:
        lines.append(f"- Exclamations appear in {cadence['exclamation_rate']:.0%} of sentences — use them deliberately, not for emphasis.")
    return "\n".join(lines)


def build_claude_md(fingerprint: dict) -> str:
    """Build the full CLAUDE.md content as a single string."""
    if fingerprint.get("empty"):
        return (
            f"# Brand voice profile: {fingerprint.get('brand_name', 'Unknown')}\n\n"
            f"No content was extracted. Rerun the engine with a corpus.\n"
        )

    name = fingerprint["brand_name"]
    source = fingerprint["source"]
    gen_at = fingerprint["generated_at"]
    cadence = fingerprint["cadence"]
    tone = fingerprint["tone"]
    signature = fingerprint["signature_words"]
    forbidden = fingerprint["forbidden_words"]
    clusters = fingerprint["clusters"]
    exemplars = fingerprint["voice"]["closest_to_centroid"]

    signature_tokens = filter_signature_tokens(signature, limit=15)
    # On thin corpora, cap forbidden list (5 instead of 15) — suppression-ratio
    # signal is too noisy with limited data and risks surfacing false positives.
    forbidden_limit = 5 if fingerprint.get("thin_corpus") else 15
    forbidden_tokens = [w["token"] for w in forbidden[:forbidden_limit]]

    cluster_lines = []
    for c in clusters["clusters"]:
        top = ", ".join(t for t, _ in c["top_tokens"][:6])
        cluster_lines.append(f"- **{c['label']}** — {top}")

    exemplar_lines = []
    for s in exemplars[:6]:
        text = s["text"].strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:217] + "..."
        exemplar_lines.append(f"> {text}")

    md = f"""# Brand voice: {name}

> System prompt for writing in {name}'s voice.
> Generated from {source} on {gen_at} by [media-tsunami](https://github.com/whystrohm/media-tsunami).

Load this file into Claude's system prompt, a CLAUDE.md, or any LLM memory. It encodes {name}'s voice as executable rules. Follow them exactly when writing on behalf of this brand.

---

## Core directive

{_tone_directive(tone)}

Primary tone: **{tone["primary_tone"]}**{f" · Secondary: *{tone['secondary_tone']}*" if tone.get("secondary_tone") else ""} · Confidence: {tone["confidence"]:.0%}

---

## Cadence rules

{_cadence_block(cadence)}

---

## Pronoun policy

{_describe_pronoun_policy(cadence["pronoun_ratio_first"], cadence["pronoun_ratio_second"])}

---

## Signature vocabulary (prefer these)

These words are this brand's signature — they appear far more often here than in generic English. Reach for them:

{", ".join(f"`{t}`" for t in signature_tokens)}

---

## Forbidden vocabulary (NEVER use)

These are common English words this brand systematically avoids. Using them breaks voice. Do not use them. If a synonym is needed, rephrase the sentence.

{", ".join(f"`{t}`" for t in forbidden_tokens)}

---

## Vocabulary territory

The brand's language clusters around these themes. Stay inside these clusters — drift outside and the voice fades:

{chr(10).join(cluster_lines)}

---

## Voice-representative examples

These sentences are the most "on-brand" — they sit closest to the brand's semantic center. Pattern-match on their rhythm, word choice, and posture:

{chr(10).join(exemplar_lines)}

---

## Pre-flight checklist (run before sending any output)

1. **Length check** — is the mean sentence length inside the target range? Count tokens.
2. **Fragment check** — is roughly {cadence["fragment_rate"]:.0%} of output under 5 tokens?
3. **Forbidden scan** — does the draft contain any forbidden word? If yes, rewrite.
4. **Signature check** — does the draft use at least 2-3 signature words? If zero, rewrite.
5. **Tone match** — does it read as {tone["primary_tone"]}? If not, cut the hedge words and tighten.

---

*This profile was generated by [media-tsunami](https://github.com/whystrohm/media-tsunami). Re-run the engine after meaningful new content (10+ new articles) to keep the voice profile fresh.*
"""
    return md


def write_claude_md(fingerprint: dict, output_path: str | Path) -> Path:
    """Write CLAUDE.md to disk."""
    content = build_claude_md(fingerprint)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path.resolve()
