"""Command-line entrypoint for media-tsunami."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import click

from tsunami.engine.pipeline import run_pipeline
from tsunami.inputs.folder_reader import read_folder
from tsunami.inputs.web_scraper import scrape_url
from tsunami.outputs.brand_config_writer import write_brand_config
from tsunami.outputs.claude_md_writer import write_claude_md
from tsunami.outputs.voice_fingerprint_writer import write_voice_fingerprint


@click.command()
@click.option("--url", help="URL to scan (alternative to --folder).")
@click.option("--folder", type=click.Path(exists=True, file_okay=False), help="Folder of md/html/txt files.")
@click.option("--output-dir", "-o", default="./fingerprint", help="Where to write outputs.")
@click.option("--brand-name", default=None, help="Brand name for the output files (auto-derived if omitted).")
@click.option("--max-pages", default=20, show_default=True, help="Max pages to crawl for URL input.")
def main(url, folder, output_dir, brand_name, max_pages):
    """Extract a brand voice fingerprint from a URL or folder."""
    if not url and not folder:
        click.echo("ERROR: provide --url or --folder", err=True)
        sys.exit(1)
    if url and folder:
        click.echo("ERROR: use --url OR --folder, not both", err=True)
        sys.exit(1)

    t_start = time.time()

    if url:
        if not brand_name:
            brand_name = urlparse(url).netloc.replace("www.", "") or "Unknown"
        click.echo(f"Scraping {url} (max {max_pages} pages)...")
        docs = scrape_url(url, max_pages=max_pages)
        source = url
    else:
        folder_path = Path(folder).resolve()
        if not brand_name:
            brand_name = folder_path.name
        click.echo(f"Reading folder {folder_path}...")
        docs = read_folder(folder_path)
        # Only store folder name in outputs — never leak user home directories.
        source = f"folder:{folder_path.name}"

    click.echo(f"  {len(docs)} documents.")
    if not docs:
        click.echo("No documents found. Aborting.", err=True)
        sys.exit(1)

    total_words = sum(len(d["text"].split()) for d in docs)
    click.echo(f"  {total_words:,} words total.")

    click.echo("Running pipeline...")
    fingerprint = run_pipeline(docs, brand_name=brand_name, source=source)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_path = write_voice_fingerprint(fingerprint, out_dir / "voice-fingerprint.json")
    bc_path = write_brand_config(fingerprint, out_dir / "brand-config.json")
    cm_path = write_claude_md(fingerprint, out_dir / "CLAUDE.md")

    t_total = time.time() - t_start

    click.echo("")
    click.echo("Done.")
    click.echo(f"  {fp_path}")
    click.echo(f"  {bc_path}")
    click.echo(f"  {cm_path}")
    click.echo(f"  Runtime: {t_total:.1f}s")
    click.echo("")
    click.echo(f"Primary tone: {fingerprint['tone']['primary_tone']} "
               f"(confidence {fingerprint['tone']['confidence']:.0%})")
    click.echo(f"Sentences analyzed: {fingerprint['cadence']['sentence_count']}")
    click.echo(f"Signature words: {', '.join(w['token'] for w in fingerprint['signature_words'][:8])}")
    click.echo(f"Forbidden words: {', '.join(w['token'] for w in fingerprint['forbidden_words'][:8])}")

    if fingerprint.get("thin_corpus"):
        click.echo("")
        click.secho(
            "⚠  Thin corpus warning: fewer than 3,000 words analyzed. "
            "The cadence, signature, and tone signals are still reliable, but the forbidden-word "
            "list is likely noisy. Re-run on a richer source (e.g. a blog folder or "
            "/blog section) for a stronger fingerprint.",
            fg="yellow"
        )


if __name__ == "__main__":
    main()
