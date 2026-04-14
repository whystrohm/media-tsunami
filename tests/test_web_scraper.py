"""Tests for web_scraper."""
from __future__ import annotations

import httpx
import pytest

from tsunami.inputs.web_scraper import (
    _normalize_url,
    _priority,
    _should_skip,
    extract_content,
    scrape_url,
)


def _wrap(body_html: str, title: str = "Test Page") -> str:
    return f"<!doctype html><html><head><title>{title}</title></head><body>{body_html}</body></html>"


REAL_ARTICLE_PARAGRAPHS = [
    "Brand voice is not a style guide. It is the compression of a thousand small decisions "
    "about how your company talks, shows up, and positions itself in the market. Most teams treat "
    "it as decoration rather than infrastructure, and that is why their content drifts.",
    "When you hand an AI model a generic prompt, it defaults to the median of the training data: "
    "safe, bland, indistinguishable. The only way to get output that sounds like you is to give the "
    "model a compressed representation of how you actually think. We call this a fingerprint.",
    "A proper fingerprint captures three layers: the words you use (and refuse to use), the rhythm "
    "of your sentences, and the stance you take toward the reader. Everything else is commentary. "
    "Once these three layers are encoded, the model stops guessing and starts sounding like you.",
    "This is the difference between tools that produce content and systems that produce coherence. "
    "Anyone can ship a tool. Very few teams ship a system that survives contact with the team.",
]


def test_extract_content_rejects_link_soup():
    body = """
    <main>
      <div class="content">
        <a href="/a">Alpha</a> <a href="/b">Beta</a> <a href="/c">Gamma</a>
        <a href="/d">Delta</a> <a href="/e">Epsilon</a> <a href="/f">Zeta</a>
        <a href="/g">Eta</a> <a href="/h">Theta</a>
      </div>
    </main>
    """
    doc = extract_content(_wrap(body), "https://example.com/")
    # Either None (no content above threshold) or doesn't include the link-soup texts
    if doc is not None:
        for token in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"):
            assert token not in doc["text"], f"link soup leaked: {token}"


def test_extract_content_rejects_breadcrumb():
    paragraphs_html = "".join(f"<p>{p}</p>" for p in REAL_ARTICLE_PARAGRAPHS)
    body = f"""
    <main>
      <div>Home \u2192 Blog \u2192 Post \u2192 This One</div>
      {paragraphs_html}
    </main>
    """
    doc = extract_content(_wrap(body), "https://example.com/post")
    assert doc is not None
    assert "Home \u2192 Blog \u2192 Post" not in doc["text"]


def test_extract_content_keeps_real_article():
    paragraphs_html = "".join(f"<p>{p}</p>" for p in REAL_ARTICLE_PARAGRAPHS)
    body = f"""
    <nav class="site-nav"><a href="/">Home</a><a href="/blog">Blog</a></nav>
    <main>
      <h1>The Real Cost of Brand Drift</h1>
      {paragraphs_html}
    </main>
    <footer>Copyright something</footer>
    """
    doc = extract_content(_wrap(body, title="Brand Drift | Example"), "https://example.com/post")
    assert doc is not None
    assert doc["path"] == "/post"
    assert "Brand Drift" in doc["title"]
    # Real article body is present
    assert "fingerprint" in doc["text"].lower()
    assert len(doc["text"].split()) > 100
    # Nav/footer stripped
    assert "Copyright something" not in doc["text"]


def test_extract_content_empty_returns_none():
    body = """
    <nav><a href="/">Home</a></nav>
    <main></main>
    <footer>(c) 2026</footer>
    """
    doc = extract_content(_wrap(body), "https://example.com/")
    assert doc is None


def test_extract_content_on_saved_whystrohm_html():
    try:
        with httpx.Client(follow_redirects=True, timeout=10.0) as client:
            r = client.get("https://whystrohm.com/")
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        pytest.skip("network unavailable")
        return
    if r.status_code >= 400:
        pytest.skip(f"whystrohm returned HTTP {r.status_code}")
        return
    doc = extract_content(r.text, "https://whystrohm.com/")
    assert doc is not None, "homepage should produce a document"
    # The breadcrumb soup that polluted the previous snapshot must not appear.
    bad = "Score your content infrastructure in 10 seconds \u2192 See the builds \u2192 Share"
    assert bad not in doc["text"]
    assert len(doc["text"]) > 500
    assert "content" in doc["text"].lower()


def test_crawl_priority_ordering():
    scored = sorted(
        [
            ("/", _priority("/")),
            ("/about", _priority("/about")),
            ("/contact", _priority("/contact")),
            ("/blog/post-slug", _priority("/blog/post-slug")),
            ("/pricing", _priority("/pricing")),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    top_two = [p for p, _ in scored[:2]]
    assert "/" in top_two
    assert "/about" in top_two
    # /contact gets base priority (2), should be last.
    assert scored[-1][0] == "/contact"


def test_path_normalization():
    assert _normalize_url(
        "https://example.com/foo/?q=1#x", "https://example.com"
    ) == "https://example.com/foo/"
    # Relative resolution
    assert _normalize_url("/bar", "https://example.com/foo/") == "https://example.com/bar"
    # Netloc lowercase
    assert _normalize_url("https://EXAMPLE.com/x", "https://example.com") == "https://example.com/x"
    # Root stays "/"
    assert _normalize_url("https://example.com", "https://example.com") == "https://example.com/"


def test_skip_patterns():
    assert _should_skip("/api/v1/thing")
    assert _should_skip("/assets/img.png")
    assert _should_skip("/wp-login.php")
    assert _should_skip("/static/main.js")
    assert not _should_skip("/blog/post-slug")
    assert not _should_skip("/about")
    assert not _should_skip("/")
