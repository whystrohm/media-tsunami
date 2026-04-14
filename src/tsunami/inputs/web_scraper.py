"""Crawl a website from a root URL and return a corpus (list[Document]).

Produces the same Document shape as folder_reader: {"path", "title", "text"}.
Rejects breadcrumb/tag-soup via a content-density filter.
"""
from __future__ import annotations

import re
import time
import urllib.robotparser
from collections import deque
from typing import TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

MAX_TITLE_LEN = 120
MIN_TEXT_CHARS = 200
MIN_TEXT_WORDS = 40

_SKIP_PREFIXES = (
    "/login", "/signup", "/auth", "/wp-admin", "/wp-login", "/api/",
    "/assets/", "/static/", "/images/", "/cdn-cgi/", "/robots.txt", "/sitemap",
)
_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp4", ".mp3", ".zip", ".css", ".js", ".ico", ".xml",
)
_JUNK_SELECTOR_RE = re.compile(
    r"nav|navigation|menu|footer|header|sidebar|breadcrumb|cookie|banner|popup|modal|subscribe|newsletter",
    re.IGNORECASE,
)
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "ul", "ol", "blockquote",
               "h1", "h2", "h3", "h4", "h5", "h6", "pre", "figure", "table"}
_BREADCRUMB_SEPS = (" \u2192 ", " / ", " | ", " \u2022 ", " > ", " \u203a ")


class Document(TypedDict):
    path: str
    title: str
    text: str


def _normalize_url(url: str, base: str) -> str:
    full = urljoin(base, url)
    p = urlparse(full)
    if not p.scheme or not p.netloc:
        return ""
    path = p.path or "/"
    return f"{p.scheme}://{p.netloc.lower()}{path}"


def _should_skip(path: str) -> bool:
    low = path.lower()
    if any(low.startswith(pfx) for pfx in _SKIP_PREFIXES):
        return True
    if any(low.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return True
    return False


def _priority(path: str) -> int:
    p = path.lower().rstrip("/")
    if p == "" or p == "/":
        return 10
    if p == "/about" or p.startswith("/about/"):
        return 8
    if p == "/blog" or p == "/posts":
        return 6
    if p.startswith("/blog/") or p.startswith("/posts/"):
        return 5
    for pre in ("/services", "/pricing", "/work", "/portfolio", "/case-studies"):
        if p == pre or p.startswith(pre + "/"):
            return 4
    return 2


def _normalize_whitespace(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in cleaned if p)


def _looks_like_breadcrumb(text: str) -> bool:
    """Detect breadcrumb/tag-soup patterns: short phrases joined by separators."""
    for sep in _BREADCRUMB_SEPS:
        if text.count(sep) >= 2:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if parts and all(len(p) < 60 for p in parts):
                return True
    return False


def _block_is_junk(block) -> bool:
    """Return True if a block element is link-soup / breadcrumb / nav inline."""
    block_text = block.get_text(" ", strip=True)
    if not block_text:
        return True
    total = len(block_text)
    link_chars = sum(len(a.get_text(" ", strip=True)) for a in block.find_all("a"))
    link_count = len(block.find_all("a"))

    if total > 0 and (link_chars / total) > 0.5 and total < 300:
        return True
    if link_count > 5 and total < 150:
        return True
    if _looks_like_breadcrumb(block_text):
        return True
    return False


def _extract_title(soup: BeautifulSoup, content_root) -> str:
    if soup.title and soup.title.string:
        t = soup.title.string.strip()
        if t:
            return t[:MAX_TITLE_LEN]
    if content_root:
        h1 = content_root.find("h1")
        if h1:
            t = h1.get_text(" ", strip=True)
            if t:
                return t[:MAX_TITLE_LEN]
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        if t:
            return t[:MAX_TITLE_LEN]
    return "Untitled"


def extract_content(html: str, url: str) -> dict | None:
    """Parse a single HTML page into a Document. Returns None if content density too low."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Remove junk by class/id regex.
    for tag in list(soup.find_all(True)):
        if getattr(tag, "attrs", None) is None:
            continue
        attrs = []
        cls = tag.attrs.get("class")
        if cls:
            attrs.append(" ".join(cls) if isinstance(cls, list) else str(cls))
        tid = tag.attrs.get("id")
        if tid:
            attrs.append(str(tid))
        if attrs and _JUNK_SELECTOR_RE.search(" ".join(attrs)):
            tag.decompose()

    # Find candidate content region.
    content_root = soup.find("main") or soup.find("article") or soup.body or soup

    title = _extract_title(soup, content_root)

    # Walk block-level children and filter.
    kept_parts: list[str] = []
    # Use descendants-level iteration at the block tag layer so we catch nested main > div > p.
    # Simpler: recursively descend; when we hit a block, decide to drop or keep its text.
    # To avoid double-counting, we do a shallow pass over direct block descendants by
    # recursing ONLY into containers (div/section/article) that are NOT themselves junk.
    def walk(node):
        for child in node.find_all(True, recursive=False):
            name = child.name
            if name not in _BLOCK_TAGS:
                continue
            if _block_is_junk(child):
                continue
            if name in {"div", "section", "article"}:
                # Recurse into containers to pick out their paragraph-level blocks.
                # But if the container has substantive direct text of its own with few links,
                # keep it whole (helps with pages where p tags are missing).
                direct_text = "".join(
                    t for t in child.find_all(string=True, recursive=False) if t.strip()
                ).strip()
                if direct_text and len(direct_text) > 80:
                    txt = child.get_text(" ", strip=True)
                    if txt:
                        kept_parts.append(txt)
                else:
                    walk(child)
            else:
                txt = child.get_text(" ", strip=True)
                if txt:
                    kept_parts.append(txt)

    walk(content_root)

    # Fallback: if walker produced nothing but content_root has real text, salvage.
    if not kept_parts:
        fallback = content_root.get_text("\n", strip=True)
        kept_parts = [line for line in fallback.split("\n") if line.strip()]

    text = _normalize_whitespace("\n\n".join(kept_parts))

    if len(text) < MIN_TEXT_CHARS or len(text.split()) < MIN_TEXT_WORDS:
        return None

    path = urlparse(url).path or "/"
    return {"path": path, "title": title, "text": text}


def _load_robots(client: httpx.Client, origin: str, timeout: float) -> urllib.robotparser.RobotFileParser | None:
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = client.get(f"{origin}/robots.txt", timeout=timeout, follow_redirects=True)
        if r.status_code >= 400:
            return None
        rp.parse(r.text.splitlines())
        return rp
    except Exception:
        return None


def _extract_links(html: str, base_url: str, origin_netloc: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        norm = _normalize_url(a["href"], base_url)
        if not norm:
            continue
        p = urlparse(norm)
        if p.netloc != origin_netloc:
            continue
        if _should_skip(p.path):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def scrape_url(
    url: str,
    max_pages: int = 20,
    timeout: float = 10.0,
    delay_seconds: float = 1.0,
    user_agent: str = "media-tsunami/0.1 (+https://github.com/whystrohm/media-tsunami)",
) -> list[dict]:
    """Crawl a website from a root URL and return a corpus as list[Document]."""
    root = _normalize_url(url, url)
    if not root:
        return []
    origin_parsed = urlparse(root)
    origin = f"{origin_parsed.scheme}://{origin_parsed.netloc}"
    origin_netloc = origin_parsed.netloc

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    docs: list[dict] = []
    visited: set[str] = set()
    # Frontier: list of (score, url). We pop the highest-scored each iteration.
    frontier: list[tuple[int, str]] = [(_priority(origin_parsed.path or "/"), root)]

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        robots = _load_robots(client, origin, timeout)

        while frontier and len(visited) < max_pages:
            # Pop highest-priority URL.
            frontier.sort(key=lambda x: x[0], reverse=True)
            _, next_url = frontier.pop(0)
            if next_url in visited:
                continue
            visited.add(next_url)

            path = urlparse(next_url).path or "/"
            if _should_skip(path):
                continue
            if robots is not None and not robots.can_fetch(user_agent, next_url):
                continue

            try:
                r = client.get(next_url, timeout=timeout)
            except Exception as e:
                print(f"[web_scraper] fetch error {next_url}: {type(e).__name__}: {e}")
                time.sleep(delay_seconds)
                continue

            if r.status_code >= 400:
                print(f"[web_scraper] HTTP {r.status_code} {next_url}")
                time.sleep(delay_seconds)
                continue

            ctype = r.headers.get("content-type", "")
            if "html" not in ctype.lower():
                time.sleep(delay_seconds)
                continue

            html = r.text
            doc = extract_content(html, next_url)
            if doc is not None:
                docs.append(doc)

            # Discover more links.
            for link in _extract_links(html, next_url, origin_netloc):
                if link in visited:
                    continue
                # Avoid duplicates already queued.
                if any(u == link for _, u in frontier):
                    continue
                frontier.append((_priority(urlparse(link).path or "/"), link))

            time.sleep(delay_seconds)

    return docs
