"""Read a folder of writing (md/html/txt) into a list of Document dicts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup

EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".txt"}
HTML_EXTENSIONS = {".html", ".htm"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
MIN_TEXT_LEN = 20
MAX_TITLE_LEN = 120


class Document(TypedDict):
    path: str
    title: str
    text: str


def _normalize_whitespace(text: str) -> str:
    # Preserve paragraph breaks (\n\n), collapse other whitespace runs to single space.
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in cleaned if p)


def _strip_markdown(raw: str) -> tuple[str, str | None]:
    """Return (cleaned_text, h1_title_or_None)."""
    h1_match = re.search(r"^\s*#\s+(.+?)\s*$", raw, flags=re.MULTILINE)
    h1 = h1_match.group(1).strip() if h1_match else None

    text = raw
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)  # header hashes
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
    text = text.replace("`", "")
    return text, h1


def _strip_html(raw: str) -> tuple[str, str | None]:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(separator=" ", strip=True) if h1_tag else None
    text = soup.get_text(separator=" ")
    return text, h1


def _pick_title(h1: str | None, text: str, fallback_stem: str) -> str:
    if h1:
        return h1[:MAX_TITLE_LEN]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if len(stripped) <= MAX_TITLE_LEN:
                return stripped
            break
    return fallback_stem


def _read_file(file_path: Path) -> Document | None:
    try:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    ext = file_path.suffix.lower()
    if ext in HTML_EXTENSIONS:
        text, h1 = _strip_html(raw)
    elif ext in MARKDOWN_EXTENSIONS:
        text, h1 = _strip_markdown(raw)
    else:  # .txt
        text, h1 = raw, None

    text = _normalize_whitespace(text)
    if len(text) < MIN_TEXT_LEN:
        return None

    title = _pick_title(h1, text, file_path.stem)
    return Document(path="", title=title, text=text)


def read_folder(path: str | Path) -> list[Document]:
    root = Path(path).resolve()
    docs: list[Document] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in EXTENSIONS:
            continue
        doc = _read_file(file_path)
        if doc is None:
            continue
        doc["path"] = str(file_path.relative_to(root))
        docs.append(doc)
    return docs
