"""Tests for folder_reader."""
from pathlib import Path

from tsunami.inputs.folder_reader import read_folder

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_corpus"


def _by_suffix(docs, suffix):
    matches = [d for d in docs if d["path"].endswith(suffix)]
    assert len(matches) == 1, f"expected exactly one {suffix} doc, got {len(matches)}"
    return matches[0]


def test_returns_three_documents():
    docs = read_folder(FIXTURE_DIR)
    # post.md + page.html + note.txt — pdf ignored, empty.md filtered
    assert len(docs) == 3
    paths = {d["path"] for d in docs}
    assert "post.md" in paths
    assert "page.html" in paths
    assert "note.txt" in paths


def test_every_doc_has_nonempty_fields():
    docs = read_folder(FIXTURE_DIR)
    for doc in docs:
        assert doc["path"], f"empty path on {doc}"
        assert doc["title"], f"empty title on {doc}"
        assert doc["text"], f"empty text on {doc}"
        assert len(doc["text"]) >= 20


def test_html_script_contents_are_stripped():
    doc = _by_suffix(read_folder(FIXTURE_DIR), "page.html")
    assert "SECRET_TRACKING_PAYLOAD" not in doc["text"]
    assert "do_not_leak_into_text" not in doc["text"]
    assert "color: red" not in doc["text"]  # <style> also stripped
    assert "Voice is the part of your brand" in doc["text"]


def test_markdown_title_is_from_h1():
    doc = _by_suffix(read_folder(FIXTURE_DIR), "post.md")
    assert doc["title"] == "The Real Cost of Brand Drift"


def test_html_title_is_from_h1():
    doc = _by_suffix(read_folder(FIXTURE_DIR), "page.html")
    assert doc["title"] == "How We Think About Voice"


def test_plain_text_title_is_first_line():
    doc = _by_suffix(read_folder(FIXTURE_DIR), "note.txt")
    assert doc["title"] == "A short field note from the road."


def test_plain_text_title_falls_back_to_stem_when_first_line_too_long(tmp_path):
    long_line = "x" * 200 + " more content follows here to clear the min length floor"
    (tmp_path / "longline.txt").write_text(long_line, encoding="utf-8")
    docs = read_folder(tmp_path)
    assert len(docs) == 1
    assert docs[0]["title"] == "longline"


def test_pdf_and_empty_are_filtered(tmp_path):
    (tmp_path / "skip.pdf").write_bytes(b"")
    (tmp_path / "empty.md").write_text("hi\n", encoding="utf-8")
    (tmp_path / "good.txt").write_text(
        "This is real content that clears the twenty character floor.", encoding="utf-8"
    )
    docs = read_folder(tmp_path)
    assert len(docs) == 1
    assert docs[0]["path"] == "good.txt"
