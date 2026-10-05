"""Extraction test on a small original story printed to PDF (tests/data/lantern_keeper.pdf)."""
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

from inkvoice.extract import load_book  # noqa: E402

PDF = Path(__file__).parent / "data" / "lantern_keeper.pdf"


@pytest.fixture(scope="module")
def book():
    return load_book(PDF)


def test_chapters(book):
    assert [c.title for c in book.chapters] == [
        "Chapter One: The Storm", "Chapter Two: The Boy on the Rocks", "Chapter Three: Morning"]


def test_headers_page_numbers_and_footnotes_removed(book):
    text = "\n".join(p.text for c in book.chapters for p in c.paragraphs)
    assert "THE LANTERN KEEPER" not in text            # running header
    assert "rotation failure" not in text               # footnote
    for c in book.chapters:
        for p in c.paragraphs:
            assert not p.text.strip().isdigit()         # page numbers


def test_paragraphs_and_breaks(book):
    ch2 = book.chapters[1].paragraphs
    assert any(p.kind == "break" for p in ch2)
    first = book.chapters[0].paragraphs[1].text
    assert first.startswith("The wind had been rising since noon")
    assert first.endswith("getting into your head.")    # joined across lines, one paragraph
    assert "  " not in first


def test_no_broken_hyphenation(book):
    words = " ".join(p.text for c in book.chapters for p in c.paragraphs).split()
    # every hyphenated token must be a real compound, not a word split at a line end
    odd = [w for w in words if "-" in w.strip("—–-") and w.lower().strip(".,;:!?”“") not in
           ("cross-legged",)]
    assert odd == []


def test_cover(book):
    assert book.cover and book.cover[:2] == b"\xff\xd8"   # JPEG
