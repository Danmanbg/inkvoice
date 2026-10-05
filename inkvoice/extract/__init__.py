"""Readers for the supported book formats."""
from __future__ import annotations

import re
from pathlib import Path

from ..model import Book, Paragraph
from ..textutil import clean_unicode
from .structure import build_chapters

SUPPORTED = (".pdf", ".epub", ".txt", ".md")


def load_book(path: str | Path, pages: str | None = None, keep_footnotes: bool = False,
              progress=None) -> Book:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        from .pdf import extract_pdf
        return extract_pdf(path, pages=pages, keep_footnotes=keep_footnotes, progress=progress)
    if ext == ".epub":
        from .epub import extract_epub
        return extract_epub(path)
    if ext in (".txt", ".md"):
        return extract_text(path)
    raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED)}")


HEADING_RX = re.compile(
    r"^(#+\s+.+|(chapter|part|book|prologue|epilogue|interlude)\b.{0,60}|[IVXLC]{1,7}\.?|\d{1,3}\.?)$", re.I)


def extract_text(path: Path) -> Book:
    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    paragraphs = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        text = clean_unicode(" ".join(lines))
        if len(lines) <= 2 and len(text) < 80 and HEADING_RX.match(lines[0]):
            paragraphs.append(Paragraph(text="\n".join(ln.lstrip("# ") for ln in lines), kind="heading"))
        elif re.fullmatch(r"([*•·~#—-]\s*){1,7}", text):
            paragraphs.append(Paragraph(text="* * *", kind="break"))
        else:
            paragraphs.append(Paragraph(text=text))
    notes: list[str] = []
    return Book(title=path.stem.replace("_", " "), author="", source=str(path),
                chapters=build_chapters(paragraphs, None, notes), notes=notes)
