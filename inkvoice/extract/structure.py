"""Turn a flat list of paragraphs into chapters (shared by the PDF/TXT readers)."""
from __future__ import annotations

import re

from ..model import Paragraph, RawChapter

FRONT_MATTER_TITLES = re.compile(
    r"^(cover|title( page)?|half[- ]title|copyright|contents|table of contents|dedication|"
    r"also by|praise for|epigraph|front ?matter|imprint)\b", re.I)
CHAPTERISH = re.compile(r"^(chapter|part|book|prologue|epilogue|interlude|\d{1,3}\b|[IVXLC]{1,7}\b)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _title_of(p: Paragraph) -> str:
    lines = [x.strip().rstrip(".") for x in p.text.split("\n") if x.strip()]
    if len(lines) == 2 and len(lines[0]) <= 20:      # "Chapter One" + "The Storm"
        return f"{lines[0]}: {lines[1]}"
    return ". ".join(lines)


def build_chapters(paragraphs: list[Paragraph], outline: list[tuple[int, str, int]] | None,
                   notes: list[str], min_words: int = 8) -> list[RawChapter]:
    if not paragraphs:
        return []
    starts = _starts_from_outline(paragraphs, outline or [], notes)
    heading_starts = [i for i, p in enumerate(paragraphs) if p.kind == "heading"]
    chapterish = [i for i in heading_starts if CHAPTERISH.match(_title_of(paragraphs[i]))]
    if starts is not None and len(chapterish) > 1.5 * len(starts) and len(chapterish) >= 3:
        notes.append("PDF bookmarks look coarser than the chapter headings; using the headings.")
        starts = None
    if starts is None:
        starts = [(i, _title_of(paragraphs[i])) for i in heading_starts]

    chapters: list[RawChapter] = []
    bounds = starts + [(len(paragraphs), "")]
    if not starts or bounds[0][0] > 0:
        first = bounds[0][0] if starts else len(paragraphs)
        chapters.append(RawChapter(title="Opening", paragraphs=paragraphs[:first]))
    for (a, title), (b, _) in zip(bounds, bounds[1:]):
        if b > a:
            chapters.append(RawChapter(title=title or "Untitled", paragraphs=paragraphs[a:b]))

    # Text before the first heading that is this short is a title page, not a prologue.
    if len(chapters) > 1 and chapters[0].title == "Opening" and chapters[0].word_count() < 150:
        notes.append(f"Dropped {chapters[0].word_count()} words of front matter before the first chapter.")
        chapters.pop(0)

    # Merge consecutive headings / tiny chapters ("Part One" page) into the next chapter.
    merged: list[RawChapter] = []
    carry: RawChapter | None = None
    for ch in chapters:
        if carry is not None:
            ch = RawChapter(title=f"{carry.title}. {ch.title}" if carry.title and carry.title != "Opening"
                            else ch.title, paragraphs=carry.paragraphs + ch.paragraphs)
            carry = None
        if ch.word_count() < min_words and ch is not chapters[-1]:
            carry = ch
            continue
        merged.append(ch)
    if carry is not None:
        merged.append(carry)

    # Make sure every chapter opens with a heading the narrator can announce.
    for ch in merged:
        if not any(p.kind == "heading" for p in ch.paragraphs[:2]) and ch.title not in ("Opening", "Untitled"):
            ch.paragraphs.insert(0, Paragraph(text=ch.title, kind="heading",
                                              page=ch.paragraphs[0].page if ch.paragraphs else 0))
    return merged


def _starts_from_outline(paragraphs: list[Paragraph], outline: list[tuple[int, str, int]],
                         notes: list[str]) -> list[tuple[int, str]] | None:
    entries = [(lvl, t, pg) for lvl, t, pg in outline if t and not FRONT_MATTER_TITLES.match(t.strip())]
    if len(entries) < 3:
        return None
    chosen = None
    for level in sorted({e[0] for e in entries}):
        sel = [e for e in entries if e[0] <= level]
        if len(sel) >= 3:
            chosen = sel
            break
    if not chosen:
        return None
    chosen.sort(key=lambda e: e[2])

    starts: list[tuple[int, str]] = []
    pi = 0
    for _, title, page in chosen:
        # first paragraph on/after the bookmark page
        while pi < len(paragraphs) and paragraphs[pi].page < page:
            pi += 1
        if pi >= len(paragraphs):
            break
        # prefer a heading on that page that matches the bookmark title
        best = pi
        nt = _norm(title)
        j = pi
        while j < len(paragraphs) and paragraphs[j].page == paragraphs[pi].page and j < pi + 12:
            if paragraphs[j].kind == "heading" and nt and (_norm(paragraphs[j].text) in nt or nt in _norm(paragraphs[j].text)):
                best = j
                break
            j += 1
        if starts and best <= starts[-1][0]:
            continue
        starts.append((best, title))
        pi = best + 1
    if len(starts) < 3:
        return None
    notes.append(f"Chapters taken from the PDF bookmarks ({len(starts)} entries).")
    return starts
