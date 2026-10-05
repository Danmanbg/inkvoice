"""Book -> Script: segmentation plus a director that assigns speakers and emotions."""
from __future__ import annotations

from ..model import NARRATOR, Book, Chapter, Script, Unit
from .heuristic import HeuristicDirector
from .segment import detect_quote_style, segment_paragraphs


def segment_book(book: Book) -> list[Chapter]:
    style = detect_quote_style([p.text for ch in book.chapters for p in ch.paragraphs if p.kind == "text"])
    chapters = [Chapter(title=ch.title, units=segment_paragraphs(ch.paragraphs, style)) for ch in book.chapters]
    if chapters and book.title:
        intro = book.title + (f"\nWritten by {book.author}" if book.author else "")
        chapters[0].units.insert(0, Unit(kind="heading", text=intro, para=-1))
    return chapters


def build_script(book: Book, director: str = "heuristic", log=print, **director_opts) -> Script:
    chapters = segment_book(book)
    characters = HeuristicDirector(chapters).run()
    script = Script(title=book.title, author=book.author, source=book.source, chapters=chapters,
                    characters=characters, director="heuristic", language=book.language)
    if director == "claude":
        from .claude_director import ClaudeDirector
        ClaudeDirector(log=log, **director_opts).direct(script)
        script.director = "claude"
    elif director != "heuristic":
        raise ValueError(f"Unknown director '{director}' (use 'heuristic' or 'claude')")
    _drop_unused_characters(script)
    script.count_lines()
    return script


def _drop_unused_characters(script: Script) -> None:
    used = {u.speaker for ch in script.chapters for u in ch.units if u.kind == "dialogue"}
    for name in list(script.characters):
        if name not in used and name != NARRATOR:
            del script.characters[name]
