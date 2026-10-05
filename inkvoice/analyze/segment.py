"""Split paragraphs into narration and dialogue units."""
from __future__ import annotations

import re
from collections import Counter

from ..model import Unit

PUNCT_END = ",.!?…—–-:;"


def detect_quote_style(texts: list[str]) -> str:
    """'double' (“ ”), 'single' (‘ ’), 'straight' (" "), or 'guillemet' (« »)."""
    joined = "\n".join(texts[:4000])
    c = Counter({
        "double": joined.count("“"),
        "single": len(re.findall(r"(?:^|[\s(\[—–-])‘", joined)),
        "straight": joined.count('"') // 2,
        "guillemet": joined.count("«"),
    })
    style, n = c.most_common(1)[0]
    return style if n else "double"


def _is_letter(ch: str) -> bool:
    return ch.isalpha()


def _spans_double(text: str, open_ch: str, close_ch: str, straight: bool) -> tuple[list[tuple[str, str]], bool]:
    spans, buf, in_q = [], [], False

    def flush(kind):
        s = "".join(buf)
        buf.clear()
        if s.strip():
            spans.append((kind, s))

    n = len(text)
    for i, ch in enumerate(text):
        prev = text[i - 1] if i else " "
        nxt = text[i + 1] if i + 1 < n else " "
        if straight and ch == '"':
            opening = not in_q
        elif ch == open_ch:
            opening = True
            if in_q and not prev.isspace():          # “ used as a closer by bad typesetting
                opening = False
        elif ch == close_ch:
            opening = False
            if not in_q and _is_letter(nxt) and (prev.isspace() or i == 0):   # ” used as an opener
                opening = True
        else:
            buf.append(ch)
            continue
        if opening and not in_q:
            flush("narration")
            in_q = True
        elif not opening and in_q:
            flush("dialogue")
            in_q = False
        else:
            buf.append(ch)
    unclosed = in_q
    flush("dialogue" if in_q else "narration")
    return spans, unclosed


def _spans_single(text: str) -> tuple[list[tuple[str, str]], bool]:
    """British style ‘…’ where ’ is also the apostrophe."""
    spans: list[tuple[str, str]] = []
    i, n, last = 0, len(text), 0
    unclosed = False
    while i < n:
        ch = text[i]
        prev = text[i - 1] if i else " "
        if ch == "‘" and (i == 0 or prev.isspace() or prev in "([—–-“"):
            # find the matching closer
            close = None
            scare = None
            for j in range(i + 1, n):
                if text[j] != "’":
                    continue
                nxt = text[j + 1] if j + 1 < n else " "
                if _is_letter(nxt):
                    continue                         # don't, it's, o’clock
                if text[j - 1] in PUNCT_END:
                    close = j
                    break
                if scare is None and j - i <= 40:
                    scare = j
            if close is None and scare is not None:
                i = scare + 1                        # ‘scare quotes’ stay in the narration
                continue
            if text[last:i].strip():
                spans.append(("narration", text[last:i]))
            if close is None:
                spans.append(("dialogue", text[i + 1:]))
                unclosed = True
                last = n
                break
            spans.append(("dialogue", text[i + 1:close]))
            last = i = close + 1
            continue
        i += 1
    if last < n and text[last:].strip():
        spans.append(("narration", text[last:]))
    return spans, unclosed


def split_paragraph(text: str, style: str) -> tuple[list[tuple[str, str]], bool]:
    if style == "single":
        spans, unclosed = _spans_single(text)
    elif style == "guillemet":
        spans, unclosed = _spans_double(text, "«", "»", straight=False)
    else:
        spans, unclosed = _spans_double(text, "“", "”", straight=(style == "straight"))
        if style == "double" and '"' in text and "“" not in text:
            spans, unclosed = _spans_double(text, "“", "”", straight=True)

    # Very short quoted bits that don't end like speech are titles/scare quotes.
    out: list[tuple[str, str]] = []
    q_open, q_close = {"single": ("‘", "’"), "guillemet": ("«", "»")}.get(style, ("“", "”"))
    for k, (kind, s) in enumerate(spans):
        stripped = s.strip()
        unclosed_tail = unclosed and k == len(spans) - 1
        if (kind == "dialogue" and not unclosed_tail and len(stripped.split()) <= 3
                and stripped[-1:] not in "!?.,…—–-" and k > 0 and spans[k - 1][0] == "narration"):
            kind, s = "narration", f"{q_open}{s}{q_close}"
        if out and out[-1][0] == kind == "narration":
            out[-1] = ("narration", out[-1][1] + s)
        else:
            out.append((kind, s))
    cleaned = []
    for kind, s in out:
        s = re.sub(r"\s+", " ", s).strip()
        if kind == "narration":
            s = s.strip()
        if re.search(r"\w", s):
            cleaned.append((kind, s))
    return cleaned, unclosed


def segment_paragraphs(paragraphs, style: str) -> list[Unit]:
    """Paragraph list -> Units (para index = position in the chapter)."""
    units: list[Unit] = []
    open_quote = False
    for pi, p in enumerate(paragraphs):
        if p.kind == "heading":
            units.append(Unit(kind="heading", text=p.text, para=pi))
            open_quote = False
            continue
        if p.kind == "break":
            units.append(Unit(kind="break", text="", para=pi))
            open_quote = False
            continue
        spans, unclosed = split_paragraph(p.text, style)
        for k, (kind, s) in enumerate(spans):
            u = Unit(kind=kind, text=s, para=pi)
            if k == 0 and kind == "dialogue" and open_quote:
                u.cont = True
            units.append(u)
        open_quote = unclosed
    return units
