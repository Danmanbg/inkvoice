"""PDF -> Book.

Books in PDF form are a pile of positioned glyphs, so most of the work here is
undoing the typesetting: rebuild lines from characters, drop running headers,
page numbers and footnotes, glue hyphenated words back together, rebuild
paragraphs from indentation/spacing and find where chapters start.
"""
from __future__ import annotations

import io
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..model import Book, Paragraph
from ..textutil import clean_unicode, roman_to_int
from .structure import build_chapters

CHAPTER_RX = re.compile(
    r"^(chapter|chap\.|part|book|volume|act|prologue|epilogue|interlude|introduction|"
    r"foreword|preface|afterword|postscript|acknowledg(e)?ments?|appendix|"
    r"author'?s note|a note|about the author)\b", re.I)
NUMBER_WORDS = set("one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                   "fifteen sixteen seventeen eighteen nineteen twenty".split())
BREAK_RX = re.compile(r"^([*•·◆◇◊~#⁂✻✱✦❖§=—–-]\s*){1,7}$")
PAGE_NUM_RX = re.compile(r"^\W*(page\s*)?\d{1,4}\W*$|^\W*\d{1,4}\s*(of|/)\s*\d{1,4}\W*$", re.I)
BOLD_RX = re.compile(r"bold|black|heavy|semibold|demi", re.I)


@dataclass
class Line:
    page: int
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool = False
    tags: set = field(default_factory=set)   # heading / break / drop


# ------------------------------------------------------------------ lines --
def _chars_to_lines(chars: list[dict], pno: int, drop_superscripts: bool = True) -> list[Line]:
    chars = [c for c in chars if c.get("upright", True) and c["text"] not in ("\n", "\r")]
    if not chars:
        return []
    sizes = [round(c["size"], 1) for c in chars if c["text"].strip()]
    page_size = Counter(sizes).most_common(1)[0][0] if sizes else 10.0

    # Drop caps: a giant first letter spanning several lines. Pull them out and
    # glue them to the line they start.
    dropcaps = [c for c in chars if c["size"] > 2.2 * page_size and c["text"].isalpha()]
    if dropcaps and len(dropcaps) <= 2:
        chars = [c for c in chars if c not in dropcaps]
    else:
        dropcaps = []

    chars.sort(key=lambda c: ((c["top"] + c["bottom"]) / 2, c["x0"]))
    clusters: list[list[dict]] = []
    for c in chars:
        mid = (c["top"] + c["bottom"]) / 2
        if clusters:
            cl = clusters[-1]
            ref = cl[0]["_mid"]
            if abs(mid - ref) <= 0.42 * max(c["size"], cl[0]["size"]):
                c["_mid"] = ref
                cl.append(c)
                continue
        c["_mid"] = mid
        clusters.append([c])

    lines = []
    for cl in clusters:
        cl.sort(key=lambda c: c["x0"])
        visible = [c for c in cl if c["text"].strip()]
        if not visible:
            continue
        line_size = Counter(round(c["size"], 1) for c in visible).most_common(1)[0][0]
        normal = [c for c in visible if c["size"] >= 0.85 * line_size]
        baseline = statistics.median(c["bottom"] for c in normal) if normal else visible[0]["bottom"]
        text, prev = [], None
        for c in cl:
            ch = c["text"]
            if (drop_superscripts and c["size"] < 0.8 * line_size and ch.strip()
                    and c["bottom"] < baseline - 0.2 * line_size and re.match(r"[\d*†‡§¹²³⁰-⁹]", ch)):
                continue          # footnote marker
            if ch.isspace():
                if text and text[-1] != " ":
                    text.append(" ")
                prev = c
                continue
            if prev is not None and text and text[-1] != " ":
                gap = c["x0"] - prev["x1"]
                if gap > 0.17 * max(c["size"], 1.0):
                    text.append(" ")
            text.append(ch)
            prev = c
        s = clean_unicode("".join(text))
        if not s:
            continue
        bold_n = sum(1 for c in visible if BOLD_RX.search(c.get("fontname", "")))
        lines.append(Line(page=pno, text=s, x0=min(c["x0"] for c in visible),
                          x1=max(c["x1"] for c in visible), top=min(c["top"] for c in normal or visible),
                          bottom=max(c["bottom"] for c in normal or visible), size=line_size,
                          bold=bold_n > 0.6 * len(visible)))

    for dc in dropcaps:
        target = next((ln for ln in lines if ln.bottom > dc["top"] and ln.x0 >= dc["x0"] - 1), None)
        if target is None:
            continue
        rest = target.text
        joiner = "" if (rest[:1].islower() or (rest[:1].isupper() and dc["text"] not in "AI")) else " "
        target.text = dc["text"] + joiner + rest
        target.x0 = min(target.x0, dc["x0"])
    lines.sort(key=lambda ln: (ln.top, ln.x0))
    return lines


# ------------------------------------------------------------ page roles --
def _looks_like_toc(lines: list[Line]) -> bool:
    if len(lines) < 5:
        return False
    numbered = sum(1 for ln in lines if re.search(r"(\.{3,}|\s)\d{1,4}$|^\d{1,4}\s+\S", ln.text))
    has_title = any(re.match(r"^(table of )?contents$", ln.text.strip(), re.I) for ln in lines[:4])
    return numbered >= 0.5 * len(lines) or (has_title and numbered >= 0.3 * len(lines))


def _looks_like_copyright(lines: list[Line]) -> bool:
    text = " ".join(ln.text for ln in lines).lower()
    hits = sum(k in text for k in ("all rights reserved", "isbn", "library of congress",
                                   "printed in", "first published", "copyright ©", "© "))
    return hits >= 2


# ------------------------------------------------------------- extraction --
def parse_pages(spec: str | None, n: int) -> set[int] | None:
    if not spec:
        return None
    keep: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a.strip() else 1
            b = int(b) if b.strip() else n
            keep.update(range(max(1, a) - 1, min(n, b)))
        else:
            keep.add(int(part) - 1)
    return keep


def extract_pdf(path: str | Path, pages: str | None = None, keep_footnotes: bool = False,
                progress=None) -> Book:
    import pdfplumber

    path = Path(path)
    notes: list[str] = []
    page_lines: list[list[Line]] = []
    page_dims: list[tuple[float, float]] = []
    with pdfplumber.open(str(path)) as pdf:
        meta = pdf.metadata or {}
        n = len(pdf.pages)
        wanted = parse_pages(pages, n)
        for i, page in enumerate(pdf.pages):
            if progress:
                progress(i + 1, n)
            if wanted is not None and i not in wanted:
                page_lines.append([])
                page_dims.append((float(page.width), float(page.height)))
                continue
            try:
                chars = page.dedupe_chars().chars
            except Exception:  # noqa: BLE001 - odd pages shouldn't kill the whole book
                chars = page.chars
            page_lines.append(_chars_to_lines(chars, i))
            page_dims.append((float(page.width), float(page.height)))
            page.flush_cache()

    total_chars = sum(len(ln.text) for pl in page_lines for ln in pl)
    if total_chars < 200 * max(1, sum(1 for pl in page_lines if pl) // 10):
        notes.append("Very little text found - this PDF may be scanned images. "
                     "Run it through OCR first (e.g. ocrmypdf) for good results.")

    paragraphs = lines_to_paragraphs(page_lines, page_dims, keep_footnotes, notes)
    title = clean_unicode(str(meta.get("Title") or "")).strip()
    if not title or title.lower() in ("untitled", "microsoft word") or title.lower().endswith((".doc", ".docx")):
        title = path.stem.replace("_", " ").strip()
    author = clean_unicode(str(meta.get("Author") or "")).strip()

    outline = _read_outline(path)
    chapters = build_chapters(paragraphs, outline, notes)
    return Book(title=title, author=author, source=str(path), chapters=chapters,
                cover=_render_cover(path), notes=notes)


def lines_to_paragraphs(page_lines: list[list[Line]], page_dims, keep_footnotes: bool,
                        notes: list[str]) -> list[Paragraph]:
    all_lines = [ln for pl in page_lines for ln in pl]
    if not all_lines:
        return []

    # ---- document statistics ------------------------------------------
    size_votes = Counter()
    for ln in all_lines:
        size_votes[round(ln.size * 2) / 2] += len(ln.text)
    body = size_votes.most_common(1)[0][0]
    is_body = lambda ln: abs(ln.size - body) <= 0.08 * body  # noqa: E731

    pitches = []
    for pl in page_lines:
        for a, b in zip(pl, pl[1:]):
            if is_body(a) and is_body(b):
                d = b.top - a.top
                if 0.8 * body < d < 2.5 * body:
                    pitches.append(d)
    pitch = statistics.median(pitches) if pitches else body * 1.25

    # ---- running headers / footers / page numbers ------------------------
    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.lower()).strip(" .-–—|")

    top_counts, bottom_counts = Counter(), Counter()
    for pl in page_lines:
        for ln in pl[:2]:
            top_counts[norm(ln.text)] += 1
        for ln in pl[-2:]:
            bottom_counts[norm(ln.text)] += 1
    n_pages = max(1, sum(1 for pl in page_lines if pl))

    def is_pagenum(t: str) -> bool:
        t = t.strip()
        return bool(PAGE_NUM_RX.match(t)) or (len(t) <= 8 and roman_to_int(t.strip(" .-")) is not None
                                              and t.strip(" .-").islower())

    removed_hf = 0
    for pl in page_lines:
        for side in ("top", "bottom"):
            for _ in range(2):
                if not pl:
                    break
                ln = pl[0] if side == "top" else pl[-1]
                nb = (pl[1] if len(pl) > 1 else None) if side == "top" else (pl[-2] if len(pl) > 1 else None)
                gap = ((nb.top - ln.bottom) if side == "top" else (ln.top - nb.bottom)) if nb else 99
                counts = top_counts if side == "top" else bottom_counts
                reps = counts[norm(ln.text)]
                separated = gap > 0.45 * pitch or not is_body(ln)
                if (is_pagenum(ln.text) and (separated or len(pl) > 3)) or (
                        reps >= 3 and len(ln.text) < 90 and (separated or reps >= 0.25 * n_pages)):
                    pl.remove(ln)
                    removed_hf += 1
                else:
                    break
    if removed_hf:
        notes.append(f"Removed {removed_hf} header/footer/page-number lines.")

    # ---- front matter: table of contents and copyright pages -----------
    first_pages = max(3, n_pages // 12)
    for i, pl in enumerate(page_lines):
        if not pl:
            continue
        near_front = i < first_pages or i >= len(page_lines) - max(2, n_pages // 25)
        if (i < first_pages * 2 and _looks_like_toc(pl)) or (near_front and _looks_like_copyright(pl)):
            notes.append(f"Skipped page {i + 1} (looks like {'contents' if _looks_like_toc(pl) else 'copyright'} page).")
            pl.clear()

    # ---- footnotes: small text below the last body line of a page ------
    if not keep_footnotes:
        dropped = 0
        for pl in page_lines:
            last_body = max((k for k, ln in enumerate(pl) if ln.size >= 0.92 * body), default=-1)
            tail = [ln for ln in pl[last_body + 1:] if ln.size < 0.9 * body]
            if tail and last_body >= 0:
                for ln in tail:
                    pl.remove(ln)
                dropped += len(tail)
        if dropped:
            notes.append(f"Removed {dropped} footnote lines (use keep_footnotes to keep them).")

    # ---- margins per page parity --------------------------------------
    margins = {}
    for parity in (0, 1):
        xs0 = [round(ln.x0) for pl in page_lines for ln in pl if ln.page % 2 == parity and is_body(ln)]
        xs1 = sorted(ln.x1 for pl in page_lines for ln in pl if ln.page % 2 == parity and is_body(ln))
        if xs0:
            left = Counter(xs0).most_common(1)[0][0]
            right = xs1[int(len(xs1) * 0.9)] if xs1 else left + 300
            margins[parity] = (left, right)
    default_m = margins.get(0) or margins.get(1) or (72, 540)

    # ---- dehyphenation vocabulary -------------------------------------
    vocab, hyph = Counter(), set()
    for ln in all_lines:
        for w in re.findall(r"[A-Za-z][A-Za-z'’-]*[A-Za-z]", ln.text):
            if "-" in w:
                hyph.add(w.lower())
                for p in w.split("-"):
                    vocab[p.lower()] += 1
            else:
                vocab[w.lower()] += 1

    def join(a: str, b: str) -> str:
        if a.endswith("-") and not a.endswith("--") and b[:1].islower():
            m1 = re.search(r"([A-Za-z]+)-$", a)
            m2 = re.match(r"([a-z]+)", b)
            if m1 and m2:
                left, right = m1.group(1), m2.group(1)
                whole = (left + right).lower()
                if f"{left}-{right}".lower() in hyph and vocab[whole] == 0:
                    return a + b                        # real compound: keep the hyphen
            return a[:-1] + b
        if a.endswith(("—", "–")) or b.startswith(("—",)):
            return a + b
        return a + " " + b

    # ---- classify lines and assemble paragraphs ------------------------
    paragraphs: list[Paragraph] = []
    cur: list[str] = []
    cur_page = 0
    cur_kind = "text"

    def flush():
        nonlocal cur, cur_kind
        if cur:
            text = cur[0]
            for nxt in cur[1:]:
                text = join(text, nxt) if cur_kind == "text" else text + "\n" + nxt
            text = re.sub(r"\s+", " ", text).strip() if cur_kind == "text" else text.strip()
            if text:
                paragraphs.append(Paragraph(text=text, kind=cur_kind, page=cur_page))
        cur, cur_kind = [], "text"

    prev: Line | None = None
    for pl in page_lines:
        for idx, ln in enumerate(pl):
            left, right = margins.get(ln.page % 2, default_m)
            width = max(50.0, right - left)
            center = (left + right) / 2
            t = ln.text.strip()
            ratio = ln.size / body
            centered = (abs((ln.x0 + ln.x1) / 2 - center) < 0.06 * width
                        and ln.x0 - left > 1.5 * body and right - ln.x1 > 1.5 * body)
            letters = [c for c in t if c.isalpha()]
            caps = len(letters) >= 2 and sum(c.isupper() for c in letters) >= 0.85 * len(letters)
            same_page = prev is not None and prev.page == ln.page
            gap = (ln.top - prev.bottom) if same_page else 0.0
            line_gap = max(0.0, pitch - (prev.bottom - prev.top)) if prev else 0.0
            big_gap = same_page and gap > line_gap + 1.6 * pitch
            num_only = t.isdigit() and len(t) <= 3 or (roman_to_int(t) is not None and len(t) <= 7
                                                       and t.isupper()) or t.lower() in NUMBER_WORDS

            kind = "text"
            if BREAK_RX.match(t) and len(t) <= 15:
                kind = "break"
            elif len(t) < 120 and (
                    ratio >= 1.2
                    or (CHAPTER_RX.match(t) and len(t) < 70 and (centered or caps or ratio > 1.05 or ln.bold
                                                                 or idx <= 1 or big_gap))
                    or (num_only and (centered or ratio > 1.05 or idx <= 1))
                    or (centered and caps and len(t) < 60 and (idx <= 2 or big_gap) and not t.endswith((".", ","))
                        and ratio >= 0.98)):
                kind = "heading"

            if kind == "break":
                flush()
                paragraphs.append(Paragraph(text="* * *", kind="break", page=ln.page))
                prev = ln
                continue

            if kind == "heading":
                if cur_kind != "heading" or (prev and not same_page) or big_gap and len(cur) >= 2:
                    flush()
                    cur_kind = "heading"
                    cur_page = ln.page
                cur.append(t)
                prev = ln
                continue

            # body text
            indent = ln.x0 - left > 0.6 * body and not centered
            prev_short = prev is not None and (prev.x1 < (margins.get(prev.page % 2, default_m)[1]) - 2.0 * body)
            prev_end = bool(cur) and re.search(r"[.!?…:\"”’)\]]$", cur[-1]) is not None
            new_para = (
                cur_kind != "text" or not cur
                or (same_page and big_gap and gap > 2.2 * pitch)
                or (same_page and gap > line_gap + 0.55 * pitch)
                or indent
                or (prev_short and prev_end)
                or centered
            )
            if same_page and big_gap and gap > 2.2 * pitch and cur and cur_kind == "text":
                flush()
                paragraphs.append(Paragraph(text="* * *", kind="break", page=ln.page))
            if new_para:
                flush()
                cur_page = ln.page
            cur.append(t)
            prev = ln
        # page boundary: nothing special - continuation is decided by indent/short-line rules
    flush()

    # A "break" right after a heading or at the very start is meaningless.
    cleaned: list[Paragraph] = []
    for p in paragraphs:
        if p.kind == "break" and (not cleaned or cleaned[-1].kind in ("break", "heading")):
            continue
        cleaned.append(p)
    return cleaned


# ---------------------------------------------------------------- outline --
def _read_outline(path: Path) -> list[tuple[int, str, int]]:
    """[(level, title, page_index)] from the PDF bookmarks, if any."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []
    out = []
    try:
        pdf = pdfium.PdfDocument(str(path))
        try:
            for item in pdf.get_toc():
                level = getattr(item, "level", 0)
                if hasattr(item, "get_title"):
                    title = item.get_title()
                    dest = item.get_dest()
                    page = dest.get_index() if dest is not None else None
                else:
                    title, page = item.title, item.page_index
                if title and page is not None:
                    out.append((int(level), clean_unicode(title), int(page)))
        finally:
            pdf.close()
    except Exception:  # noqa: BLE001 - a broken outline just means "no outline"
        return []
    return out


def _render_cover(path: Path) -> bytes | None:
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            page = pdf[0]
            w, h = page.get_size()
            scale = min(3.0, 1400 / max(w, h))
            img = page.render(scale=scale).to_pil().convert("RGB")
        finally:
            pdf.close()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None
