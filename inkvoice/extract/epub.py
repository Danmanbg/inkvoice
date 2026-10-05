"""EPUB -> Book, using only the standard library."""
from __future__ import annotations

import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from ..model import Book, Paragraph
from ..textutil import clean_unicode
from .structure import build_chapters

BLOCK = {"p", "div", "li", "blockquote", "section", "article", "td", "dd", "dt", "figcaption", "pre"}
HEADINGS = {"h1", "h2", "h3", "h4"}
SKIP = {"script", "style", "head", "title", "nav", "aside", "svg", "math", "rt", "rp"}


class _Collector(HTMLParser):
    def __init__(self, page: int):
        super().__init__(convert_charrefs=True)
        self.page = page
        self.out: list[Paragraph] = []
        self.buf: list[str] = []
        self.kind = "text"
        self.skip = 0
        self.sup = 0

    def _flush(self):
        text = clean_unicode("".join(self.buf))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            if self.kind == "text" and re.fullmatch(r"([*•·◆◇~#⁂✻✱✦❖§=—–-]\s*){1,7}", text):
                self.out.append(Paragraph("* * *", "break", self.page))
            else:
                self.out.append(Paragraph(text, self.kind, self.page))
        self.buf, self.kind = [], "text"

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        etype = (a.get("epub:type") or "") + " " + (a.get("role") or "")
        if tag in SKIP or "footnote" in etype or "endnote" in etype or "rearnote" in etype:
            self.skip += 1
            return
        if tag == "sup" or "noteref" in etype:
            self.sup += 1
            return
        if tag in HEADINGS:
            self._flush()
            self.kind = "heading"
        elif tag in BLOCK:
            self._flush()
        elif tag == "br":
            self.buf.append(" ")
        elif tag == "hr":
            self._flush()
            self.out.append(Paragraph("* * *", "break", self.page))

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "sup" or tag == "a" and self.sup:
            self.sup = max(0, self.sup - 1)
            return
        if tag in HEADINGS or tag in BLOCK:
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        if self.sup and re.fullmatch(r"\s*[\d*†‡§]+\s*", data):
            return
        self.buf.append(data)

    def close(self):
        super().close()
        self._flush()


def extract_epub(path: str | Path) -> Book:
    path = Path(path)
    notes: list[str] = []
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
        opf_dir = posixpath.dirname(opf_path)
        opf = ET.fromstring(z.read(opf_path))

        def find_all(root, name):
            return [el for el in root.iter() if el.tag.split("}")[-1] == name]

        title = next((el.text for el in find_all(opf, "title") if el.text), path.stem)
        author = next((el.text for el in find_all(opf, "creator") if el.text), "")
        language = next((el.text for el in find_all(opf, "language") if el.text), "en")
        manifest = {el.get("id"): el for el in find_all(opf, "item")}
        spine = [manifest[el.get("idref")] for el in find_all(opf, "itemref") if el.get("idref") in manifest]

        def full(href: str) -> str:
            return posixpath.normpath(posixpath.join(opf_dir, unquote(href)))

        paragraphs: list[Paragraph] = []
        index_of: dict[str, int] = {}
        for i, item in enumerate(spine):
            href = full(item.get("href"))
            index_of[href] = i
            try:
                html = z.read(href).decode("utf-8", errors="replace")
            except KeyError:
                continue
            c = _Collector(page=i)
            c.feed(html)
            c.close()
            paragraphs.extend(c.out)

        outline = _read_toc(z, opf, manifest, full, index_of)
        cover = _find_cover(z, opf, manifest, full)

    chapters = build_chapters(paragraphs, outline, notes)
    return Book(title=clean_unicode(title), author=clean_unicode(author or ""), source=str(path),
                chapters=chapters, cover=cover, language=(language or "en")[:2], notes=notes)


def _read_toc(z, opf, manifest, full, index_of) -> list[tuple[int, str, int]]:
    out: list[tuple[int, str, int]] = []
    ncx = next((it for it in manifest.values() if it.get("media-type") == "application/x-dtbncx+xml"), None)
    nav = next((it for it in manifest.values() if "nav" in (it.get("properties") or "").split()), None)
    try:
        if ncx is not None:
            root = ET.fromstring(z.read(full(ncx.get("href"))))
            base = posixpath.dirname(full(ncx.get("href")))

            def walk(el, level):
                for np in el:
                    if np.tag.split("}")[-1] != "navPoint":
                        continue
                    label = "".join(t for t in np.itertext() if t.strip()).strip()
                    lab_el = next((x for x in np.iter() if x.tag.split("}")[-1] == "text"), None)
                    label = (lab_el.text or label).strip() if lab_el is not None else label
                    content = next((x for x in np.iter() if x.tag.split("}")[-1] == "content"), None)
                    if content is not None:
                        src = posixpath.normpath(posixpath.join(base, unquote(content.get("src", "").split("#")[0])))
                        if src in index_of:
                            out.append((level, label, index_of[src]))
                    walk(np, level + 1)
            navmap = next((x for x in root.iter() if x.tag.split("}")[-1] == "navMap"), None)
            if navmap is not None:
                walk(navmap, 0)
        elif nav is not None:
            html = z.read(full(nav.get("href"))).decode("utf-8", errors="replace")
            base = posixpath.dirname(full(nav.get("href")))
            for href, label in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
                src = posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0])))
                label = clean_unicode(re.sub(r"<[^>]+>", "", label))
                if src in index_of and label:
                    out.append((0, label, index_of[src]))
    except Exception:  # noqa: BLE001 - a broken TOC just means we fall back to headings
        return []
    seen, uniq = set(), []
    for e in out:
        if e[2] not in seen:
            seen.add(e[2])
            uniq.append(e)
    return uniq


def _find_cover(z, opf, manifest, full) -> bytes | None:
    cover_id = None
    for el in opf.iter():
        if el.tag.split("}")[-1] == "meta" and el.get("name") == "cover":
            cover_id = el.get("content")
    item = manifest.get(cover_id) if cover_id else None
    if item is None:
        item = next((it for it in manifest.values() if "cover-image" in (it.get("properties") or "")), None)
    if item is None or not (item.get("media-type") or "").startswith("image/"):
        return None
    try:
        data = z.read(full(item.get("href")))
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((1400, 1400))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None
