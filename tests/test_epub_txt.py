import zipfile

from inkvoice.extract import load_book

CONTAINER = """<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""
OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>The Clockmaker</dc:title>
<dc:creator>A. Writer</dc:creator><dc:language>en</dc:language></metadata>
<manifest>
 <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
 <item id="c1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>
 <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
</manifest>
<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"""
NAV = """<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>
<li><a href="text/ch1.xhtml">One: The Shop</a></li><li><a href="text/ch2.xhtml#top">Two: The Key</a></li>
</ol></nav></body></html>"""
CH = """<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>{title}</h1>
<p>{p1}<sup>3</sup></p><p>“{q},” said Odo.</p><hr/><p>{p2}</p>
<aside epub:type="footnote">A footnote that must not be read.</aside></body></html>"""


def make_epub(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/content.opf", OPF)
        z.writestr("OEBPS/nav.xhtml", NAV)
        z.writestr("OEBPS/text/ch1.xhtml", CH.format(
            title="One: The Shop", p1="The shop was full of ticking clocks, each a little out of step.",
            q="Mind the pendulums", p2="Nobody had wound the tall clock in years, yet it was right twice a day."))
        z.writestr("OEBPS/text/ch2.xhtml", CH.format(
            title="Two: The Key", p1="The key was warm, as if someone had only just let go of it.",
            q="Where did you find this", p2="Odo turned it over and over in his fingers."))


def test_epub(tmp_path):
    f = tmp_path / "clock.epub"
    make_epub(f)
    book = load_book(f)
    assert book.title == "The Clockmaker" and book.author == "A. Writer"
    assert [c.title for c in book.chapters] == ["One: The Shop", "Two: The Key"]
    text = " ".join(p.text for c in book.chapters for p in c.paragraphs)
    assert "footnote" not in text
    assert "step.3" not in text and "out of step." in text      # footnote marker dropped
    assert any(p.kind == "break" for p in book.chapters[0].paragraphs)


def test_txt(tmp_path):
    f = tmp_path / "story.txt"
    f.write_text("Chapter 1\n\nIt began with a letter.\nIt was short.\n\n* * *\n\nThe end came later.\n\n"
                 "Chapter 2\n\nA new day.\n", encoding="utf-8")
    book = load_book(f)
    assert [c.title for c in book.chapters] == ["Chapter 1", "Chapter 2"]
    assert book.chapters[0].paragraphs[1].text == "It began with a letter. It was short."
