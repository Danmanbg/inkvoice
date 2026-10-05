"""Render pipeline with a fake engine (no models needed)."""
import shutil

import numpy as np
import pytest

from inkvoice.analyze import build_script
from inkvoice.engines.base import Engine, Voice
from inkvoice.model import Book, Paragraph, RawChapter
from inkvoice.render import Renderer, RenderOptions

pytest.importorskip("soundfile")


class ToneEngine(Engine):
    """Says every line as a short tone whose length follows the text."""
    name = "tone"
    calls = 0

    def voices(self):
        return [Voice("n", "female", quality=5), Voice("m1", "male"), Voice("m2", "male"), Voice("f1", "female")]

    def synthesize(self, utt):
        ToneEngine.calls += 1
        n = int(self.sample_rate * (0.2 + 0.01 * len(utt.text)))
        t = np.arange(n) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def book():
    paras = [Paragraph("Chapter One", "heading"),
             Paragraph("“Hello there,” Ada said. “Is anyone in?”"),
             Paragraph("“Only me,” Bram replied. Ada smiled at Bram."),
             Paragraph("* * *", "break"),
             Paragraph("The lamps went out one by one.")]
    return Book("T", "A", "t.txt", [RawChapter("Chapter One", paras)])


def test_plan_has_voices_and_pauses(tmp_path):
    script = build_script(book())
    r = Renderer(script, ToneEngine(), tmp_path, RenderOptions(), log=lambda *_: None)
    steps = r.plan_chapter(0)
    utts = [s.utt for s in steps if s.kind == "utt"]
    voices = {u.speaker: u.voice for u in utts}
    assert voices["narrator"] == "n"
    assert len({voices["narrator"], voices["Ada"], voices["Bram"]}) == 3      # everyone sounds different
    assert max(s.seconds for s in steps if s.kind == "pause") >= 1.8       # scene break / chapter end

    single = Renderer(script, ToneEngine(), tmp_path, RenderOptions(mode="single"), log=lambda *_: None)
    assert {s.utt.voice for s in single.plan_chapter(0) if s.kind == "utt"} == {"n"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_and_cache(tmp_path):
    from inkvoice.audio import duration
    from inkvoice.export import export_m4b

    script = build_script(book())
    r = Renderer(script, ToneEngine(), tmp_path / "work", RenderOptions(), log=lambda *_: None)
    out = r.render()
    assert len(out) == 1 and duration(out[0][1]) > 2
    first_calls = ToneEngine.calls
    Renderer(script, ToneEngine(), tmp_path / "work", RenderOptions(), log=lambda *_: None).render()
    assert ToneEngine.calls == first_calls                # second run is fully cached
    m4b = export_m4b(out, tmp_path / "book.m4b", "T", "A", None, workdir=tmp_path / "work")
    assert m4b.stat().st_size > 1000
