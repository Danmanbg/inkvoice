"""Script -> audio. Plans utterances and pauses, synthesizes with a cache, assembles chapters."""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio as A
from .casting import cast, describe_character
from .engines.base import DELIVERY_GAIN_DB, Engine, Part, Utterance
from .model import NARRATOR, Script, Unit
from .textutil import Pronunciations, chunk_text, speakable_heading, tts_clean

DEFAULT_PAUSES = {
    "speaker_change": 0.22,    # between narration and dialogue inside a paragraph
    "paragraph": 0.65,
    "after_heading": 1.2,
    "scene_break": 1.8,
    "chapter_end": 2.2,
    "chapter_start": 0.6,
}
SIMPLE_TAG = re.compile(r"^[,.]?\s*(he|she|they|I|we|[A-Z][\w’'-]+(\s[A-Z][\w’'-]+)?)\s+"
                        r"(said|says|asked|replied|answered|added|whispered)[.,]?$|"
                        r"^[,.]?\s*(said|asked|replied)\s+[A-Z][\w’'-]+[.,]?$")


@dataclass
class RenderOptions:
    mode: str = "cast"                     # cast = voice per character, single = one narrator
    pauses: dict = field(default_factory=lambda: dict(DEFAULT_PAUSES))
    level_db: float = -20.0
    target_lufs: float = -18.0
    chapters: list[int] | None = None      # 0-based chapter indices; None = all included
    preview_seconds: float | None = None
    drop_dialogue_tags: bool = False       # cast mode: skip bare "he said."
    seed: int = 0
    qa: bool = False                       # transcribe each line and retake bad ones
    qa_threshold: float = 0.25
    retakes: int = 2
    min_lines_for_own_voice: int = 2


@dataclass
class Step:
    kind: str                              # utt | pause
    utt: Utterance | None = None
    seconds: float = 0.0
    gain_db: float = 0.0


class Progress:
    def __init__(self, total_chars: int, callback=None, log=print):
        self.total = max(1, total_chars)
        self.done = 0
        self.synth_chars = 0
        self.synth_time = 0.0
        self.audio_seconds = 0.0
        self.callback = callback
        self.log = log
        self.t0 = time.time()
        self.last_log = 0.0
        self.chapter = ""

    def eta(self) -> float | None:
        if self.synth_chars < 200:
            return None
        return max(0.0, (self.total - self.done) * self.synth_time / self.synth_chars)

    def tick(self, chars: int, synth_seconds: float | None, audio_seconds: float) -> None:
        self.done += chars
        self.audio_seconds += audio_seconds
        if synth_seconds is not None:
            self.synth_chars += chars
            self.synth_time += synth_seconds
        now = time.time()
        if self.callback:
            self.callback(self.status())
        if now - self.last_log > 15:
            self.last_log = now
            self.log(self.line())

    def status(self) -> dict:
        return {"fraction": min(1.0, self.done / self.total), "eta": self.eta(), "chapter": self.chapter,
                "audio_seconds": self.audio_seconds,
                "speed": (self.audio_seconds / max(1e-6, time.time() - self.t0))}

    def line(self) -> str:
        eta = self.eta()
        eta_s = _hms(eta) if eta is not None else "estimating..."
        return (f"  {min(100.0, 100 * self.done / self.total):5.1f}%  {self.chapter[:40]:40}  "
                f"audio {_hms(self.audio_seconds)}  ETA {eta_s}")


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Renderer:
    def __init__(self, script: Script, engine: Engine, workdir: str | Path, opts: RenderOptions | None = None,
                 pronunciations: Pronunciations | None = None, log=print, progress_cb=None):
        self.script = script
        self.engine = engine
        self.opts = opts or RenderOptions()
        self.workdir = Path(workdir)
        self.cache = self.workdir / "cache" / engine.name
        self.chapter_dir = self.workdir / "chapters" / engine.name
        self.pron = pronunciations or Pronunciations()
        self.log = log
        self.progress_cb = progress_cb
        self.voices = cast(script, engine, self.opts.mode, self.opts.min_lines_for_own_voice, log)
        self.sig = engine.signature()
        self._qa = None

    # ---------------------------------------------------------------- plan --
    def _prep(self, text: str) -> str:
        return self.pron.apply(tts_clean(text), "misaki" if self.engine.name == "kokoro" else None)

    def _part(self, u: Unit) -> Part:
        text = speakable_heading(u.text) if u.kind == "heading" else self._prep(u.text)
        return Part(text=text, kind=u.kind, emotion=u.emotion, intensity=u.intensity,
                    delivery=u.delivery, note=u.note, speaker=u.speaker)

    def _voice_for(self, u: Unit) -> str:
        if u.kind != "dialogue" or self.opts.mode == "single":
            return self.voices[NARRATOR]
        return self.voices.get(u.speaker) or self.voices.get("unknown") or self.voices[NARRATOR]

    def plan_chapter(self, ci: int) -> list[Step]:
        ch = self.script.chapters[ci]
        P = self.opts.pauses
        steps: list[Step] = [Step("pause", seconds=P["chapter_start"])]
        paras: list[list[Unit]] = []
        for u in ch.units:
            if paras and paras[-1][0].para == u.para:
                paras[-1].append(u)
            else:
                paras.append([u])
        texts = [" ".join(u.text for u in p) for p in paras]
        for pi, units in enumerate(paras):
            first = units[0]
            if first.kind == "break":
                if steps and steps[-1].kind == "pause":
                    steps[-1].seconds = max(steps[-1].seconds, P["scene_break"])
                else:
                    steps.append(Step("pause", seconds=P["scene_break"]))
                continue
            if self.opts.drop_dialogue_tags and self.opts.mode == "cast":
                has_dialogue = any(u.kind == "dialogue" for u in units)
                units = [u for u in units if not (has_dialogue and u.kind == "narration"
                                                  and SIMPLE_TAG.match(u.text.strip()))] or units
            groups: list[tuple[str, list[Unit]]] = []
            for u in units:
                v = self._voice_for(u)
                if groups and groups[-1][0] == v and u.kind != "heading" and groups[-1][1][-1].kind != "heading":
                    groups[-1][1].append(u)
                else:
                    groups.append((v, [u]))
            for gi, (voice, gunits) in enumerate(groups):
                speaker = NARRATOR if voice == self.voices[NARRATOR] and self.opts.mode == "single" else (
                    gunits[0].speaker if gunits[0].kind == "dialogue" else NARRATOR)
                parts = [self._part(u) for u in gunits if u.text.strip()]
                for chunk_parts in self._split_long(parts):
                    utt = Utterance(voice=voice, speaker=speaker, parts=chunk_parts,
                                    persona=describe_character(self.script, speaker) if speaker != NARRATOR else "",
                                    narrator_style=self.script.narrator_style,
                                    context_before=texts[pi - 1] if pi > 0 else "",
                                    context_after=texts[pi + 1] if pi + 1 < len(texts) else "",
                                    seed=self.opts.seed)
                    deliveries = {p.delivery for p in chunk_parts}
                    gain = DELIVERY_GAIN_DB.get(deliveries.pop(), 0.0) if len(deliveries) == 1 else 0.0
                    steps.append(Step("utt", utt=utt, gain_db=gain))
                if gi < len(groups) - 1:
                    steps.append(Step("pause", seconds=P["speaker_change"]))
            pause = P["after_heading"] if first.kind == "heading" else P["paragraph"]
            steps.append(Step("pause", seconds=pause))
        if steps and steps[-1].kind == "pause":
            steps[-1].seconds = max(steps[-1].seconds, P["chapter_end"])
        return steps

    def _split_long(self, parts: list[Part]) -> list[list[Part]]:
        limit = max(200, self.engine.max_chars) if self.engine.name in ("gemini", "elevenlabs") else 10 ** 9
        out, cur, size = [], [], 0
        for p in parts:
            pieces = chunk_text(p.text, limit) if len(p.text) > limit else [p.text]
            for t in pieces:
                q = Part(**{**p.__dict__, "text": t})
                if cur and size + len(t) > limit:
                    out.append(cur)
                    cur, size = [], 0
                cur.append(q)
                size += len(t)
        if cur:
            out.append(cur)
        return out

    # ------------------------------------------------------------ synth ----
    def _cache_path(self, utt: Utterance) -> Path:
        return self.cache / f"{utt.key(self.sig)}.flac"

    def _synth(self, step: Step) -> tuple[np.ndarray, float | None]:
        """Return processed audio for a step and the synthesis time (None if cached)."""
        utt = step.utt
        path = self._cache_path(utt)
        if path.is_file():
            a, sr = A.read_audio(path)
            return a, None
        t0 = time.time()
        best, best_score = None, None
        tries = 1 + (self.opts.retakes if (self.opts.qa and self.engine.stochastic) else 0)
        for k in range(tries):
            u = utt if k == 0 else Utterance(**{**utt.__dict__, "seed": utt.seed + 1000 * k})
            raw = self.engine.synthesize(u)
            sr = self.engine.sample_rate
            a = A.trim(raw.astype(np.float32), sr)
            if len(a) == 0:
                a = A.silence(0.25, sr)
            a = A.fade(A.level(a, sr, self.opts.level_db, step.gain_db), sr)
            if not self.opts.qa:
                best = a
                break
            score = self._qa_score(a, sr, utt.text)
            if best_score is None or score < best_score:
                best, best_score = a, score
            if score <= self.opts.qa_threshold:
                break
            self.log(f"  retake ({score:.0%} mismatch): {utt.text[:60]}...")
        if self.opts.qa and best_score is not None and best_score > self.opts.qa_threshold:
            with open(self.workdir / "qa_report.txt", "a", encoding="utf-8") as f:
                f.write(f"{best_score:.0%}\t{path.name}\t{utt.speaker}\t{utt.text}\n")
        A.write_audio(path, best, self.engine.sample_rate)
        return best, time.time() - t0

    def _qa_score(self, a: np.ndarray, sr: int, text: str) -> float:
        if self._qa is None:
            from .qa import Checker
            self._qa = Checker(log=self.log)
        return self._qa.mismatch(a, sr, text)

    # ------------------------------------------------------------ render ---
    def selected(self) -> list[int]:
        if self.opts.chapters is not None:
            return [i for i in self.opts.chapters if 0 <= i < len(self.script.chapters)]
        return [i for i, ch in enumerate(self.script.chapters) if ch.include]

    def render_chapter(self, ci: int, progress: Progress) -> Path:
        steps = self.plan_chapter(ci)
        sr = self.engine.sample_rate
        keys = [s.utt.key(self.sig) if s.kind == "utt" else f"p{s.seconds:.3f}" for s in steps]
        ch_key = hashlib.sha1(json.dumps([keys, self.opts.level_db, self.opts.target_lufs, A.TRUE_PEAK]).encode()).hexdigest()[:16]
        final = self.chapter_dir / f"{ci + 1:03d}_{ch_key}.wav"
        if final.is_file():
            progress.tick(sum(len(s.utt.text) for s in steps if s.kind == "utt"), None, A.duration(final))
            return final

        utt_steps = [s for s in steps if s.kind == "utt"]
        results: dict[int, np.ndarray] = {}
        todo = [s for s in utt_steps if not self._cache_path(s.utt).is_file()]
        if self.engine.parallel > 1 and len(todo) > 1:
            with ThreadPoolExecutor(self.engine.parallel) as pool:
                futs = {id(s): pool.submit(self._synth, s) for s in todo}
                for s in todo:
                    a, t = futs[id(s)].result()
                    results[id(s)] = a
                    progress.tick(len(s.utt.text), t, len(a) / sr)
        for s in utt_steps:
            if id(s) not in results:
                a, t = self._synth(s)
                results[id(s)] = a
                progress.tick(len(s.utt.text), t, len(a) / sr)

        pieces = [results[id(s)] if s.kind == "utt" else A.silence(s.seconds, sr) for s in steps]
        raw = self.chapter_dir / f"{ci + 1:03d}_{ch_key}.raw.wav"
        A.write_audio(raw, np.concatenate(pieces) if pieces else A.silence(1, sr), sr)
        A.loudnorm(raw, final, self.opts.target_lufs)
        raw.unlink(missing_ok=True)
        for old in self.chapter_dir.glob(f"{ci + 1:03d}_*.wav"):
            if old != final:
                old.unlink(missing_ok=True)
        return final

    def render(self) -> list[tuple[str, Path]]:
        chosen = self.selected()
        total = sum(len(u.text) for ci in chosen for u in self.script.chapters[ci].units)
        progress = Progress(total, self.progress_cb, self.log)
        out = []
        for n, ci in enumerate(chosen, 1):
            ch = self.script.chapters[ci]
            progress.chapter = f"[{n}/{len(chosen)}] {ch.title}"
            self.log(f"Chapter {n}/{len(chosen)}: {ch.title}")
            out.append((ch.title, self.render_chapter(ci, progress)))
        self.log(progress.line())
        return out

    def preview(self, seconds: float, start_chapter: int | None = None) -> np.ndarray:
        """Render the first `seconds` of a chapter quickly (no loudness pass)."""
        chosen = self.selected()
        ci = start_chapter if start_chapter is not None else (chosen[0] if chosen else 0)
        sr = self.engine.sample_rate
        pieces, total, last = [], 0.0, time.time()
        for s in self.plan_chapter(ci):
            if s.kind == "pause":
                pieces.append(A.silence(s.seconds, sr))
                total += s.seconds
                continue
            a, _ = self._synth(s)
            pieces.append(a)
            total += len(a) / sr
            if time.time() - last > 10:
                last = time.time()
                self.log(f"  preview: {total:.0f}/{seconds:.0f} s")
            if total >= seconds:
                break
        return np.concatenate(pieces) if pieces else A.silence(1, sr)

    def estimate(self) -> dict:
        chosen = self.selected()
        chars = sum(len(u.text) for ci in chosen for u in self.script.chapters[ci].units)
        words = sum(len(u.text.split()) for ci in chosen for u in self.script.chapters[ci].units)
        seconds = words / 155 * 60            # typical audiobook pace
        return {"chapters": len(chosen), "chars": chars, "words": words, "audio_seconds": seconds,
                "cost_usd": self.engine.estimate_cost(chars, seconds)}
