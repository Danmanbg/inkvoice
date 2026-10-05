"""Optional quality check: listen back with Whisper and flag lines the TTS garbled.

Expressive models sometimes skip words, repeat themselves or mumble. With --qa
each line is transcribed and compared with the text; bad takes are re-generated
(on engines where a new take differs) and anything still wrong is listed in
qa_report.txt so you can fix it in the script.
"""
from __future__ import annotations

import re
import threading

import numpy as np

from .audio import resample


def _words(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    s = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", s)          # audio tags
    return re.findall(r"[a-z0-9']+", s)


def word_error_rate(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1] / len(r)


class Checker:
    def __init__(self, model: str = "base.en", models_dir=None, log=print):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("Quality check needs faster-whisper:  pip install \"inkvoice[qa]\"") from e
        from .download import fetch_repo
        folder = fetch_repo(f"Systran/faster-whisper-{model}", models_dir=models_dir, log=log)
        self.model = WhisperModel(str(folder), device="cpu", compute_type="int8")
        self.lock = threading.Lock()

    def transcribe(self, a: np.ndarray, sr: int) -> str:
        x = resample(a, sr, 16000)
        with self.lock:
            segs, _ = self.model.transcribe(x, language="en", beam_size=1, without_timestamps=True,
                                            condition_on_previous_text=False)
            return " ".join(s.text for s in segs)

    def mismatch(self, a: np.ndarray, sr: int, text: str) -> float:
        return word_error_rate(text, self.transcribe(a, sr))
