"""Google Gemini TTS (cloud). Takes natural-language acting direction per line.

Needs a key from https://aistudio.google.com/apikey  (env GEMINI_API_KEY or settings).
Roughly $0.8 per hour of audio on gemini-3.8-flash-tts (Oct 2026 prices); the free
tier is limited to ~100 requests/day, which is enough for testing but not a book.
"""
from __future__ import annotations

import base64
import io
import os
import re

import numpy as np

from .base import Engine, EngineError, Utterance, Voice, describe, retry_http

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
# (name, gender, character) - the prebuilt voices
_VOICES = [
    ("Zephyr", "female", "bright"), ("Puck", "male", "upbeat"), ("Charon", "male", "informative"),
    ("Kore", "female", "firm"), ("Fenrir", "male", "excitable"), ("Leda", "female", "youthful"),
    ("Orus", "male", "firm"), ("Aoede", "female", "breezy"), ("Callirrhoe", "female", "easy-going"),
    ("Autonoe", "female", "bright"), ("Enceladus", "male", "breathy"), ("Iapetus", "male", "clear"),
    ("Umbriel", "male", "easy-going"), ("Algieba", "male", "smooth"), ("Despina", "female", "smooth"),
    ("Erinome", "female", "clear"), ("Algenib", "male", "gravelly"), ("Rasalgethi", "male", "informative"),
    ("Laomedeia", "female", "upbeat"), ("Achernar", "female", "soft"), ("Alnilam", "male", "firm"),
    ("Schedar", "male", "even"), ("Gacrux", "female", "mature"), ("Pulcherrima", "female", "forward"),
    ("Achird", "male", "friendly"), ("Zubenelgenubi", "male", "casual"), ("Vindemiatrix", "female", "gentle"),
    ("Sadachbia", "male", "lively"), ("Sadaltager", "male", "knowledgeable"), ("Sulafat", "female", "warm"),
]
_GOOD = {"Charon", "Kore", "Sulafat", "Algieba", "Despina", "Gacrux", "Iapetus", "Schedar", "Achernar", "Algenib"}
# point-in-time sounds the model understands as inline tags
_TAGS = {"laugh": "<laugh>", "chuckl": "<chuckle>", "giggl": "<giggle>", "sigh": "<sigh>", "gasp": "<gasp>",
         "sob": "<sob>", "cry": "<cry>", "whimper": "<whimper>", "groan": "<groan>", "yawn": "<yawn>",
         "cough": "<cough>", "scream": "<scream>", "breath": "<breath>", "pant": "<pant>", "growl": "<growl>"}


class GeminiEngine(Engine):
    name = "gemini"
    sample_rate = 24000
    max_chars = 4000
    parallel = 4
    stochastic = True

    def __init__(self, settings=None, models_dir=None, log=print):
        super().__init__(settings, models_dir, log)
        self.model = self.settings.get("model") or "gemini-3.8-flash-tts"
        self.key = self.settings.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.parallel = int(self.settings.get("parallel", 4))

    def voices(self) -> list[Voice]:
        return [Voice(n, g, description=d, quality=5 if n in _GOOD else 4) for n, g, d in _VOICES]

    def default_narrator(self) -> str:
        return self.settings.get("narrator") or "Charon"

    def signature(self) -> dict:
        return {"engine": self.name, "model": self.model,
                "narrator_style": self.settings.get("narrator_style", "")}

    def estimate_cost(self, chars: int, seconds: float) -> float | None:
        per_hour = float(self.settings.get("usd_per_audio_hour", 0.81))
        return seconds / 3600 * per_hour

    # ------------------------------------------------------------ prompt --
    def _style(self, utt: Utterance) -> str:
        narrator_style = (utt.narrator_style or self.settings.get("narrator_style")
                          or "a warm, engaging audiobook narrator")
        is_narrator = utt.speaker in ("narrator", "")
        if is_narrator:
            who = f"Read as {narrator_style}."
        else:
            who = f"Perform the lines of {utt.speaker}" + (f" ({utt.persona})" if utt.persona else "") + \
                  " in an audiobook. Stay in character; sound natural, not theatrical."
        emotional = [p for p in utt.parts if p.emotion != "neutral" or p.delivery != "normal" or p.note]
        if not emotional:
            return who + " Natural, even storytelling pace."
        if len({(p.emotion, p.delivery) for p in utt.parts}) == 1 or len(utt.parts) == 1:
            m = utt.main
            return f"{who} Emotion: {describe(m.emotion, m.intensity, m.delivery, m.note)}."
        # mixed paragraph (single-narrator mode): narration calm, dialogue acted
        notes = []
        for p in emotional:
            snippet = " ".join(p.text.split()[:6])
            notes.append(f'"{snippet}..." {describe(p.emotion, p.intensity, p.delivery, p.note)}')
        return who + " Narration stays natural; act the dialogue: " + "; ".join(notes) + "."

    def _text(self, utt: Utterance) -> str:
        out = []
        for p in utt.parts:
            t = p.text
            note = p.note.lower()
            tag = next((tg for w, tg in _TAGS.items() if w in note), None)
            if tag:
                t = f"{tag} {t}"
            out.append(t)
        return re.sub(r"\s+", " ", " ".join(out)).strip()

    # -------------------------------------------------------- synthesize --
    def synthesize(self, utt: Utterance) -> np.ndarray:
        import requests

        if not self.key:
            raise EngineError("No Gemini API key. Get one at https://aistudio.google.com/apikey and set "
                              "GEMINI_API_KEY or [gemini] api_key in settings.toml.")
        body = {
            "model": self.model,
            "input": [{"type": "user_input", "content": [{
                "type": "text", "text": self._text(utt),
                "annotations": [{"type": "speech_metadata", "style": self._style(utt)}],
            }]}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": utt.voice or self.default_narrator()}]},
        }
        r = retry_http(lambda: requests.post(URL, json=body, timeout=300,
                                             headers={"x-goog-api-key": self.key}),
                       "Gemini TTS", self.log)
        data = _find_audio(r.json())
        if data is None:
            raise EngineError(f"Gemini returned no audio: {r.text[:400]}")
        return _decode_audio(base64.b64decode(data), self.sample_rate)


def _find_audio(obj):
    """Locate the base64 audio payload wherever the response puts it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "audio" in k.lower():
                if isinstance(v, dict) and isinstance(v.get("data"), str):
                    return v["data"]
                if isinstance(v, str) and len(v) > 1000:
                    return v
        if isinstance(obj.get("data"), str) and len(obj["data"]) > 1000 and "audio" in str(obj.get("mime_type", obj.get("mimeType", "audio"))):
            return obj["data"]
        for v in obj.values():
            hit = _find_audio(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_audio(v)
            if hit:
                return hit
    return None


def _decode_audio(raw: bytes, sr: int) -> np.ndarray:
    if raw[:4] == b"RIFF":
        import soundfile as sf
        a, file_sr = sf.read(io.BytesIO(raw), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if file_sr != sr:
            from ..audio import resample
            a = resample(a, file_sr, sr)
        return a.astype(np.float32)
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
