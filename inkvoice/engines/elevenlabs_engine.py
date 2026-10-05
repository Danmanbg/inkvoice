"""ElevenLabs (cloud): the most expressive voices available, directed with audio tags.

Needs a key from https://elevenlabs.io (env ELEVENLABS_API_KEY or settings).
Eleven v4 reads inline tags such as [whispers], [sad], [laughs] and free-form ones
like [low, steady voice]. A tag carries forward until the next one, so we only
insert tags where the delivery changes. List price is ~$80 per million characters
(~$40 for a typical novel); eleven_v4_turbo is half that.
"""
from __future__ import annotations

import os

import numpy as np

from .base import Engine, EngineError, Utterance, Voice, retry_http

API = "https://api.elevenlabs.io/v1"
TAG_EMOTION = {
    "happy": "happy", "excited": "excited", "amused": "amused", "playful": "playful", "tender": "gently",
    "calm": "calm", "sad": "sad", "angry": "angry", "fearful": "scared", "anxious": "anxious",
    "surprised": "surprised", "disgusted": "disgusted", "sarcastic": "sarcastic", "cold": "coldly",
    "hesitant": "hesitant", "pleading": "pleading", "determined": "determined", "proud": "proud",
    "tired": "tired", "solemn": "solemn", "suspense": "low, suspenseful",
}
TAG_DELIVERY = {"whisper": "whispers", "soft": "softly", "loud": "loudly", "shout": "shouting"}
TAG_INTENSITY = {1: "slightly ", 2: "", 3: "very "}


class ElevenLabsEngine(Engine):
    name = "elevenlabs"
    sample_rate = 24000
    max_chars = 3000
    parallel = 3
    stochastic = True

    def __init__(self, settings=None, models_dir=None, log=print):
        super().__init__(settings, models_dir, log)
        self.model = self.settings.get("model") or "eleven_v4"
        self.key = self.settings.get("api_key") or os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
        self.parallel = int(self.settings.get("parallel", 3))
        self._voices: list[Voice] | None = None
        self._names: dict[str, str] = {}

    # ----------------------------------------------------------- catalog --
    def _need_key(self):
        if not self.key:
            raise EngineError("No ElevenLabs API key. Create one at https://elevenlabs.io/app/settings/api-keys "
                              "and set ELEVENLABS_API_KEY or [elevenlabs] api_key in settings.toml.")

    def voices(self) -> list[Voice]:
        if self._voices is None:
            import requests
            self._need_key()
            r = retry_http(lambda: requests.get(f"{API}/voices", headers={"xi-api-key": self.key}, timeout=60),
                           "ElevenLabs voices", self.log)
            out = []
            for v in r.json().get("voices", []):
                lab = v.get("labels") or {}
                g = (lab.get("gender") or "unknown").lower()
                age = (lab.get("age") or "adult").lower().replace("middle aged", "middle-aged")
                use = (lab.get("use_case") or lab.get("use case") or "").lower()
                desc = ", ".join(x for x in (lab.get("description"), lab.get("accent"), use) if x)
                q = 5 if "narrat" in use or "audiobook" in use else 4
                out.append(Voice(v["voice_id"], g if g in ("male", "female") else "unknown", age=age,
                                 accent=lab.get("accent", ""), description=f"{v.get('name')}: {desc}", quality=q))
                self._names[v.get("name", "").lower()] = v["voice_id"]
            self._voices = out
        return self._voices

    def _resolve(self, voice: str) -> str:
        if not voice:
            voice = self.default_narrator()
        self.voices()
        return self._names.get(voice.lower(), voice)

    def valid_voice(self, voice: str) -> bool:
        try:
            vs = self.voices()
        except EngineError:
            return bool(voice)
        return any(v.id == voice for v in vs) or voice.lower() in self._names

    def default_narrator(self) -> str:
        if self.settings.get("narrator"):
            return self.settings["narrator"]
        vs = sorted(self.voices(), key=lambda v: (-v.quality, v.gender != "male"))
        if not vs:
            raise EngineError("Your ElevenLabs account has no voices - add one in the Voice Library first.")
        return vs[0].id

    def signature(self) -> dict:
        return {"engine": self.name, "model": self.model, "stability": float(self.settings.get("stability", 0.5)),
                "similarity": float(self.settings.get("similarity", 0.75)), "style": float(self.settings.get("style", 0.0)),
                "speed": float(self.settings.get("speed", 1.0))}

    def estimate_cost(self, chars: int, seconds: float) -> float | None:
        per_m = float(self.settings.get("usd_per_million_chars", 40 if "turbo" in self.model else 80))
        return chars / 1e6 * per_m

    # -------------------------------------------------------------- text --
    @staticmethod
    def _tag(p) -> str:
        bits = []
        if p.delivery in TAG_DELIVERY:
            bits.append(TAG_DELIVERY[p.delivery])
        if p.emotion in TAG_EMOTION:
            bits.append(TAG_INTENSITY.get(p.intensity, "") + TAG_EMOTION[p.emotion])
        if p.note:
            bits.append(p.note)
        return "[" + ", ".join(bits) + "]" if bits else ""

    def _text(self, utt: Utterance) -> str:
        out, current = [], ""
        for p in utt.parts:
            tag = self._tag(p)
            if tag != current:
                out.append(tag if tag else ("[calm]" if current else ""))
                current = tag
            out.append(p.text)
        return " ".join(x for x in out if x).strip()

    # -------------------------------------------------------- synthesize --
    def synthesize(self, utt: Utterance) -> np.ndarray:
        import requests

        self._need_key()
        voice_id = self._resolve(utt.voice)
        body = {
            "text": self._text(utt),
            "model_id": self.model,
            "voice_settings": {
                "stability": float(self.settings.get("stability", 0.5)),
                "similarity_boost": float(self.settings.get("similarity", 0.75)),
                "style": float(self.settings.get("style", 0.0)),
                "use_speaker_boost": True,
                "speed": float(self.settings.get("speed", 1.0)),
            },
            "seed": utt.seed & 0xFFFFFFFF,
        }
        if self.settings.get("use_context", True):
            if utt.context_before:
                body["previous_text"] = utt.context_before[-1000:]
            if utt.context_after:
                body["next_text"] = utt.context_after[:1000]
        url = f"{API}/text-to-speech/{voice_id}?output_format=pcm_24000"
        headers = {"xi-api-key": self.key, "Content-Type": "application/json"}
        try:
            r = retry_http(lambda: requests.post(url, json=body, headers=headers, timeout=300),
                           "ElevenLabs TTS", self.log)
        except EngineError as e:
            bad_request = "(400)" in str(e) or "(422)" in str(e)
            if bad_request and ("previous_text" in body or "next_text" in body):   # model rejects context fields
                body.pop("previous_text", None)
                body.pop("next_text", None)
                self.settings["use_context"] = False
                r = retry_http(lambda: requests.post(url, json=body, headers=headers, timeout=300),
                               "ElevenLabs TTS", self.log)
            else:
                raise e
        return np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
