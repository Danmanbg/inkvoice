"""Common interface for speech engines."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

# How much each emotion speeds up (>1) or slows down (<1) delivery at intensity 2.
EMOTION_PACE = {
    "neutral": 1.0, "happy": 1.03, "excited": 1.08, "amused": 1.03, "playful": 1.04, "tender": 0.95,
    "calm": 0.96, "sad": 0.92, "angry": 1.05, "fearful": 1.07, "anxious": 1.06, "surprised": 1.05,
    "disgusted": 0.98, "sarcastic": 0.98, "cold": 0.97, "hesitant": 0.93, "pleading": 1.0,
    "determined": 1.0, "proud": 0.98, "tired": 0.9, "solemn": 0.92, "suspense": 0.94,
}
INTENSITY_SCALE = {1: 0.6, 2: 1.0, 3: 1.4}
# Level offsets (dB) applied after normalization so whispers stay quiet and shouts stand out.
DELIVERY_GAIN_DB = {"normal": 0.0, "whisper": -7.0, "soft": -3.5, "loud": 1.5, "shout": 3.0}

# Plain-English descriptions used by engines that take natural-language direction.
EMOTION_WORDS = {
    "neutral": "", "happy": "happy, warm", "excited": "excited, energetic", "amused": "amused, with a smile in the voice",
    "playful": "playful, teasing", "tender": "tender, gentle", "calm": "calm, measured", "sad": "sad, subdued",
    "angry": "angry", "fearful": "frightened", "anxious": "anxious, tense", "surprised": "surprised",
    "disgusted": "disgusted", "sarcastic": "sarcastic, dry", "cold": "cold, flat", "hesitant": "hesitant, unsure",
    "pleading": "pleading, desperate", "determined": "firm, determined", "proud": "proud, confident",
    "tired": "tired, weary", "solemn": "solemn, grave", "suspense": "low and suspenseful",
}
INTENSITY_WORDS = {1: "slightly ", 2: "", 3: "very "}
DELIVERY_WORDS = {"normal": "", "whisper": "whispering", "soft": "softly, quietly", "loud": "loudly",
                  "shout": "shouting"}


def pace_for(emotion: str, intensity: int) -> float:
    base = EMOTION_PACE.get(emotion, 1.0)
    return 1.0 + (base - 1.0) * INTENSITY_SCALE.get(intensity, 1.0)


def describe(emotion: str, intensity: int, delivery: str, note: str = "") -> str:
    """'very frightened, whispering - voice trembling'"""
    bits = []
    if EMOTION_WORDS.get(emotion):
        bits.append(INTENSITY_WORDS.get(intensity, "") + EMOTION_WORDS[emotion])
    if DELIVERY_WORDS.get(delivery):
        bits.append(DELIVERY_WORDS[delivery])
    s = ", ".join(bits)
    if note:
        s = f"{s} - {note}" if s else note
    return s


@dataclass
class Voice:
    id: str
    gender: str = "unknown"        # male / female / unknown
    age: str = "adult"             # child / young / adult / elderly
    accent: str = ""
    description: str = ""
    quality: int = 3               # 1 (poor) .. 5 (best); casting prefers better voices


@dataclass
class Part:
    text: str
    kind: str = "narration"        # narration / dialogue / heading
    emotion: str = "neutral"
    intensity: int = 1
    delivery: str = "normal"
    note: str = ""
    speaker: str = ""


@dataclass
class Utterance:
    """What one engine call should say: same voice, one paragraph at most."""
    voice: str
    speaker: str
    parts: list[Part]
    persona: str = ""              # character description for engines that take direction
    narrator_style: str = ""
    context_before: str = ""
    context_after: str = ""
    seed: int = 0

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.parts)

    @property
    def main(self) -> Part:
        """The most emotional part (drives single-setting engines)."""
        return max(self.parts, key=lambda p: (p.emotion != "neutral", p.intensity, p.delivery != "normal"))

    def key(self, engine_sig: dict) -> str:
        payload = {"e": engine_sig, "v": self.voice, "p": [asdict(p) for p in self.parts],
                   "persona": self.persona, "style": self.narrator_style, "seed": self.seed}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


class EngineError(RuntimeError):
    """A failure the user can act on (bad key, missing package...)."""


@dataclass
class EngineInfo:
    name: str
    label: str
    local: bool
    emotion: str                   # how emotion is controlled, for the UI
    needs: list[str] = field(default_factory=list)


class Engine:
    name = "base"
    sample_rate = 24000
    max_chars = 1000               # longest text per call
    parallel = 1                   # concurrent calls that make sense
    stochastic = False             # same input can give different takes (retakes help)
    multi_voice_per_call = False

    def __init__(self, settings: dict | None = None, models_dir=None, log=print):
        self.settings = dict(settings or {})
        self.models_dir = models_dir
        self.log = log

    # -- catalog --------------------------------------------------------
    def voices(self) -> list[Voice]:
        raise NotImplementedError

    def default_narrator(self) -> str:
        vs = sorted(self.voices(), key=lambda v: -v.quality)
        return vs[0].id if vs else ""

    def valid_voice(self, voice: str) -> bool:
        return any(v.id == voice for v in self.voices())

    # -- synthesis ------------------------------------------------------
    def signature(self) -> dict:
        """Settings that change the audio (part of the cache key)."""
        return {"engine": self.name}

    def synthesize(self, utt: Utterance) -> np.ndarray:
        raise NotImplementedError

    def estimate_cost(self, chars: int, seconds: float) -> float | None:
        return None

    def close(self) -> None:
        pass


def retry_http(call, what: str, log=print, tries: int = 6):
    """Retry transient HTTP failures (429/5xx/network) with backoff."""
    import requests

    delay = 4.0
    for attempt in range(tries):
        try:
            r = call()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == tries - 1:
                raise EngineError(f"{what}: network error ({e})") from e
            log(f"  {what}: network problem, retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        if r.status_code in (401, 403):
            raise EngineError(f"{what}: the API key was rejected ({r.status_code}). {r.text[:300]}")
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == tries - 1:
                raise EngineError(f"{what}: gave up after {tries} tries ({r.status_code}) {r.text[:300]}")
            wait = float(r.headers.get("retry-after") or delay)
            log(f"  {what}: busy/rate-limited ({r.status_code}), waiting {wait:.0f}s...")
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        if r.status_code >= 400:
            raise EngineError(f"{what}: request failed ({r.status_code}) {r.text[:500]}")
        return r
    raise EngineError(f"{what}: failed")
