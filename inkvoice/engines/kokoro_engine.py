"""Kokoro-82M: small, fast, runs on any CPU, very clean voices (Apache-2.0).

Kokoro has no emotion control, so emotion is expressed through pace and level,
and characters get distinct voices (including blends like "af_bella+af_sky").
"""
from __future__ import annotations

import threading

import numpy as np

from .base import Engine, EngineError, Utterance, Voice, pace_for

REPO = "hexgrad/Kokoro-82M"

# id, gender, accent, quality (from the Kokoro voice notes; higher = better)
_VOICES = [
    ("af_heart", "female", "US", 5), ("af_bella", "female", "US", 5), ("af_nicole", "female", "US", 4),
    ("af_aoede", "female", "US", 3), ("af_kore", "female", "US", 3), ("af_sarah", "female", "US", 3),
    ("af_nova", "female", "US", 3), ("af_alloy", "female", "US", 3), ("af_sky", "female", "US", 2),
    ("af_jessica", "female", "US", 2), ("af_river", "female", "US", 2),
    ("am_michael", "male", "US", 4), ("am_fenrir", "male", "US", 4), ("am_puck", "male", "US", 4),
    ("am_echo", "male", "US", 2), ("am_eric", "male", "US", 2), ("am_liam", "male", "US", 2),
    ("am_onyx", "male", "US", 2), ("am_adam", "male", "US", 1), ("am_santa", "male", "US", 1),
    ("bf_emma", "female", "UK", 4), ("bf_isabella", "female", "UK", 3), ("bf_alice", "female", "UK", 2),
    ("bf_lily", "female", "UK", 2),
    ("bm_george", "male", "UK", 3), ("bm_fable", "male", "UK", 3), ("bm_lewis", "male", "UK", 2),
    ("bm_daniel", "male", "UK", 2),
]
# A few blends that sound like new people - handy for books with big casts.
_BLENDS = [
    ("af_bella+af_nicole", "female", "US", 4), ("af_heart+bf_emma", "female", "UK", 4),
    ("af_sarah+af_kore", "female", "US", 3), ("am_michael+am_fenrir", "male", "US", 4),
    ("am_puck+bm_george", "male", "UK", 3), ("bm_fable+bm_george", "male", "UK", 3),
    ("am_michael+bm_lewis", "male", "UK", 3), ("am_fenrir+am_echo", "male", "US", 3),
]


class KokoroEngine(Engine):
    name = "kokoro"
    sample_rate = 24000
    max_chars = 600

    def __init__(self, settings=None, models_dir=None, log=print):
        super().__init__(settings, models_dir, log)
        # Two lines at once uses a laptop CPU better than one line on all cores.
        self.parallel = max(1, int(self.settings.get("workers", 2)))
        self._model = None
        self._pipes: dict[str, object] = {}
        self._voice_cache: dict[str, object] = {}
        self._folder = None
        self._lock = threading.Lock()

    def voices(self) -> list[Voice]:
        out = [Voice(v, g, accent=a, quality=q) for v, g, a, q in _VOICES]
        out += [Voice(v, g, accent=a, quality=q, description="blend") for v, g, a, q in _BLENDS]
        return out

    def valid_voice(self, voice: str) -> bool:
        known = {v for v, *_ in _VOICES}
        return bool(voice) and all(p in known for p in voice.replace(",", "+").split("+"))

    def default_narrator(self) -> str:
        return self.settings.get("narrator") or "af_heart"

    def signature(self) -> dict:
        return {"engine": self.name, "speed": float(self.settings.get("speed", 1.0)),
                "emotion_pace": float(self.settings.get("emotion_pace", 1.0))}

    # -------------------------------------------------------------- load --
    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from kokoro import KModel
        except ImportError as e:
            raise EngineError("Kokoro is not installed. Run:  pip install \"inkvoice[kokoro]\"") from e
        from ..download import fetch_repo
        self._folder = fetch_repo(REPO, ["kokoro-v1_0.pth", "config.json", "voices/a*.pt", "voices/b*.pt"],
                                  models_dir=self.models_dir, log=self.log)
        threads = int(self.settings.get("threads") or 0)
        if threads > 0:
            torch.set_num_threads(threads)
        self._model = KModel(repo_id=REPO, config=str(self._folder / "config.json"),
                             model=str(self._folder / "kokoro-v1_0.pth")).eval()

    def _pipe(self, lang: str):
        """Text -> phonemes only; the model runs separately so calls can overlap."""
        if lang not in self._pipes:
            from kokoro import KPipeline
            self._pipes[lang] = KPipeline(lang_code=lang, repo_id=REPO, model=False)
        return self._pipes[lang]

    def _voice_tensor(self, voice: str):
        if voice not in self._voice_cache:
            import torch
            parts = [p.strip() for p in voice.replace(",", "+").split("+") if p.strip()]
            tensors = []
            for p in parts:
                f = self._folder / "voices" / f"{p}.pt"
                if not f.is_file():
                    raise EngineError(f"Unknown Kokoro voice '{p}'. Run 'inkvoice voices --engine kokoro'.")
                tensors.append(torch.load(str(f), weights_only=True))
            self._voice_cache[voice] = torch.mean(torch.stack(tensors), dim=0)
        return self._voice_cache[voice]

    # -------------------------------------------------------- synthesize --
    def synthesize(self, utt: Utterance) -> np.ndarray:
        import torch

        voice = utt.voice or self.default_narrator()
        base = float(self.settings.get("speed", 1.0))
        strength = float(self.settings.get("emotion_pace", 1.0))

        def speed_of(p):
            return base * (1.0 + (pace_for(p.emotion, p.intensity) - 1.0) * strength)

        # Parts that share a speed are spoken in one go for natural flow.
        groups: list[tuple[float, list[str]]] = []
        for p in utt.parts:
            sp = round(speed_of(p), 3)
            if groups and abs(groups[-1][0] - sp) < 0.015:
                groups[-1][1].append(p.text)
            else:
                groups.append((sp, [p.text]))

        with self._lock:                      # loading and phonemizing are not thread-safe
            self._load()
            lang = "b" if voice.split("+")[0].startswith("b") else "a"
            pipe = self._pipe(lang)
            pack = self._voice_tensor(voice)
            jobs = [(r.phonemes, sp) for sp, texts in groups
                    for r in pipe(" ".join(texts), split_pattern=r"\n+") if r.phonemes]

        pieces = []
        gap = np.zeros(int(0.09 * self.sample_rate), dtype=np.float32)
        for ps, sp in jobs:
            with torch.inference_mode():
                audio = self._model(ps, pack[len(ps) - 1], sp)
            pieces += [audio.detach().cpu().numpy().astype(np.float32).reshape(-1), gap]
        if not pieces:
            return np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
        return np.concatenate(pieces[:-1])
