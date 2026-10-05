"""Chatterbox (Resemble AI, MIT): expressive local TTS with voice cloning.

Variants
  standard  500M, real emotion control ("exaggeration")  - best acting, slowest
  turbo     350M, faster, no exaggeration knob             - emotion comes from the text

Voices are short reference recordings (5-15 s). Three sources:
  default        the voice that ships with the model
  <file>.wav     any clip in the voices folder (only use voices you have the right to use)
  k:<kokoro id>  a clip rendered once with Kokoro, so you get ~30 distinct people for free
"""
from __future__ import annotations

import os
import random
import threading
from pathlib import Path

import numpy as np

from ..textutil import chunk_text
from .base import Engine, EngineError, Utterance, Voice

REPOS = {
    "standard": ("ResembleAI/chatterbox", ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors",
                                            "tokenizer.json", "conds.pt"]),
    "turbo": ("ResembleAI/chatterbox-turbo", ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]),
}
# Original reference passage read by Kokoro to create cloning clips.
REFERENCE_TEXT = ("The morning was quiet, and the old road wound slowly through the hills. "
                  "I remember thinking that every long journey starts with a small and ordinary step, "
                  "and that this one had been waiting for me for a very long time.")
HIGH_AROUSAL = {"excited", "angry", "fearful", "surprised", "anxious", "pleading", "amused", "playful",
                "disgusted", "happy", "determined"}
LOW_AROUSAL = {"sad", "tired", "tender", "calm", "solemn", "suspense", "hesitant", "cold"}
TAG_WORDS = {"laugh": "[laugh]", "chuckl": "[chuckle]", "cough": "[cough]", "sigh": "[sigh]", "gasp": "[gasp]"}


class ChatterboxEngine(Engine):
    name = "chatterbox"
    sample_rate = 24000
    max_chars = 280
    stochastic = True

    def __init__(self, settings=None, models_dir=None, log=print, voices_dir=None):
        super().__init__(settings, models_dir, log)
        self.variant = (self.settings.get("variant") or "standard").lower()
        if self.variant not in REPOS:
            raise EngineError(f"Unknown Chatterbox variant '{self.variant}' (standard or turbo)")
        self.voices_dir = Path(voices_dir or self.settings.get("voices_dir") or "voices")
        self._model = None
        self._conds: dict[str, object] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- catalog --
    def voices(self) -> list[Voice]:
        from .kokoro_engine import _VOICES
        out = [Voice("default", "female", description="built-in Chatterbox voice", quality=4)]
        if self.voices_dir.is_dir():
            for f in sorted(self.voices_dir.glob("*.wav")):
                g = "female" if f.stem.lower().startswith(("f_", "female")) else (
                    "male" if f.stem.lower().startswith(("m_", "male")) else "unknown")
                out.append(Voice(f.stem, g, description=f"your clip {f.name}", quality=5))
        out += [Voice(f"k:{v}", g, accent=a, description="cloned from Kokoro", quality=q)
                for v, g, a, q in _VOICES if q >= 3]
        return out

    def valid_voice(self, voice: str) -> bool:
        return voice == "default" or voice.startswith("k:") or (self.voices_dir / f"{voice}.wav").is_file()

    def default_narrator(self) -> str:
        return self.settings.get("narrator") or "default"

    def signature(self) -> dict:
        return {"engine": self.name, "variant": self.variant,
                "exaggeration": float(self.settings.get("exaggeration", 0.5)),
                "emotion_strength": float(self.settings.get("emotion_strength", 1.0)),
                "temperature": float(self.settings.get("temperature", 0.8))}

    # -------------------------------------------------------------- load --
    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            if self.variant == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
            else:
                from chatterbox.tts import ChatterboxTTS as Model
        except ImportError as e:
            raise EngineError("Chatterbox is not installed. Run:  pip install \"inkvoice[chatterbox]\"") from e
        from ..download import fetch_repo
        repo, patterns = REPOS[self.variant]
        folder = fetch_repo(repo, patterns, models_dir=self.models_dir, log=self.log)
        device = self.settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        threads = int(self.settings.get("threads") or 0)
        if threads > 0:
            torch.set_num_threads(threads)
        self.log(f"Loading Chatterbox {self.variant} on {device}...")
        _quiet_progress_bars()
        self._model = Model.from_local(folder, device)
        self._default_conds = self._model.conds

    def _reference_wav(self, voice: str) -> Path | None:
        if voice == "default":
            return None
        if voice.startswith("k:"):
            kid = voice[2:]
            ref_dir = Path(self.models_dir or ".") / "chatterbox_refs"
            ref_dir.mkdir(parents=True, exist_ok=True)
            path = ref_dir / f"{kid}.wav"
            if not path.is_file():
                from ..audio import write_audio
                from .base import Part
                from .kokoro_engine import KokoroEngine
                self.log(f"  making a reference clip for voice {voice} with Kokoro...")
                k = KokoroEngine({}, models_dir=self.models_dir, log=self.log)
                audio = k.synthesize(Utterance(voice=kid, speaker="ref", parts=[Part(REFERENCE_TEXT)]))
                write_audio(path, audio, k.sample_rate)
            return path
        path = self.voices_dir / f"{voice}.wav"
        if not path.is_file():
            raise EngineError(f"Voice clip not found: {path}")
        return path

    def _use_voice(self, voice: str, exaggeration: float):
        key = voice
        if key not in self._conds:
            ref = self._reference_wav(voice)
            if ref is None:
                self._conds[key] = self._default_conds
            else:
                self._model.prepare_conditionals(str(ref), exaggeration=exaggeration)
                self._conds[key] = self._model.conds
        self._model.conds = self._conds[key]

    # -------------------------------------------------------- synthesize --
    def _emotion_params(self, part) -> tuple[float, float]:
        base = float(self.settings.get("exaggeration", 0.5))
        strength = float(self.settings.get("emotion_strength", 1.0))
        e, i = part.emotion, part.intensity
        if part.delivery in ("shout", "loud"):
            e, i = (e if e != "neutral" else "excited"), max(i, 2)
        if e in HIGH_AROUSAL:
            exag = base + strength * {1: 0.15, 2: 0.35, 3: 0.6}[i]
        elif e in LOW_AROUSAL or part.delivery in ("whisper", "soft"):
            exag = base - strength * {1: 0.05, 2: 0.12, 3: 0.18}[i]
        else:
            exag = base
        exag = float(min(1.6, max(0.25, exag)))
        cfg = 0.5 if exag <= 0.7 else 0.35          # high exaggeration speeds speech up; less CFG evens it out
        if e in LOW_AROUSAL:
            cfg = 0.55
        return exag, cfg

    def synthesize(self, utt: Utterance) -> np.ndarray:
        import torch
        with self._lock:
            self._load()
            voice = utt.voice or self.default_narrator()
            pieces = []
            gap = np.zeros(int(0.18 * self.sample_rate), dtype=np.float32)
            temp = float(self.settings.get("temperature", 0.8))
            for k, part in enumerate(utt.parts):
                exag, cfg = self._emotion_params(part)
                self._use_voice(voice, exag)
                text = part.text
                note = part.note.lower()
                tag = next((t for w, t in TAG_WORDS.items() if w in note), None)
                if tag and self.variant == "turbo":
                    text = f"{tag} {text}"
                for j, chunk in enumerate(chunk_text(text, self.max_chars)):
                    seed = (utt.seed * 1000 + k * 37 + j) & 0x7FFFFFFF
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed % (2 ** 32))
                    if self.variant == "turbo":
                        wav = self._model.generate(chunk, temperature=temp)
                    else:
                        wav = self._model.generate(chunk, exaggeration=exag, cfg_weight=cfg, temperature=temp)
                    a = wav.squeeze().detach().cpu().numpy().astype(np.float32)
                    pieces += [a, gap]
            if not pieces:
                return np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
            return np.concatenate(pieces[:-1])


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _quiet_progress_bars() -> None:
    """Chatterbox draws a progress bar for every line it speaks; hide them."""
    import importlib
    for mod in ("chatterbox.models.t3.t3", "chatterbox.models.s3gen.flow_matching"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "tqdm"):
                m.tqdm = lambda it, *a, **k: it
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
