"""Speech engines. Each one turns an Utterance into float32 mono audio."""
from __future__ import annotations

from .base import Engine, EngineError, EngineInfo, Part, Utterance, Voice  # noqa: F401

ENGINES = {
    "kokoro": EngineInfo("kokoro", "Kokoro (free, offline, fast on CPU)", True,
                         "pace + volume + a distinct voice per character", ["kokoro"]),
    "chatterbox": EngineInfo("chatterbox", "Chatterbox (free, offline, expressive, slow on CPU)", True,
                             "real emotion control (exaggeration) + voice cloning", ["chatterbox-tts"]),
    "gemini": EngineInfo("gemini", "Google Gemini TTS (cloud, ~$0.8 per audio hour)", False,
                         "natural-language acting direction for every line", ["requests"]),
    "elevenlabs": EngineInfo("elevenlabs", "ElevenLabs v4 (cloud, best quality, ~$40-80 per novel)", False,
                             "inline audio tags like [whispers] [sad] [laughs]", ["requests"]),
}


def get_engine(name: str, settings: dict | None = None, models_dir=None, log=print, voices_dir=None) -> Engine:
    name = (name or "kokoro").lower()
    if name == "kokoro":
        from .kokoro_engine import KokoroEngine
        return KokoroEngine(settings, models_dir, log)
    if name == "chatterbox":
        from .chatterbox_engine import ChatterboxEngine
        return ChatterboxEngine(settings, models_dir, log, voices_dir=voices_dir)
    if name == "gemini":
        from .gemini_engine import GeminiEngine
        return GeminiEngine(settings, models_dir, log)
    if name == "elevenlabs":
        from .elevenlabs_engine import ElevenLabsEngine
        return ElevenLabsEngine(settings, models_dir, log)
    raise EngineError(f"Unknown engine '{name}'. Choose one of: {', '.join(ENGINES)}")
