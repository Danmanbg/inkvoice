"""Settings: defaults <- settings.toml <- environment <- command line."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

from .render import DEFAULT_PAUSES

DEFAULTS: dict = {
    "general": {
        "engine": "kokoro",          # kokoro | chatterbox | gemini | elevenlabs
        "director": "heuristic",     # heuristic | claude
        "mode": "cast",              # cast (voice per character) | single (one narrator)
        "formats": ["m4b"],          # m4b and/or mp3
        "output_dir": "",            # empty = next to the book
        "models_dir": "",            # empty = ~/.cache/inkvoice/models
        "voices_dir": "voices",      # your own reference clips for Chatterbox
        "pronunciations": "pronunciations.txt",
        "drop_dialogue_tags": False,
    },
    "text": {"pages": "", "keep_footnotes": False},
    "audio": {"bitrate": "64k", "target_lufs": -18.0, "level_db": -20.0},
    "pauses": dict(DEFAULT_PAUSES),
    "qa": {"enabled": False, "threshold": 0.25, "retakes": 2, "model": "base.en"},
    "kokoro": {"narrator": "af_heart", "speed": 1.0, "emotion_pace": 1.0, "threads": 0, "workers": 2},
    "chatterbox": {"variant": "standard", "narrator": "default", "exaggeration": 0.5,
                   "emotion_strength": 1.0, "temperature": 0.8, "threads": 0, "device": ""},
    "gemini": {"model": "gemini-3.8-flash-tts", "narrator": "Charon",
               "narrator_style": "a warm, engaging audiobook narrator", "api_key": "", "parallel": 4},
    "elevenlabs": {"model": "eleven_v4", "narrator": "", "stability": 0.5, "similarity": 0.75,
                   "style": 0.0, "speed": 1.0, "api_key": "", "parallel": 3},
    "claude": {"model": "claude-opus-5-5", "effort": "low", "api_key": "", "parallel": 4},
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def find_settings(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    for cand in (Path.cwd() / "settings.toml", Path(__file__).resolve().parent.parent / "settings.toml",
                 Path.home() / ".config" / "inkvoice" / "settings.toml"):
        if cand.is_file():
            return cand
    return None


def load_settings(path: str | None = None) -> dict:
    s = copy.deepcopy(DEFAULTS)
    p = find_settings(path)
    if p and p.is_file():
        with open(p, "rb") as f:
            _merge(s, tomllib.load(f))
        s["_file"] = str(p)
        base = p.parent
        for key in ("voices_dir", "pronunciations", "models_dir", "output_dir"):
            v = s["general"].get(key)
            if v and not Path(v).is_absolute():
                s["general"][key] = str((base / v).resolve())
    for section, env in (("gemini", "GEMINI_API_KEY"), ("elevenlabs", "ELEVENLABS_API_KEY"),
                         ("claude", "ANTHROPIC_API_KEY")):
        if not s[section].get("api_key") and os.environ.get(env):
            s[section]["api_key"] = os.environ[env]
    return s
