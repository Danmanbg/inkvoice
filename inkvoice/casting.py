"""Assign engine voices to the narrator and characters."""
from __future__ import annotations

from collections import Counter

from .engines.base import Engine, Voice
from .model import NARRATOR, UNKNOWN, Script


def cast(script: Script, engine: Engine, mode: str = "cast", min_lines: int = 2, log=print) -> dict[str, str]:
    """Return {speaker: voice} and remember the choices in the script (per engine)."""
    name = engine.name
    narrator = script.narrator_voices.get(name)
    if not narrator or not engine.valid_voice(narrator):
        narrator = engine.default_narrator()
        script.narrator_voices[name] = narrator
    mapping = {NARRATOR: narrator}
    if mode == "single":
        for c in script.characters:
            mapping[c] = narrator
        mapping[UNKNOWN] = narrator
        return mapping

    voices = engine.voices()
    narrator_ids = set(narrator.replace(",", "+").split("+")) | {narrator}
    pool = [v for v in voices if v.id not in narrator_ids]
    used = Counter()

    def pick(gender: str, avoid: set[str]) -> str:
        cands = [v for v in pool if (gender == "unknown" or v.gender in (gender, "unknown")) and v.id not in avoid]
        if not cands:
            cands = [v for v in pool if v.id not in avoid] or pool or voices
        cands.sort(key=lambda v: (used[v.id], -v.quality))
        choice: Voice = cands[0]
        used[choice.id] += 1
        return choice.id

    # Existing user/earlier choices win.
    for cname, c in script.characters.items():
        v = c.voices.get(name)
        if v and engine.valid_voice(v):
            mapping[cname] = v
            used[v] += 1

    main = sorted((c for c in script.characters.values() if c.name not in mapping),
                  key=lambda c: -c.lines)
    extras: dict[str, str] = {}
    for c in main:
        if c.lines < min_lines:
            g = c.gender if c.gender in ("male", "female") else "unknown"
            if g not in extras:
                extras[g] = pick(g, set(mapping.values()))
            mapping[c.name] = extras[g]
        else:
            mapping[c.name] = pick(c.gender, set(mapping.values()))
        c.voices[name] = mapping[c.name]
    if UNKNOWN not in mapping:
        mapping[UNKNOWN] = extras.get("unknown") or pick("unknown", {narrator})
    return mapping


def describe_character(script: Script, speaker: str) -> str:
    c = script.characters.get(speaker)
    if not c:
        return ""
    bits = []
    if c.age and c.age != "adult":
        bits.append(c.age)
    if c.gender in ("male", "female"):
        bits.append("man" if c.gender == "male" and c.age not in ("child", "teen") else
                    "woman" if c.gender == "female" and c.age not in ("child", "teen") else
                    "boy" if c.gender == "male" else "girl")
    s = " ".join(bits)
    if c.description:
        s = f"{s}; {c.description}" if s else c.description
    return s
