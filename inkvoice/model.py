"""Data model shared by every stage.

Pipeline:  Book (extracted text)  ->  Script (who says what, how)  ->  audio.

The Script is saved as JSON next to the audiobook so it can be edited by hand
(change a speaker, an emotion, fix a word) and re-rendered. Only the units that
changed are synthesized again.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Emotions the director can assign. Every engine maps these to whatever control
# it has (style prompts, audio tags, exaggeration, speed...).
EMOTIONS = [
    "neutral", "happy", "excited", "amused", "playful", "tender", "calm",
    "sad", "angry", "fearful", "anxious", "surprised", "disgusted",
    "sarcastic", "cold", "hesitant", "pleading", "determined", "proud",
    "tired", "solemn", "suspense",
]
DELIVERIES = ["normal", "whisper", "soft", "loud", "shout"]
GENDERS = ["male", "female", "unknown"]
AGES = ["child", "teen", "young", "adult", "middle-aged", "elderly"]

NARRATOR = "narrator"
UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Extracted book
# --------------------------------------------------------------------------
@dataclass
class Paragraph:
    text: str
    kind: str = "text"          # text | heading | break
    page: int = 0               # 0-based source page (PDF only)


@dataclass
class RawChapter:
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.paragraphs if p.kind == "text")


@dataclass
class Book:
    title: str
    author: str
    source: str
    chapters: list[RawChapter]
    cover: bytes | None = None   # JPEG bytes
    language: str = "en"
    notes: list[str] = field(default_factory=list)   # extraction warnings

    def word_count(self) -> int:
        return sum(c.word_count() for c in self.chapters)


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------
@dataclass
class Unit:
    """One stretch of text with a single speaker and a single delivery."""
    kind: str                    # heading | narration | dialogue | break
    text: str
    para: int = 0                # paragraph index inside the chapter
    speaker: str = NARRATOR
    emotion: str = "neutral"
    intensity: int = 1           # 1 = a hint, 2 = clear, 3 = strong
    delivery: str = "normal"
    note: str = ""               # short acting note ("voice cracking")
    cont: bool = False           # dialogue continues a quote from the previous paragraph

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "text": self.text, "para": self.para}
        if self.kind in ("narration", "dialogue", "heading"):
            d["speaker"] = self.speaker
            if self.emotion != "neutral":
                d["emotion"] = self.emotion
                d["intensity"] = self.intensity
            if self.delivery != "normal":
                d["delivery"] = self.delivery
            if self.note:
                d["note"] = self.note
            if self.cont:
                d["cont"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Unit":
        u = cls(kind=d["kind"], text=d.get("text", ""), para=int(d.get("para", 0)))
        u.speaker = d.get("speaker", NARRATOR) or NARRATOR
        u.emotion = d.get("emotion", "neutral") if d.get("emotion") in EMOTIONS else "neutral"
        u.intensity = max(1, min(3, int(d.get("intensity", 1))))
        u.delivery = d.get("delivery", "normal") if d.get("delivery") in DELIVERIES else "normal"
        u.note = d.get("note", "") or ""
        u.cont = bool(d.get("cont", False))
        return u


@dataclass
class Character:
    name: str
    gender: str = "unknown"
    age: str = "adult"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    voices: dict[str, str] = field(default_factory=dict)   # engine -> voice (casting or user)
    lines: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("name")
        return d


@dataclass
class Chapter:
    title: str
    units: list[Unit] = field(default_factory=list)
    include: bool = True

    def word_count(self) -> int:
        return sum(len(u.text.split()) for u in self.units)


@dataclass
class Script:
    title: str
    author: str
    source: str
    chapters: list[Chapter]
    characters: dict[str, Character] = field(default_factory=dict)
    narrator_voices: dict[str, str] = field(default_factory=dict)   # engine -> voice
    narrator_style: str = ""     # e.g. "first-person, wry, intimate"
    director: str = "heuristic"
    language: str = "en"
    version: int = 1

    # ---------------------------------------------------------------- io --
    def save(self, path: str | Path) -> None:
        """Write JSON with one unit per line so the file stays hand-editable."""
        path = Path(path)
        dumps = lambda o: json.dumps(o, ensure_ascii=False)  # noqa: E731
        out = ["{"]
        head = {
            "version": self.version, "title": self.title, "author": self.author,
            "source": self.source, "language": self.language, "director": self.director,
            "narrator_voices": self.narrator_voices, "narrator_style": self.narrator_style,
        }
        for k, v in head.items():
            out.append(f"  {dumps(k)}: {dumps(v)},")
        out.append('  "characters": {')
        chars = list(self.characters.values())
        for i, c in enumerate(chars):
            comma = "," if i < len(chars) - 1 else ""
            out.append(f"    {dumps(c.name)}: {dumps(c.to_dict())}{comma}")
        out.append("  },")
        out.append('  "chapters": [')
        for ci, ch in enumerate(self.chapters):
            out.append("    {")
            out.append(f'      "title": {dumps(ch.title)},')
            out.append(f'      "include": {dumps(ch.include)},')
            out.append('      "units": [')
            for ui, u in enumerate(ch.units):
                comma = "," if ui < len(ch.units) - 1 else ""
                out.append(f"        {dumps(u.to_dict())}{comma}")
            out.append("      ]")
            out.append("    }" + ("," if ci < len(self.chapters) - 1 else ""))
        out.append("  ]")
        out.append("}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "Script":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        chars = {}
        for name, cd in (d.get("characters") or {}).items():
            chars[name] = Character(
                name=name, gender=cd.get("gender", "unknown"), age=cd.get("age", "adult"),
                description=cd.get("description", ""), aliases=list(cd.get("aliases", [])),
                voices=dict(cd.get("voices") or {}), lines=int(cd.get("lines", 0)))
        chapters = [
            Chapter(title=c.get("title", ""), include=bool(c.get("include", True)),
                    units=[Unit.from_dict(u) for u in c.get("units", [])])
            for c in d.get("chapters", [])
        ]
        return cls(
            title=d.get("title", ""), author=d.get("author", ""), source=d.get("source", ""),
            chapters=chapters, characters=chars, narrator_voices=dict(d.get("narrator_voices") or {}),
            narrator_style=d.get("narrator_style", ""), director=d.get("director", "heuristic"),
            language=d.get("language", "en"), version=int(d.get("version", 1)))

    # ----------------------------------------------------------- helpers --
    def count_lines(self) -> None:
        for c in self.characters.values():
            c.lines = 0
        for ch in self.chapters:
            for u in ch.units:
                if u.kind == "dialogue" and u.speaker in self.characters:
                    self.characters[u.speaker].lines += 1

    def word_count(self) -> int:
        return sum(ch.word_count() for ch in self.chapters if ch.include)

    def char_count(self) -> int:
        return sum(len(u.text) for ch in self.chapters if ch.include for u in ch.units)
