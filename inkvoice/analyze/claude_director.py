"""Claude as the audiobook director.

Two passes:
  1. Casting - Claude reads the book and lists the characters (gender, age,
     a short voice description) plus how the narrator should sound.
  2. Direction - for every chapter, Claude labels each line of dialogue with
     its speaker, emotion, intensity, delivery and a short acting note.

The text itself is never rewritten: Claude only returns labels for numbered
units, so the audiobook always reads exactly what is in the book.
Needs an Anthropic API key (env ANTHROPIC_API_KEY or [claude] api_key).
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from ..model import AGES, DELIVERIES, EMOTIONS, NARRATOR, UNKNOWN, Character, Chapter, Script

PRICES = {  # USD per million tokens (input, output)
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}
WINDOW_UNITS = 260
CONTEXT_UNITS = 14
MAX_BOOK_CHARS = 1_400_000

CAST_SCHEMA = {
    "type": "object",
    "properties": {
        "narrator_style": {"type": "string"},
        "first_person_narrator": {"type": "string"},
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                "age": {"type": "string", "enum": AGES},
                "description": {"type": "string"},
            },
            "required": ["name", "aliases", "gender", "age", "description"],
            "additionalProperties": False,
        }},
    },
    "required": ["narrator_style", "first_person_narrator", "characters"],
    "additionalProperties": False,
}

DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "speaker": {"type": "string"},
                "emotion": {"type": "string", "enum": EMOTIONS},
                "intensity": {"type": "integer", "enum": [1, 2, 3]},
                "delivery": {"type": "string", "enum": DELIVERIES},
                "note": {"type": "string"},
            },
            "required": ["i", "speaker", "emotion", "intensity", "delivery", "note"],
            "additionalProperties": False,
        }},
        "new_characters": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                "age": {"type": "string", "enum": AGES},
                "description": {"type": "string"},
            },
            "required": ["name", "gender", "age", "description"],
            "additionalProperties": False,
        }},
    },
    "required": ["labels", "new_characters"],
    "additionalProperties": False,
}

CAST_SYSTEM = """You are casting an audiobook. Read the book and list every character who speaks at least once.

For each character give:
- name: the name the book most often uses for them (this becomes their ID)
- aliases: other names or titles the text uses for the same person ("Mr. Hale", "the captain")
- gender, age (child, teen, young, adult, middle-aged, elderly)
- description: one short line a voice director would use - personality and how they sound
  (e.g. "gruff, slow-spoken old sailor, dry humor"; "bright, quick-tongued girl of nine")

Also give:
- narrator_style: one line on how the narration should sound (point of view, tone, pacing),
  e.g. "close third person, wry and warm, unhurried"
- first_person_narrator: if the book is narrated in first person by a character, that character's
  name (their dialogue will be read by the narrator voice); otherwise an empty string."""

DIRECT_SYSTEM = """You are the director of an audiobook recording. You receive one stretch of a chapter split into numbered units:
  N = narration, D = dialogue (the quotation marks have been removed), H = heading.
Lines marked [context] come from just before this stretch; they are only there to help you, do not label them.

Return a label for EVERY D unit, and for N units only when the narration itself needs a special tone
(a tense, suspenseful passage; a moment of grief; breathless action). Unlabeled narration is read neutrally.

Label fields:
- speaker: the character's name exactly as in the cast list. For dialogue spoken by the first-person
  narrator use "narrator". For narration units always use "narrator". If someone not on the list speaks,
  give them a short stable name ("Ferryman", "Guard 2") and add them to new_characters.
  Use "unknown" only when it truly cannot be worked out.
- emotion: how the line is actually delivered, judged from the whole scene - what just happened, the
  dialogue tags, what the character wants. Ordinary lines are "neutral"; do not over-act.
- intensity: 1 = a hint, 2 = clearly emotional, 3 = only for extreme moments (screaming, sobbing, fury).
- delivery: normal, soft, whisper, loud or shout - only when the text implies it.
- note: an optional acting direction of at most 10 words a voice actor would find useful
  ("voice cracking", "through gritted teeth", "trying not to laugh"). Use "" when there is nothing to add.
  Never ask for sounds or words that are not in the text."""


def estimate_cost(script: Script, model: str = "claude-opus-5-5") -> tuple[float, int, int]:
    """Rough (cost in USD, input tokens, output tokens) for directing the included chapters."""
    chars = sum(len(u.text) for ch in script.chapters if ch.include for u in ch.units)
    n_dialogue = sum(1 for ch in script.chapters if ch.include for u in ch.units if u.kind == "dialogue")
    n_narr = sum(1 for ch in script.chapters if ch.include for u in ch.units if u.kind == "narration")
    tok_in = int(min(chars, MAX_BOOK_CHARS) / 3.3) + int(chars / 3.0) + 2500 * (len(script.chapters) + 2)
    tok_out = int((n_dialogue + 0.2 * n_narr) * 34 * 1.6) + 6000      # labels + some thinking
    pin, pout = PRICES.get(model, (4.0, 20.0))
    return (tok_in * pin + tok_out * pout) / 1e6, tok_in, tok_out


class ClaudeDirector:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", api_key: str | None = None,
                 parallel: int = 4, log=print, confirm=None, **_):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("The Claude director needs the Anthropic SDK:  pip install \"inkvoice[claude]\"") from e
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.parallel = max(1, int(parallel))
        self.log = log
        self.confirm = confirm
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- calls ---
    def _call(self, system: str, user: str, schema: dict, max_tokens: int = 32000) -> dict:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if not self.model.startswith("claude-haiku"):
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except self.anthropic.AuthenticationError as e:
            raise RuntimeError("Your Anthropic API key was rejected. Check [claude] api_key or ANTHROPIC_API_KEY.") from e
        except self.anthropic.NotFoundError as e:
            raise RuntimeError(f"Model '{self.model}' was not found for this API key.") from e
        except self.anthropic.BadRequestError as e:
            raise RuntimeError(f"Claude rejected the request: {e.message}") from e
        u = msg.usage
        with self._lock:
            self.usage["in"] += u.input_tokens or 0
            self.usage["out"] += u.output_tokens or 0
            self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        if msg.stop_reason == "refusal":
            raise RuntimeError("Claude declined to direct this part of the book.")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("Claude's answer was cut off (max_tokens) - try a smaller window.")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return json.loads(text)

    # ------------------------------------------------------------ cost ---
    def estimate(self, script: Script) -> tuple[float, int, int]:
        return estimate_cost(script, self.model)

    def spent(self) -> float:
        pin, pout = PRICES.get(self.model, (4.0, 20.0))
        u = self.usage
        return (u["in"] * pin + u["cache_write"] * pin * 1.25 + u["cache_read"] * pin * 0.1 + u["out"] * pout) / 1e6

    # ------------------------------------------------------------ main ---
    def direct(self, script: Script) -> None:
        cost, tin, tout = self.estimate(script)
        self.log(f"Claude director ({self.model}, effort {self.effort}): ~{tin / 1000:.0f}k input + "
                 f"~{tout / 1000:.0f}k output tokens, roughly ${cost:.2f}.")
        if self.confirm and not self.confirm(cost):
            raise RuntimeError("Cancelled.")
        self._cast(script)
        chapters = [ch for ch in script.chapters if ch.include]
        with ThreadPoolExecutor(self.parallel) as pool:
            futures = [pool.submit(self._direct_chapter, script, ch, n) for n, ch in enumerate(chapters, 1)]
            for f in futures:
                f.result()
        self.log(f"Claude director finished. Spent about ${self.spent():.2f}.")

    def _cast(self, script: Script) -> None:
        self.log("Claude is reading the book to cast the characters...")
        parts, size = [], 0
        for ch in script.chapters:
            if not ch.include:
                continue
            block = f"\n## {ch.title}\n" + "\n".join(
                (f"“{u.text}”" if u.kind == "dialogue" else u.text) for u in ch.units if u.kind != "break")
            if size + len(block) > MAX_BOOK_CHARS:
                parts.append("\n[... rest of the book omitted for length ...]")
                break
            parts.append(block)
            size += len(block)
        hints = ", ".join(sorted(script.characters)) or "(none)"
        user = (f"Book: {script.title}" + (f" by {script.author}" if script.author else "") +
                f"\nNames a simple tagger found (may be incomplete or wrong): {hints}\n\n<book>{''.join(parts)}\n</book>")
        data = self._call(CAST_SYSTEM, user, CAST_SCHEMA, max_tokens=32000)
        chars: dict[str, Character] = {}
        for c in data.get("characters", []):
            name = c["name"].strip()
            if not name or name.lower() in (NARRATOR, UNKNOWN):
                continue
            chars[name] = Character(name=name, gender=c.get("gender", "unknown"), age=c.get("age", "adult"),
                                    description=c.get("description", ""), aliases=list(c.get("aliases", [])))
        # keep any voice choices already made for the same names
        for name, c in chars.items():
            old = script.characters.get(name)
            if old:
                c.voices = old.voices
        script.characters = chars
        script.narrator_style = data.get("narrator_style", "")
        self.first_person = (data.get("first_person_narrator") or "").strip()
        self.log(f"Cast: {', '.join(chars) or 'no speaking characters'}")
        self._alias = {}
        for c in chars.values():
            self._alias[c.name.lower()] = c.name
            for a in c.aliases:
                self._alias.setdefault(a.lower(), c.name)

    def _roster(self, script: Script) -> str:
        lines = [f"- {c.name} ({c.gender}, {c.age}){' aka ' + ', '.join(c.aliases) if c.aliases else ''}: "
                 f"{c.description}" for c in script.characters.values()]
        fp = f"\nFirst-person narrator: {self.first_person} (their dialogue -> \"narrator\")" if self.first_person else ""
        return "Cast list:\n" + ("\n".join(lines) or "(none yet)") + fp

    def _direct_chapter(self, script: Script, ch: Chapter, n: int) -> None:
        units = ch.units
        idx = [i for i, u in enumerate(units) if u.kind in ("narration", "dialogue", "heading")]
        if not any(units[i].kind == "dialogue" for i in idx):
            for i in idx:
                units[i].speaker = NARRATOR
            return
        windows = [idx[a:a + WINDOW_UNITS] for a in range(0, len(idx), WINDOW_UNITS)]
        for w, window in enumerate(windows):
            ctx = idx[max(0, idx.index(window[0]) - CONTEXT_UNITS):idx.index(window[0])]
            lines = []
            last_para = None
            for i in ctx:
                u = units[i]
                who = f" ({u.speaker})" if u.kind == "dialogue" and u.speaker not in (NARRATOR, UNKNOWN) else ""
                lines.append(f"[context] {u.kind[0].upper()}{who}: {u.text}")
            for i in window:
                u = units[i]
                if last_para is not None and u.para != last_para:
                    lines.append("")
                last_para = u.para
                lines.append(f"[{i}] {u.kind[0].upper()}: {u.text}")
            user = (f"{self._roster(script)}\n\nBook: {script.title}\nChapter: {ch.title} "
                    f"(part {w + 1} of {len(windows)})\n\n" + "\n".join(lines))
            data = self._call(DIRECT_SYSTEM, user, DIRECT_SCHEMA, max_tokens=48000)
            self._apply(script, units, set(window), data)
        self.log(f"  directed chapter {n}: {ch.title}")

    def _apply(self, script: Script, units, allowed: set[int], data: dict) -> None:
        with self._lock:
            for c in data.get("new_characters", []):
                name = c["name"].strip()
                if name and name not in script.characters and name.lower() not in (NARRATOR, UNKNOWN):
                    script.characters[name] = Character(name=name, gender=c.get("gender", "unknown"),
                                                        age=c.get("age", "adult"), description=c.get("description", ""))
                    self._alias[name.lower()] = name
        for lab in data.get("labels", []):
            i = lab.get("i")
            if i not in allowed:
                continue
            u = units[i]
            spk = (lab.get("speaker") or "").strip()
            if u.kind == "dialogue":
                low = spk.lower()
                if low in (NARRATOR, "the narrator") or (self.first_person and low == self.first_person.lower()):
                    u.speaker = NARRATOR
                elif low in self._alias:
                    u.speaker = self._alias[low]
                elif spk and low != UNKNOWN:
                    u.speaker = spk
                    with self._lock:
                        script.characters.setdefault(spk, Character(name=spk))
                        self._alias[low] = spk
                else:
                    u.speaker = UNKNOWN
            else:
                u.speaker = NARRATOR
            u.emotion = lab.get("emotion") if lab.get("emotion") in EMOTIONS else "neutral"
            u.intensity = int(lab.get("intensity") or 1) if u.emotion != "neutral" else 1
            u.delivery = lab.get("delivery") if lab.get("delivery") in DELIVERIES else "normal"
            u.note = (lab.get("note") or "").strip()[:80]
        # narration units Claude skipped are neutral
        for i in allowed:
            u = units[i]
            if u.kind == "narration" and not any(l.get("i") == i for l in data.get("labels", [])):
                u.emotion, u.intensity, u.delivery, u.note = "neutral", 1, "normal", ""
