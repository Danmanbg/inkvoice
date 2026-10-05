"""Small text helpers: sentence splitting, chunking, TTS clean-up, pronunciations."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "gen", "col", "capt",
    "lt", "sgt", "mt", "ft", "vs", "etc", "e.g", "i.e", "no", "vol", "ch", "fig",
    "approx", "dept", "est", "inc", "ltd", "co", "corp", "ave", "blvd", "rd", "sq",
}

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

ROMAN = re.compile(r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def clean_unicode(text: str) -> str:
    """Normalize ligatures/odd spaces but keep typographic quotes and dashes."""
    text = text.translate(_ZERO_WIDTH)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t  -   　]+", " ", text)
    return text.strip()


def roman_to_int(s: str) -> int | None:
    if not s or not ROMAN.match(s):
        return None
    total, prev = 0, 0
    for ch in reversed(s.lower()):
        v = _ROMAN_VALUES[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


_NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                 "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_to_words(n: int) -> str:
    if n < 20:
        return _NUMBER_WORDS[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + ("-" + _NUMBER_WORDS[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _NUMBER_WORDS[h] + " hundred" + (" " + number_to_words(r) if r else "")
    return str(n)


def speakable_heading(title: str) -> str:
    """'CHAPTER IV' -> 'Chapter Four.'  '12' -> 'Twelve.'  Keeps real titles intact."""
    t = clean_unicode(title).strip()
    parts = []
    for piece in re.split(r"\s*[\n:|]\s*|\s{2,}", t):
        piece = piece.strip(" .")
        if not piece:
            continue
        if piece.isupper() and len(piece) > 3:
            piece = piece.title()
        words = piece.split()
        out = []
        for i, w in enumerate(words):
            core = w.strip(".,;:")
            val = None
            if core.isdigit() and len(core) <= 3:
                val = int(core)
            elif i > 0 and words[i - 1].lower().strip(".") in ("chapter", "part", "book", "volume", "act"):
                val = roman_to_int(core)
            elif len(words) == 1:
                val = roman_to_int(core) if len(core) > 1 or core.upper() in ("I", "V", "X") else None
            out.append(number_to_words(val).title() if val is not None else w)
        parts.append(" ".join(out))
    return ". ".join(parts) + "." if parts else ""


# ------------------------------------------------------------- sentences --
_SENT_END = re.compile(r"""([.!?…]+["'”’)\]]*)(\s+)(?=["'“‘(\[]?[A-Z0-9])""")


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, ignoring common abbreviations and initials."""
    text = text.strip()
    if not text:
        return []
    out, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        before = text[start:m.start(1)]
        last_word = re.split(r"[\s\"'“‘(]+", before)[-1].lower() if before else ""
        if m.group(1).startswith(".") and (
                last_word.rstrip(".") in ABBREVIATIONS
                or (len(last_word) == 1 and last_word.isalpha())):   # initials: "J. R. Smith"
            continue
        out.append(text[start:end].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Group sentences into chunks of at most max_chars (long sentences split on commas)."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for s in split_sentences(text):
        if len(s) <= max_chars:
            pieces.append(s)
            continue
        cur = ""
        for part in re.split(r"(?<=[,;:—])\s+", s):
            if cur and len(cur) + len(part) + 1 > max_chars:
                pieces.append(cur)
                cur = part
            else:
                cur = f"{cur} {part}".strip()
        while len(cur) > max_chars:          # pathological: no punctuation at all
            cut = cur.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(cur[:cut])
            cur = cur[cut:].strip()
        if cur:
            pieces.append(cur)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------ tts prep --
def tts_clean(text: str) -> str:
    """Make text friendlier for speech engines without changing its words."""
    t = clean_unicode(text)
    t = re.sub(r"\[\d+\]|\(\d+\)(?=\W|$)", "", t)           # [12] style footnote refs
    t = re.sub(r"(?<=\w)[¹²³⁰-⁹]+", "", t)   # superscript digits
    t = re.sub(r"https?://\S+", "link", t)
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"\s*[-–—]{2,}\s*", " — ", t)                # '--' or '———' as a dash
    t = re.sub(r"[*_~#]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class Pronunciations:
    """User pronunciation list.

    File format, one entry per line:
        Hermione = her-MY-oh-nee        (respelling, used by every engine)
        Daenerys = /dɪˈnɛəɹɪs/           (IPA, used by Kokoro; others keep the word)
        # comments are allowed
    """

    def __init__(self, entries: dict[str, str] | None = None):
        self.entries = entries or {}
        self._rx = None
        if self.entries:
            words = sorted(self.entries, key=len, reverse=True)
            self._rx = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")

    @classmethod
    def load(cls, path: str | Path | None) -> "Pronunciations":
        if not path or not Path(path).is_file():
            return cls()
        entries = {}
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split(" #", 1)[0].strip()      # trailing comment
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    entries[k.strip()] = v.strip()
        return cls(entries)

    def apply(self, text: str, ipa_style: str | None = None) -> str:
        """ipa_style='misaki' writes [word](/ipa/) for Kokoro; None drops IPA entries."""
        if not self._rx:
            return text

        def sub(m):
            word, rep = m.group(1), self.entries[m.group(1)]
            if rep.startswith("/") and rep.endswith("/"):
                return f"[{word}]({rep})" if ipa_style == "misaki" else word
            return rep
        return self._rx.sub(sub, text)
