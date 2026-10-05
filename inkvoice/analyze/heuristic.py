"""Rule-based director: free, offline, decent.

Finds who speaks from dialogue tags ("…," said Mira / Mira whispered), action
beats and turn-taking, and picks emotions from speech verbs, adverbs,
punctuation and a small word list. The Claude director does a much better job
on subtle scenes; this one is the fallback that needs nothing.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..model import NARRATOR, UNKNOWN, Character, Chapter, Unit

# verb -> (emotion, intensity, delivery)
SPEECH_VERBS: dict[str, tuple[str, int, str]] = {
    "said": ("neutral", 1, "normal"), "says": ("neutral", 1, "normal"), "say": ("neutral", 1, "normal"),
    "asked": ("neutral", 1, "normal"), "asks": ("neutral", 1, "normal"), "replied": ("neutral", 1, "normal"),
    "answered": ("neutral", 1, "normal"), "added": ("neutral", 1, "normal"), "continued": ("neutral", 1, "normal"),
    "explained": ("calm", 1, "normal"), "told": ("neutral", 1, "normal"), "went on": ("neutral", 1, "normal"),
    "whispered": ("neutral", 1, "whisper"), "whispers": ("neutral", 1, "whisper"),
    "breathed": ("tender", 1, "soft"), "murmured": ("tender", 1, "soft"), "mumbled": ("hesitant", 1, "soft"),
    "muttered": ("cold", 1, "soft"), "shouted": ("neutral", 2, "shout"), "yelled": ("neutral", 2, "shout"),
    "bellowed": ("angry", 3, "shout"), "roared": ("angry", 3, "shout"), "hollered": ("excited", 2, "shout"),
    "screamed": ("fearful", 3, "shout"), "shrieked": ("fearful", 3, "shout"), "cried": ("excited", 2, "loud"),
    "exclaimed": ("excited", 2, "loud"), "called": ("neutral", 1, "loud"), "sobbed": ("sad", 3, "normal"),
    "wept": ("sad", 3, "soft"), "whimpered": ("fearful", 2, "soft"), "wailed": ("sad", 3, "loud"),
    "laughed": ("amused", 2, "normal"), "chuckled": ("amused", 1, "normal"), "giggled": ("amused", 2, "normal"),
    "snickered": ("sarcastic", 1, "normal"), "teased": ("playful", 2, "normal"), "joked": ("playful", 2, "normal"),
    "grinned": ("happy", 2, "normal"), "smiled": ("happy", 1, "normal"), "beamed": ("happy", 2, "normal"),
    "snapped": ("angry", 2, "normal"), "snarled": ("angry", 3, "normal"), "growled": ("angry", 2, "normal"),
    "hissed": ("angry", 2, "soft"), "spat": ("angry", 3, "normal"), "barked": ("angry", 2, "loud"),
    "fumed": ("angry", 2, "normal"), "seethed": ("angry", 3, "soft"), "thundered": ("angry", 3, "loud"),
    "demanded": ("angry", 1, "normal"), "stammered": ("anxious", 2, "normal"), "stuttered": ("anxious", 2, "normal"),
    "faltered": ("hesitant", 2, "soft"), "pleaded": ("pleading", 2, "normal"), "begged": ("pleading", 3, "normal"),
    "implored": ("pleading", 2, "normal"), "sighed": ("tired", 1, "soft"), "groaned": ("tired", 2, "normal"),
    "moaned": ("sad", 2, "soft"), "whined": ("pleading", 1, "normal"), "gasped": ("surprised", 2, "normal"),
    "admitted": ("hesitant", 1, "normal"), "confessed": ("hesitant", 1, "soft"), "warned": ("anxious", 1, "normal"),
    "insisted": ("determined", 2, "normal"), "declared": ("proud", 1, "loud"), "announced": ("proud", 1, "loud"),
    "sneered": ("sarcastic", 2, "normal"), "scoffed": ("sarcastic", 2, "normal"), "mocked": ("sarcastic", 2, "normal"),
    "purred": ("playful", 2, "soft"), "cooed": ("tender", 2, "soft"), "soothed": ("tender", 2, "soft"),
    "promised": ("tender", 1, "normal"), "ordered": ("determined", 2, "normal"), "commanded": ("determined", 2, "loud"),
    "protested": ("angry", 1, "normal"), "objected": ("angry", 1, "normal"), "argued": ("angry", 1, "normal"),
    "repeated": ("neutral", 1, "normal"), "agreed": ("neutral", 1, "normal"), "suggested": ("calm", 1, "normal"),
    "wondered": ("hesitant", 1, "normal"), "mused": ("calm", 1, "soft"), "managed": ("tired", 2, "soft"),
    "croaked": ("tired", 2, "soft"), "rasped": ("tired", 2, "soft"), "panted": ("tired", 2, "normal"),
    "blurted": ("surprised", 1, "normal"), "retorted": ("sarcastic", 1, "normal"), "sang": ("happy", 1, "normal"),
}
ADVERBS: dict[str, tuple[str, int, str | None]] = {
    "angrily": ("angry", 2, None), "furiously": ("angry", 3, "loud"), "sharply": ("angry", 1, None),
    "impatiently": ("angry", 1, None), "irritably": ("angry", 1, None), "sadly": ("sad", 2, None),
    "tearfully": ("sad", 3, None), "miserably": ("sad", 2, None), "softly": ("tender", 1, "soft"),
    "quietly": ("calm", 1, "soft"), "gently": ("tender", 2, "soft"), "warmly": ("tender", 2, None),
    "nervously": ("anxious", 2, None), "anxiously": ("anxious", 2, None), "fearfully": ("fearful", 2, None),
    "desperately": ("pleading", 3, None), "urgently": ("anxious", 2, None), "excitedly": ("excited", 2, None),
    "eagerly": ("excited", 1, None), "happily": ("happy", 2, None), "cheerfully": ("happy", 2, None),
    "brightly": ("happy", 1, None), "coldly": ("cold", 2, None), "icily": ("cold", 3, None),
    "flatly": ("cold", 1, None), "bitterly": ("cold", 2, None), "sarcastically": ("sarcastic", 2, None),
    "dryly": ("sarcastic", 1, None), "drily": ("sarcastic", 1, None), "mockingly": ("sarcastic", 2, None),
    "playfully": ("playful", 2, None), "calmly": ("calm", 1, None), "firmly": ("determined", 2, None),
    "wearily": ("tired", 2, None), "tiredly": ("tired", 2, None), "weakly": ("tired", 2, "soft"),
    "hesitantly": ("hesitant", 2, None), "shyly": ("hesitant", 1, "soft"), "timidly": ("hesitant", 2, "soft"),
    "proudly": ("proud", 2, None), "grimly": ("solemn", 2, None), "solemnly": ("solemn", 2, None),
    "seriously": ("solemn", 1, None), "loudly": ("neutral", 1, "loud"), "breathlessly": ("excited", 2, None),
    "incredulously": ("surprised", 2, None), "suspiciously": ("cold", 1, None), "kindly": ("tender", 1, None),
}
CONTEXT_CUES: list[tuple[str, str]] = [
    (r"\b(tears?|sobb\w*|weep\w*|cried|crying|grief|mourn\w*|heartbroken|choked)\b", "sad"),
    (r"\b(trembl\w*|shak(ing|y)|terrif\w*|horror|panic\w*|afraid|scared|frightened|dread)\b", "fearful"),
    (r"\b(grinn\w*|laugh\w*|beam\w*|delight\w*|chuckl\w*|giggl\w*)\b", "amused"),
    (r"\b(glar\w*|clench\w*|furious|rage|fury|seeth\w*|scowl\w*|slamm\w*)\b", "angry"),
    (r"\b(nervous\w*|anxious\w*|fidget\w*|swallow\w*|uneas\w*|worried)\b", "anxious"),
    (r"\b(smil\w*|happ(y|ily)|joy\w*|relie(f|ved))\b", "happy"),
    (r"\b(stunned|shocked|astonish\w*|startled|wide-eyed|gaped)\b", "surprised"),
    (r"\b(exhaust\w*|weary|wearily|tired|yawn\w*|drained)\b", "tired"),
]
NARRATION_MOOD: list[tuple[str, str]] = [
    (r"\b(shadow\w*|dark(ness)?|silen(t|ce)|creak\w*|footsteps|blood|crept|whisper\w*|cold|"
     r"heart (pound|hammer|race)\w*|something moved)\b", "suspense"),
    (r"\b(funeral|grave|grief|tears|mourn\w*|empty|alone|loss|died|death)\b", "solemn"),
    (r"\b(laugh\w*|sunlight|joy\w*|danc\w*|cheer\w*|celebrat\w*)\b", "happy"),
]
TITLES = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Doctor|Professor|Prof|Lord|Lady|Sir|Dame|Captain|Capt|Aunt|Uncle|Father|Mother|King|Queen|Prince|Princess|Master|Madam|Madame|Sister|Brother|General|Colonel|Sergeant|Detective|Inspector|Officer)"
MALE_TITLES = {"mr", "lord", "sir", "uncle", "father", "king", "prince", "master", "brother"}
FEMALE_TITLES = {"mrs", "ms", "miss", "lady", "dame", "aunt", "mother", "queen", "princess", "madam", "madame", "sister"}
STOP = set("""The A An He She It They We You I But And Then When What Why How Where Who Whose This That These
Those There Here Yes No Oh Ah Well So Now Still Just Even After Before If As At In On For With His Her Their Our My
Your Its One Someone Everyone Nobody Somebody Something Nothing Everything Anyone Chapter Mr Mrs Ms Miss Dr Sir Lady
Lord Okay OK Hey Hi Hello Please Thank Thanks Sorry Maybe Perhaps Not Don't Can't Won't Let Let's God Good Great
Come Go Look Listen Wait Stop Run Help Who's What's That's It's Still Yet Once Twice Again Also Some Any All Every
Each Both Either Neither Because Since Until While Although Though Or Nor Of To From By Into Over Under About Through
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September
October November December Christmas English French""".split())

_VERBS_RX = "|".join(sorted((re.escape(v) for v in SPEECH_VERBS), key=len, reverse=True))
NAME = rf"(?:{TITLES}\.?\s+)?[A-Z][a-zA-Z’'\-]+(?:\s+[A-Z][a-zA-Z’'\-]+)?"
AFTER_TAG = re.compile(
    rf"^[\s,;:—–-]*(?:(?P<verb>{_VERBS_RX})\s+(?P<name>{NAME})"
    rf"|(?P<name2>{NAME}|he|she|they|I|we)\s+(?:(?P<adv0>\w+ly)\s+)?(?P<verb2>{_VERBS_RX}))\b"
    rf"(?:,?\s+(?P<adv>\w+ly)\b)?")
BEFORE_TAG = re.compile(
    rf"(?P<name>{NAME}|he|she|they|I)\s+(?:(?P<adv0>\w+ly)\s+)?(?P<verb>{_VERBS_RX})\b[^.!?]*[,:]\s*$")
SUBJECT = re.compile(rf"^(?P<name>{NAME})\s+[a-z]")
PRONOUN_GENDER = {"he": "male", "she": "female"}


def _clean_name(n: str | None) -> str | None:
    if not n:
        return None
    n = n.strip()
    if n.lower() in ("he", "she", "they", "i", "we"):
        return n.lower()
    first = n.split()[0].rstrip(".")
    if first in STOP and not re.match(TITLES, first):
        return None
    if n in STOP:
        return None
    return n


class HeuristicDirector:
    def __init__(self, chapters: list[Chapter]):
        self.chapters = chapters
        self.proper = self._proper_nouns()
        self.characters: dict[str, Character] = {}

    # ----------------------------------------------------------- setup --
    def _proper_nouns(self) -> Counter:
        """Capitalized words that (almost) never appear in lower case - likely names."""
        cap, low = Counter(), Counter()
        for ch in self.chapters:
            for u in ch.units:
                if u.kind not in ("narration", "dialogue"):
                    continue
                for w in re.findall(r"[A-Za-z][a-zA-Z’'\-]+", u.text):
                    if w[0].isupper():
                        cap[w] += 1
                    else:
                        low[w.capitalize()] += 1
        return Counter({w: n for w, n in cap.items()
                        if w not in STOP and not w.isupper() and low[w] <= 0.1 * n})

    def _norm_name(self, n: str | None) -> str | None:
        """'Old Tobias' -> 'Tobias'; keeps titles ('Mr. Smith') and pronouns."""
        n = _clean_name(n)
        if not n or n in ("he", "she", "they", "i", "we"):
            return n
        parts = n.split()
        while len(parts) > 1 and not re.fullmatch(TITLES + r"\.?", parts[0]) and self.proper[parts[0]] < 2:
            parts.pop(0)
        return " ".join(parts)

    def _is_name(self, n: str, tagged: bool = False) -> bool:
        """tagged=True: found right next to a speech verb ("Bram said"), so one sighting is enough."""
        if n in ("he", "she", "they", "i", "we"):
            return True
        parts = n.replace(".", "").split()
        if re.fullmatch(TITLES, parts[0]):
            return True
        need = 1 if tagged else 2
        return any(self.proper[p] >= need for p in parts)

    # ------------------------------------------------------------- run --
    def run(self) -> dict[str, Character]:
        # Pass 1 finds the names and their genders; pass 2 uses the genders to
        # resolve "he said" / "she said".
        self._gender_cache: dict[str, str] = {}
        self._alias: dict[str, str] = {}
        self._pronoun_votes: dict[str, Counter] = defaultdict(Counter)
        for pass_no in (1, 2):
            self._pass = pass_no
            self._pronoun_votes.clear()
            raw_counts = Counter()
            for ch in self.chapters:
                for u in ch.units:
                    if u.kind == "dialogue":
                        u.speaker = NARRATOR
                self._attribute_chapter(ch, raw_counts)
            self.characters = {}
            self._canonicalize(raw_counts)
            self._guess_genders()
            if pass_no == 1:
                for name, c in self.characters.items():
                    if c.gender != "unknown":
                        for key in [name] + c.aliases:
                            self._gender_cache[key] = c.gender
        for ch in self.chapters:
            self._emotions(ch)
        return self.characters

    def _attribute_chapter(self, ch: Chapter, counts: Counter) -> None:
        by_para: dict[int, list[Unit]] = defaultdict(list)
        for u in ch.units:
            by_para[u.para].append(u)
        recent: list[str] = []          # speakers in the current exchange, latest last
        quiet = 0
        last_names: list[tuple[str, str | None]] = []   # recently mentioned names with gender guesses
        prev_para_speaker = None
        for pi in sorted(by_para):
            units = by_para[pi]
            if units[0].kind in ("heading", "break"):
                recent, quiet, prev_para_speaker = [], 0, None
                continue
            dialogue = [u for u in units if u.kind == "dialogue"]
            for u in units:
                if u.kind == "narration":
                    for m in re.finditer(NAME, u.text):
                        nm = self._norm_name(m.group(0))
                        if nm and nm not in ("he", "she", "they", "i", "we") and self._is_name(nm):
                            last_names.append((nm, None))
                    last_names = last_names[-12:]
            if not dialogue:
                quiet += 1
                if quiet >= 2:
                    recent = []
                prev_para_speaker = None
                continue
            quiet = 0

            para_speaker = None
            for idx, u in enumerate(units):
                if u.kind != "dialogue":
                    continue
                spk, verb, adv = None, None, None
                nxt = units[idx + 1] if idx + 1 < len(units) else None
                prv = units[idx - 1] if idx > 0 else None
                if nxt is not None and nxt.kind == "narration":
                    m = AFTER_TAG.match(nxt.text)
                    if m:
                        spk = self._norm_name(m.group("name") or m.group("name2"))
                        verb = m.group("verb") or m.group("verb2")
                        adv = m.group("adv") or m.group("adv0")
                if spk is None and prv is not None and prv.kind == "narration":
                    m = BEFORE_TAG.search(prv.text)
                    if m:
                        spk = self._norm_name(m.group("name"))
                        verb, adv = m.group("verb"), m.group("adv0")
                    elif not verb:
                        # "Then, more gently, “…”" - a manner adverb right before the quote
                        tail = re.split(r"[.!?]\s", prv.text)[-1]
                        hit = [w for w in re.findall(r"\b\w+ly\b", tail.lower()) if w in ADVERBS]
                        if hit:
                            adv = hit[-1]
                if not verb and not adv and idx >= 2 and units[idx - 2].kind == "dialogue":
                    # second quote in the same paragraph keeps the first one's tag
                    verb = getattr(units[idx - 2], "_verb_keep", None)
                    adv = getattr(units[idx - 2], "_adv_keep", None)
                if verb:
                    u._verb_keep = verb.lower()  # type: ignore[attr-defined]
                if adv:
                    u._adv_keep = adv.lower()  # type: ignore[attr-defined]
                if verb:
                    u._verb = verb.lower()  # type: ignore[attr-defined]
                if adv:
                    u._adv = adv.lower()  # type: ignore[attr-defined]
                if spk and not self._is_name(spk, tagged=True):
                    spk = None
                if spk:
                    u.speaker = spk
                    para_speaker = para_speaker or spk

            if para_speaker is None:
                # action beat: "Mira slammed the door. “Enough!”"
                for u in units:
                    if u.kind == "narration":
                        m = SUBJECT.match(u.text)
                        if m:
                            nm = self._norm_name(m.group("name"))
                            if nm and self._is_name(nm) and nm not in ("he", "she", "they", "i", "we"):
                                para_speaker = nm
                        break
            if para_speaker is None and dialogue[0].cont and prev_para_speaker:
                para_speaker = prev_para_speaker
            if para_speaker is None and len(recent) >= 2:
                para_speaker = recent[-2]            # turn-taking: A, B, A, B...
            for u in dialogue:
                if u.speaker == NARRATOR:
                    u.speaker = para_speaker or UNKNOWN
            # resolve pronouns against recent names
            for u in dialogue:
                if u.speaker in ("he", "she"):
                    want = PRONOUN_GENDER[u.speaker]
                    pool = list(dict.fromkeys(list(reversed(recent)) + [nm for nm, _ in reversed(last_names)]))
                    cands = [s for s in pool if self._gender_hint(s) == want]
                    if not cands and self._pass == 2:   # nobody known to match: someone whose gender is open
                        cands = [s for s in pool if self._gender_hint(s) is None]
                    if cands:
                        self._pronoun_votes[cands[0]][want] += 1
                        u.speaker = cands[0]
                if u.speaker in ("i", "we"):
                    u.speaker = NARRATOR
                if u.speaker == "they":
                    u.speaker = UNKNOWN
                if u.speaker not in (NARRATOR, UNKNOWN, "he", "she"):
                    counts[u.speaker] += 1
            spk_now = dialogue[-1].speaker
            if spk_now not in (UNKNOWN, NARRATOR, "he", "she"):
                if not recent or recent[-1] != spk_now:
                    recent.append(spk_now)
                    recent = recent[-4:]
            prev_para_speaker = spk_now

    def _gender_hint(self, name: str) -> str | None:
        first = name.split()[0].rstrip(".").lower()
        if first in MALE_TITLES:
            return "male"
        if first in FEMALE_TITLES:
            return "female"
        c = self.characters.get(name)
        if c and c.gender != "unknown":
            return c.gender
        return getattr(self, "_gender_cache", {}).get(name)

    def _canonicalize(self, counts: Counter) -> None:
        names = [n for n, _ in counts.most_common()]
        alias_of: dict[str, str] = {}
        for n in names:
            # "Smith" -> "Mr. Smith" / "Mira" -> "Mira Vance" when unambiguous
            longer = [m for m in names if m != n and n in m.replace(".", "").split() + [m]]
            longer = [m for m in longer if re.search(rf"\b{re.escape(n)}\b", m)]
            if len(longer) == 1 and counts[longer[0]] >= 1:
                alias_of[n] = longer[0]
        for n in names:
            canon = alias_of.get(n, n)
            ch = self.characters.setdefault(canon, Character(name=canon))
            if canon != n and n not in ch.aliases:
                ch.aliases.append(n)
        for ch in self.chapters:
            for u in ch.units:
                if u.kind == "dialogue" and u.speaker in alias_of:
                    u.speaker = alias_of[u.speaker]
                if u.kind == "dialogue" and u.speaker in ("he", "she"):
                    u.speaker = UNKNOWN

    def _guess_genders(self) -> None:
        votes: dict[str, Counter] = defaultdict(Counter)
        texts = [u.text for ch in self.chapters for u in ch.units if u.kind == "narration"]
        for name, c in self.characters.items():
            first = name.split()[0].rstrip(".").lower()
            if first in MALE_TITLES:
                votes[name]["male"] += 10
            elif first in FEMALE_TITLES:
                votes[name]["female"] += 10
            keys = [name] + c.aliases
            for k in keys:                 # "she said" resolved to this person during attribution
                for g, n in self._pronoun_votes.get(k, {}).items():
                    votes[name][g] += 2 * n
            rx = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b(.{0,140})")
            for t in texts:
                for m in rx.finditer(t):
                    p = re.search(r"\b(he|him|his|she|her|hers)\b", m.group(2), re.I)
                    if p:
                        votes[name]["male" if p.group(1).lower() in ("he", "him", "his") else "female"] += 1
        for name, c in self.characters.items():
            v = votes[name]
            total = v["male"] + v["female"]
            if total >= 1 and max(v.values()) >= 0.66 * total:
                c.gender = v.most_common(1)[0][0]

    # -------------------------------------------------------- emotions --
    def _emotions(self, ch: Chapter) -> None:
        by_para: dict[int, list[Unit]] = defaultdict(list)
        for u in ch.units:
            by_para[u.para].append(u)
        for units in by_para.values():
            narr = " ".join(u.text for u in units if u.kind == "narration")
            for u in units:
                if u.kind == "dialogue":
                    self._dialogue_emotion(u, narr)
                elif u.kind == "narration" and len(units) == 1:
                    self._narration_mood(u)

    @staticmethod
    def _dialogue_emotion(u: Unit, context: str) -> None:
        emo, inten, deliv = "neutral", 1, "normal"
        verb = getattr(u, "_verb", None)
        adv = getattr(u, "_adv", None)
        if verb and verb in SPEECH_VERBS:
            emo, inten, deliv = SPEECH_VERBS[verb]
        if adv and adv in ADVERBS:
            e2, i2, d2 = ADVERBS[adv]
            if e2 != "neutral":
                emo, inten = e2, max(inten, i2)
            deliv = d2 or deliv
        t = u.text
        if emo == "neutral":
            for rx, e in CONTEXT_CUES:
                if re.search(rx, context, re.I):
                    emo, inten = e, 1
                    break
        bangs = t.count("!")
        if "?!" in t or "!?" in t:
            emo = "surprised" if emo == "neutral" else emo
            inten = max(inten, 2)
        elif bangs:
            if emo == "neutral":
                emo = "excited"
            inten = min(3, max(inten, 1 + bangs))
            if deliv == "normal" and bangs >= 2:
                deliv = "loud"
        if emo == "neutral" and re.search(r"(\.\.\.|…|—|–)\s*$", t):
            emo = "hesitant"
        caps = [w for w in re.findall(r"\b[A-Z]{2,}\b", t) if w not in ("OK", "TV", "I")]
        if len(caps) >= 2 or (caps and len(t.split()) <= 3):
            deliv = "shout"
            inten = max(inten, 2)
        u.emotion, u.intensity, u.delivery = emo, inten, deliv
        for attr in ("_verb", "_adv", "_verb_keep", "_adv_keep"):
            if hasattr(u, attr):
                delattr(u, attr)

    @staticmethod
    def _narration_mood(u: Unit) -> None:
        for rx, mood in NARRATION_MOOD:
            if len(re.findall(rx, u.text, re.I)) >= 3:
                u.emotion, u.intensity = mood, 1
                return
