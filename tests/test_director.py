from inkvoice.analyze import build_script
from inkvoice.analyze.claude_director import ClaudeDirector
from inkvoice.model import NARRATOR, Book, Paragraph, RawChapter, Script

STORY = [
    "Ada pushed the door open. The workshop smelled of oil and cold metal.",
    "“Anyone here?” Ada called.",
    "“Over here,” Bram said from behind the lathe. He wiped his hands on a rag.",
    "“You’re late,” he added.",
    "“The bridge was out,” she replied angrily. “I had to go round by the mill.”",
    "“By the mill? In the dark?”",
    "“Yes.”",
]


def make_book():
    paras = [Paragraph("Chapter One", "heading")] + [Paragraph(t) for t in STORY]
    return Book(title="Test", author="Someone", source="x.txt", chapters=[RawChapter("Chapter One", paras)])


def test_heuristic_speakers_and_emotion():
    script = build_script(make_book())
    d = [u for u in script.chapters[0].units if u.kind == "dialogue"]
    assert d[0].speaker == "Ada"                     # "… ” Ada called.
    assert d[1].speaker == "Bram"                    # "… ” Bram said
    assert d[2].speaker == "Bram"                    # "he added" -> male -> Bram
    assert d[3].speaker == "Ada" and d[3].emotion == "angry"   # "she replied angrily"
    assert script.characters["Bram"].gender == "male"
    assert script.characters["Ada"].gender == "female"
    assert script.chapters[0].units[0].kind == "heading"       # title announcement


def test_script_roundtrip(tmp_path):
    script = build_script(make_book())
    script.narrator_voices["kokoro"] = "af_heart"
    script.characters["Ada"].voices["kokoro"] = "bf_emma"
    f = tmp_path / "s.json"
    script.save(f)
    again = Script.load(f)
    assert again.narrator_voices == {"kokoro": "af_heart"}
    assert again.characters["Ada"].voices == {"kokoro": "bf_emma"}
    assert [u.text for u in again.chapters[0].units] == [u.text for u in script.chapters[0].units]
    assert [u.emotion for u in again.chapters[0].units] == [u.emotion for u in script.chapters[0].units]


class FakeDirector(ClaudeDirector):
    """Exercise the label handling without calling the API."""

    def __init__(self):  # noqa: D401 - no client needed
        self.log = lambda *_: None
        self.first_person = ""
        self._alias = {"ada": "Ada", "bram": "Bram", "mr. bram": "Bram"}
        import threading
        self._lock = threading.Lock()


def test_claude_labels_are_applied():
    script = build_script(make_book())
    units = script.chapters[0].units
    ids = [i for i, u in enumerate(units) if u.kind == "dialogue"]
    narr = next(i for i, u in enumerate(units) if u.kind == "narration")
    labels = [{"i": ids[0], "speaker": "Ada", "emotion": "anxious", "intensity": 1, "delivery": "loud", "note": ""},
              {"i": ids[1], "speaker": "Mr. Bram", "emotion": "calm", "intensity": 1, "delivery": "normal", "note": ""},
              {"i": ids[2], "speaker": "Night Guard", "emotion": "sarcastic", "intensity": 2, "delivery": "normal",
               "note": "deadpan"},
              {"i": 9999, "speaker": "Nobody", "emotion": "sad", "intensity": 3, "delivery": "normal", "note": ""},
              {"i": narr, "speaker": "Ada", "emotion": "suspense", "intensity": 1, "delivery": "normal", "note": ""}]
    FakeDirector()._apply(script, units, set(ids + [narr]), {"labels": labels, "new_characters": []})
    assert units[ids[0]].speaker == "Ada" and units[ids[0]].delivery == "loud"
    assert units[ids[1]].speaker == "Bram"                      # alias resolved
    assert units[ids[2]].speaker == "Night Guard" and units[ids[2]].note == "deadpan"
    assert "Night Guard" in script.characters                   # new speaker registered
    assert units[narr].speaker == NARRATOR and units[narr].emotion == "suspense"
