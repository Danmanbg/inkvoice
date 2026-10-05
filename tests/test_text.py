from inkvoice.textutil import (Pronunciations, chunk_text, roman_to_int, speakable_heading, split_sentences,
                               tts_clean)


def test_roman():
    assert roman_to_int("IV") == 4
    assert roman_to_int("xii") == 12
    assert roman_to_int("MCMXC") == 1990
    assert roman_to_int("hello") is None
    assert roman_to_int("") is None


def test_speakable_heading():
    assert speakable_heading("CHAPTER IV") == "Chapter Four."
    assert speakable_heading("Chapter 12\nThe Long Night") == "Chapter Twelve. The Long Night."
    assert speakable_heading("7") == "Seven."
    assert speakable_heading("Prologue") == "Prologue."


def test_split_sentences_respects_abbreviations():
    s = split_sentences("Mr. Hale arrived at noon. He was late! Was he? Dr. J. R. Moss said so.")
    assert s == ["Mr. Hale arrived at noon.", "He was late!", "Was he?", "Dr. J. R. Moss said so."]


def test_chunk_text_limits():
    text = " ".join(f"This is sentence number {i}, and it goes on a while." for i in range(40))
    chunks = chunk_text(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_tts_clean():
    assert tts_clean("word[12] and more -- then") == "word and more — then"
    assert tts_clean("see https://example.com now") == "see link now"


def test_pronunciations(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# names\nAoife = EE-fa\nNiamh = /niːv/\n", encoding="utf-8")
    p = Pronunciations.load(f)
    assert p.apply("Aoife met Niamh.") == "EE-fa met Niamh."
    assert p.apply("Aoife met Niamh.", "misaki") == "EE-fa met [Niamh](/niːv/)."
