from inkvoice.analyze.segment import detect_quote_style, segment_paragraphs, split_paragraph
from inkvoice.model import Paragraph


def kinds(spans):
    return [(k, t) for k, t in spans]


def test_double_quotes():
    spans, unclosed = split_paragraph("“Wait,” he said, “I’m coming.”", "double")
    assert kinds(spans) == [("dialogue", "Wait,"), ("narration", "he said,"), ("dialogue", "I’m coming.")]
    assert not unclosed


def test_straight_quotes():
    spans, _ = split_paragraph('"Run!" she shouted. "Now!"', "straight")
    assert kinds(spans) == [("dialogue", "Run!"), ("narration", "she shouted."), ("dialogue", "Now!")]


def test_single_quotes_with_apostrophes():
    text = "‘I can’t find the dogs’ leads,’ said Ann. ‘Have you seen them?’"
    spans, _ = split_paragraph(text, "single")
    assert kinds(spans) == [("dialogue", "I can’t find the dogs’ leads,"), ("narration", "said Ann."),
                            ("dialogue", "Have you seen them?")]


def test_scare_quotes_stay_narration():
    spans, _ = split_paragraph("He called it his “office” and laughed.", "double")
    assert [k for k, _ in spans] == ["narration"]
    assert "“office”" in spans[0][1]


def test_unclosed_quote_continues():
    paras = [Paragraph("“It was a long winter,” she said. “The longest I remember."),
             Paragraph("“Then spring came all at once.”")]
    units = segment_paragraphs(paras, "double")
    assert units[-1].kind == "dialogue" and units[-1].cont


def test_detect_style():
    assert detect_quote_style(["‘Hello,’ she said.", "‘Hi.’"]) == "single"
    assert detect_quote_style(["“Hello,” she said."]) == "double"
