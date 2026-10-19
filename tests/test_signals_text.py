import pytest

from spaniq.signals import TEXT_SIGNALS, split_sentences, text_signals


def test_every_signal_is_present_and_numeric():
    s = text_signals("Paris is the capital of France. It has 2 million people.")
    assert set(s) == set(TEXT_SIGNALS)
    assert all(isinstance(v, float) for v in s.values())


def test_empty_output_is_all_zero():
    assert all(v == 0.0 for v in text_signals("").values())


def test_counts():
    s = text_signals("One two three. Four five?")
    assert s["words"] == 5
    assert s["sentences"] == 2
    assert s["questions"] == 1
    assert s["words_per_sentence"] == pytest.approx(2.5)


def test_list_and_headers():
    s = text_signals("# Answer\n- first\n- second\n1. third")
    assert s["headers"] == 1
    assert s["list_lines"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', 1.0),
        ('```json\n{"a": [1, 2]}\n```', 1.0),
        ("{not json}", 0.0),
        ("plain text", 0.0),
    ],
)
def test_json_valid(text, expected):
    assert text_signals(text)["json_valid"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, I can't help with that.",
        "The context does not mention the launch date.",
        "I don't have enough information to answer.",
    ],
)
def test_refusal_detected(text):
    assert text_signals(text)["refusal"] == 1.0


def test_refusal_not_triggered_by_normal_answer():
    assert text_signals("The launch was in 1969.")["refusal"] == 0.0


def test_repetition_and_diversity():
    looped = text_signals("the cat sat the cat sat the cat sat")
    fresh = text_signals("a quick brown fox jumps over lazy dogs")
    assert looped["repetition"] > 0.5
    assert fresh["repetition"] == 0.0
    assert looped["lexical_diversity"] < fresh["lexical_diversity"]


def test_hedges_are_a_rate():
    assert text_signals("It might possibly rain")["hedges"] == pytest.approx(0.5)


def test_split_sentences():
    assert split_sentences("A b. C d!\nE f") == ["A b.", "C d!", "E f"]
