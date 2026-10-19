"""Signals that check an output against the context it was given.

The old similarity metrics compared an answer with a reference, so an answer
that copied the reference and added an invented claim looked better than a
faithful paraphrase. These signals instead ask whether what the answer states
can be found in its own retrieved context, which every RAG trace already has.
They are lexical, so they catch invented numbers, names and vocabulary, not a
reworded false claim; spaniq.signals.nli covers that with a local NLI model.
"""

from __future__ import annotations

import re

from spaniq.signals.text import split_sentences, words

GROUNDING_SIGNALS = ("unsupported_numbers", "unsupported_names", "novel_words")
RELEVANCE_SIGNALS = ("question_coverage",)

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_STOPWORD_TEXT = """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers him his how i if in into is it its
    itself just me more most my no nor not now of off on once only or other our ours out over
    own same she should so some such than that the their theirs them then there these they
    this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours based according context passage text document
    answer question short summary summarize yes"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _stem(word: str) -> str:
    w = word.lower().removesuffix("'s")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}


def _names(text: str) -> set[str]:
    """Capitalised words that do not start a sentence, as a cheap stand-in for named entities."""
    found = set()
    for sentence in split_sentences(text):
        for token in words(sentence)[1:]:
            if token[0].isupper() and token.lower() not in STOPWORDS:
                found.add(token.lower())
    return found


def _content(text: str) -> list[str]:
    return [_stem(w) for w in words(text) if len(w) > 2 and w.lower() not in STOPWORDS]


def _unsupported(claimed: set[str], available: set[str]) -> float:
    return len(claimed - available) / len(claimed) if claimed else 0.0


def grounding_signals(output: str, context: str) -> dict[str, float]:
    """Share of the output's numbers, names and content words that the context never mentions."""
    context_words = {w.lower() for w in words(context)} | {_stem(w) for w in words(context)}
    content = _content(output)
    return {
        "unsupported_numbers": _unsupported(_numbers(output), _numbers(context)),
        "unsupported_names": _unsupported(_names(output), context_words),
        "novel_words": (
            sum(w not in context_words for w in content) / len(content) if content else 0.0
        ),
    }


def relevance_signals(question: str, retrieved: str) -> dict[str, float]:
    """Share of the question's content words that the retrieved text contains."""
    asked = set(_content(question))
    found = set(_content(retrieved))
    return {"question_coverage": len(asked & found) / len(asked) if asked else 1.0}
