"""Reference-free signals: numbers that describe one output on its own.

None of these say whether an output is good. They are cheap enough to compute
on every trace, and a change in their distribution between a baseline run and
a candidate run is what the drift detectors look for.
"""

from __future__ import annotations

import json
import re

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADER = re.compile(r"^\s*#{1,6}\s")
_REFUSAL = re.compile(
    r"\b(?:i(?:'m| am) (?:sorry|unable|not able)"
    r"|i can(?:'t|not) (?:help|assist|provide|answer|find)"
    r"|as an ai\b"
    r"|i do(?:n't| not) (?:know|have (?:enough |any )?information)"
    r"|cannot be determined"
    r"|(?:does not|doesn't) (?:mention|say|contain|provide)"
    r"|not (?:mentioned|provided|stated) in the (?:context|text|passage|document))",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:might|may|possibly|perhaps|probably|likely|unclear|uncertain|seems?|appears?)\b",
    re.IGNORECASE,
)

TEXT_SIGNALS = (
    "chars",
    "words",
    "sentences",
    "words_per_sentence",
    "lines",
    "list_lines",
    "headers",
    "code_blocks",
    "json_valid",
    "refusal",
    "hedges",
    "questions",
    "digits",
    "uppercase",
    "non_ascii",
    "lexical_diversity",
    "repetition",
)


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _is_json(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def _repetition(tokens: list[str]) -> float:
    """Share of word trigrams that already appeared earlier in the output."""
    grams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def text_signals(text: str) -> dict[str, float]:
    """All reference-free signals of one output, keyed by TEXT_SIGNALS names."""
    text = text or ""
    tokens = [w.lower() for w in words(text)]
    n_words = len(tokens)
    sentences = split_sentences(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    letters = [c for c in text if c.isalpha()]
    n_chars = len(text)
    return {
        "chars": float(n_chars),
        "words": float(n_words),
        "sentences": float(len(sentences)),
        "words_per_sentence": n_words / len(sentences) if sentences else 0.0,
        "lines": float(len(lines)),
        "list_lines": sum(1 for ln in lines if _BULLET.match(ln)) / len(lines) if lines else 0.0,
        "headers": float(sum(1 for ln in lines if _HEADER.match(ln))),
        "code_blocks": float(text.count("```") // 2),
        "json_valid": 1.0 if _is_json(text) else 0.0,
        "refusal": 1.0 if _REFUSAL.search(text) else 0.0,
        "hedges": len(_HEDGE.findall(text)) / n_words if n_words else 0.0,
        "questions": float(text.count("?")),
        "digits": sum(c.isdigit() for c in text) / n_chars if n_chars else 0.0,
        "uppercase": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "non_ascii": sum(ord(c) > 127 for c in text) / n_chars if n_chars else 0.0,
        "lexical_diversity": len(set(tokens)) / n_words if n_words else 0.0,
        "repetition": _repetition(tokens),
    }
