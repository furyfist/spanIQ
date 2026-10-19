"""Sentence-level groundedness with a local NLI cross-encoder.

Each output sentence is checked against chunks of the context: is it entailed,
contradicted, or neither. This catches reworded false claims that the lexical
grounding signals miss. The model runs on CPU through sentence-transformers,
which spanIQ already depends on, and is downloaded on first use; set
SPANIQ_NLI_MODEL to a local path to run offline.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Protocol

from spaniq.signals.text import split_sentences

DEFAULT_NLI_MODEL = os.environ.get("SPANIQ_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
NLI_SIGNALS = ("entailment", "unsupported_sentences", "contradicted_sentences")
_LABELS = ("contradiction", "entailment", "neutral")


class NLIModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Probabilities keyed contradiction/entailment/neutral for each (premise, hypothesis)."""
        ...


class CrossEncoderNLI:
    """NLIModel backed by a sentence-transformers CrossEncoder, loaded on first predict()."""

    def __init__(self, model_name: str | None = None, batch_size: int = 16) -> None:
        self.model_name = model_name or DEFAULT_NLI_MODEL
        self.batch_size = batch_size
        self._model = None
        self._labels: list[str] | None = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.model_name)
                config = getattr(model, "config", None) or model.model.config
                id2label = getattr(config, "id2label", None) or {}
                labels = [str(id2label.get(i, "")).lower() for i in range(len(_LABELS))]
                if sorted(labels) != sorted(_LABELS):
                    # nli-deberta cross-encoders ship with this order
                    labels = list(_LABELS)
                self._labels = labels
                self._model = model
        return self._model

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        model = self._load()
        probs = model.predict(pairs, apply_softmax=True, batch_size=self.batch_size)
        return [
            {label: float(p) for label, p in zip(self._labels, row, strict=True)} for row in probs
        ]


_shared: dict[str, CrossEncoderNLI] = {}
_shared_lock = threading.Lock()


def get_nli(model_name: str | None = None) -> CrossEncoderNLI:
    """One shared CrossEncoderNLI per model name."""
    name = model_name or DEFAULT_NLI_MODEL
    with _shared_lock:
        if name not in _shared:
            _shared[name] = CrossEncoderNLI(name)
        return _shared[name]


@dataclass
class SentenceVerdict:
    sentence: str
    entailment: float
    contradiction: float


@dataclass
class Groundedness:
    score: float
    unsupported: float
    contradicted: float
    sentences: list[SentenceVerdict]


def chunk_context(context: str, max_chars: int = 1200) -> list[str]:
    """Group context sentences into chunks short enough for a cross-encoder."""
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(context):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def check_groundedness(
    output: str,
    context: str,
    model: NLIModel,
    threshold: float = 0.5,
    max_chars: int = 1200,
) -> Groundedness:
    """Entailment of each output sentence by its best supporting context chunk.

    A sentence's contradiction comes from the chunk that is least neutral about
    it, so an unrelated chunk cannot make a supported sentence look contradicted.
    """
    sentences = split_sentences(output)
    chunks = chunk_context(context, max_chars) or [context]
    if not sentences:
        return Groundedness(score=1.0, unsupported=0.0, contradicted=0.0, sentences=[])
    probs = model.predict([(chunk, s) for s in sentences for chunk in chunks])
    verdicts = []
    for i, sentence in enumerate(sentences):
        rows = probs[i * len(chunks) : (i + 1) * len(chunks)]
        entail = max(r["entailment"] for r in rows)
        decisive = max(rows, key=lambda r: r["entailment"] + r["contradiction"])
        verdicts.append(SentenceVerdict(sentence, entail, decisive["contradiction"]))
    n = len(verdicts)
    return Groundedness(
        score=sum(v.entailment for v in verdicts) / n,
        unsupported=sum(v.entailment < threshold for v in verdicts) / n,
        contradicted=sum(v.contradiction >= threshold for v in verdicts) / n,
        sentences=verdicts,
    )


def nli_signals(output: str, context: str, model: NLIModel) -> dict[str, float]:
    g = check_groundedness(output, context, model)
    return {
        "entailment": g.score,
        "unsupported_sentences": g.unsupported,
        "contradicted_sentences": g.contradicted,
    }
