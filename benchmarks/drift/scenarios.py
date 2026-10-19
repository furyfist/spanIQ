"""Offline trace streams for the drift benchmark.

Baseline traces are built from the labeled RAG and summarization rows in
benchmarks/datasets: the retrieval span returns the item's context (sometimes
with a distractor passage), the generation span returns one of the item's good
answers in one of a few phrasings. Each scenario rewrites traces the way a real
change tends to: a broken retriever returns other items' passages and the model
starts refusing or making things up, a smaller model answers tersely, a new
prompt turns answers into lists.

These streams are synthetic and the perturbations were written by us, so a
result here shows that a detector's mechanics work, not how it does on real
traffic. benchmarks/drift/live.py builds the same scenarios from real LLM calls.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from benchmarks.config import DATASETS_DIR
from spaniq.signals import split_sentences

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

FABRICATIONS = (
    "This was first documented in 1873 by researchers at Uppsala University.",
    "Roughly 42 percent of recorded cases occur in Brazil.",
    "The estimate was revised to 3,400 in a 2019 Harvard study.",
    "Dr. Elena Marsh confirmed this at the Geneva summit.",
    "Maintaining it costs about $12 million per year.",
    "The original finding came from a 1962 NASA report.",
)
_PREFIXES = ("", "", "", "Based on the context, ", "In short, ", "According to the passage, ")


@dataclass(frozen=True)
class Item:
    question: str
    context: str
    answers: tuple[str, ...]


def load_items() -> list[Item]:
    """Distinct (question, context, good answers) items from the labeled datasets."""
    grouped: dict[str, dict] = {}
    for name in ("rag_retrieval", "summarization"):
        with open(DATASETS_DIR / f"{name}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if name == "summarization":
                    question, context = "Summarize the document.", row["input"]
                else:
                    question, context = row["input"], row["context"]
                entry = grouped.setdefault(
                    context, {"question": question, "answers": [row["reference_output"]]}
                )
                if row.get("label") == "good" and row["output"] not in entry["answers"]:
                    entry["answers"].append(row["output"])
    return [
        Item(question=v["question"], context=ctx, answers=tuple(v["answers"]))
        for ctx, v in grouped.items()
    ]


def _phrase(answer: str, rng: random.Random) -> str:
    prefix = rng.choice(_PREFIXES)
    return prefix + (answer[0].lower() + answer[1:] if prefix else answer)


def make_record(items: list[Item], index: int, rng: random.Random) -> dict:
    """One unchanged retrieval -> generation trace in FileCollector JSONL shape."""
    item = rng.choice(items)
    context = item.context
    if rng.random() < 0.3:
        context += "\n\n" + rng.choice([i for i in items if i is not item]).context
    answer = _phrase(rng.choice(item.answers), rng)
    return _record(index, item.question, context, answer)


def _record(index: int, question: str, context: str, answer: str) -> dict:
    return {
        "trace_id": f"t{index:06d}",
        "timestamp": (_EPOCH + timedelta(seconds=5 * index)).isoformat(),
        "input": question,
        "output": answer,
        "metadata": {"context": context},
        "components": [
            {"name": "retrieval", "kind": "retrieval", "output": context},
            {"name": "generation", "kind": "chat", "output": answer},
        ],
    }


def _with(record: dict, context: str | None = None, answer: str | None = None) -> dict:
    question = record["input"]
    context = record["metadata"]["context"] if context is None else context
    answer = record["output"] if answer is None else answer
    index = int(record["trace_id"][1:])
    return _record(index, question, context, answer)


def _topic(question: str) -> str:
    return question.rstrip("?.").removeprefix("Based on the context, ").lower()


def _retriever_degraded(record: dict, items: list[Item], rng: random.Random) -> dict:
    others = [i for i in items if i.context not in record["metadata"]["context"]]
    context = rng.choice(others).context
    if rng.random() < 0.5:
        answer = f"The context does not mention {_topic(record['input'])}."
    else:
        answer = record["output"] + " " + rng.choice(FABRICATIONS)
    return _with(record, context=context, answer=answer)


def _context_truncated(record: dict, items: list[Item], rng: random.Random) -> dict:
    context = record["metadata"]["context"][:80]
    first = split_sentences(record["output"])[0]
    short = " ".join(first.split()[:12]).rstrip(".,;") + "."
    if rng.random() < 0.5:
        short += " The context does not say more."
    return _with(record, context=context, answer=short)


def _prompt_format_change(record: dict, items: list[Item], rng: random.Random) -> dict:
    parts = [p.strip(" .") for p in record["output"].replace(";", ",").split(",") if p.strip()]
    bullets = "\n".join(f"- {p[0].upper() + p[1:]}." for p in parts)
    if rng.random() < 0.5:
        bullets = "## Answer\n" + bullets
    return _with(record, answer=bullets)


def _model_swap(record: dict, items: list[Item], rng: random.Random) -> dict:
    first = record["output"].split(",")[0].split(";")[0]
    answer = " ".join(first.split()[:10]).rstrip(".") + "."
    if rng.random() < 0.4:
        answer = "It may be that " + answer[0].lower() + answer[1:]
    if rng.random() < 0.3:
        answer += " " + rng.choice(FABRICATIONS)
    return _with(record, answer=answer)


def _verbose_drift(record: dict, items: list[Item], rng: random.Random) -> dict:
    answer = record["output"]
    answer = f"{answer} To elaborate, {answer[0].lower() + answer[1:]} In other words, {answer}"
    return _with(record, answer=answer)


_NUMBER = re.compile(r"\d[\d,.]*")
_NAME = re.compile(r"(?<=[a-z,] )[A-Z][a-z]+")
_SWAP_NAMES = ("Lisbon", "Kepler", "Ontario", "Darwin", "Madrid", "Halley")


def _fact_swap(record: dict, items: list[Item], rng: random.Random) -> dict:
    """Same length and shape, wrong facts: numbers and mid-sentence names are replaced."""

    def number(m: re.Match) -> str:
        digits = m.group(0).rstrip(".,")
        tail = m.group(0)[len(digits) :]
        return str(int(digits.replace(",", "").split(".")[0] or 0) + rng.randint(3, 40)) + tail

    answer = _NUMBER.sub(number, record["output"])
    answer = _NAME.sub(lambda m: rng.choice(_SWAP_NAMES), answer)
    return _with(record, answer=answer)


def _wrong_passage(record: dict, items: list[Item], rng: random.Random) -> dict:
    """The retriever returns an unrelated passage but the answer text does not change."""
    others = [i for i in items if i.context not in record["metadata"]["context"]]
    return _with(record, context=rng.choice(others).context)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    root: str | None
    perturb: Callable[[dict, list[Item], random.Random], dict] | None
    fraction: float = 1.0


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("control", "nothing changes; every alarm is a false alarm", None, None),
        Scenario(
            "retriever_degraded",
            "retriever returns other items' passages; the model refuses or invents facts",
            "retrieval",
            _retriever_degraded,
        ),
        Scenario(
            "context_truncated",
            "retrieved context is cut to 80 characters; answers get short and hedged",
            "retrieval",
            _context_truncated,
        ),
        Scenario(
            "prompt_format_change",
            "a new system prompt turns answers into bullet lists",
            "generation",
            _prompt_format_change,
        ),
        Scenario(
            "model_swap",
            "a smaller model answers tersely, hedges, and sometimes invents facts",
            "generation",
            _model_swap,
        ),
        Scenario(
            "verbose_drift",
            "answers become long and repetitive",
            "generation",
            _verbose_drift,
        ),
        Scenario(
            "fact_swap",
            "half the answers keep their shape but get wrong numbers and names",
            "generation",
            _fact_swap,
            fraction=0.5,
        ),
        Scenario(
            "partial_retriever",
            "30% of retrievals return an unrelated passage; answers look unchanged",
            "retrieval",
            _wrong_passage,
            fraction=0.3,
        ),
    )
}


def build_stream(
    scenario: Scenario,
    items: list[Item],
    seed: int,
    n_baseline: int = 100,
    n_pre: int = 200,
    n_post: int = 300,
) -> tuple[list[dict], list[dict], int | None]:
    """(baseline, stream, change_at). change_at is None for the control scenario."""
    rng = random.Random(seed)
    baseline = [make_record(items, i, rng) for i in range(n_baseline)]
    offset = n_baseline
    stream = [make_record(items, offset + i, rng) for i in range(n_pre + n_post)]
    if scenario.perturb is None:
        return baseline, stream, None
    for i in range(n_pre, len(stream)):
        if rng.random() < scenario.fraction:
            stream[i] = scenario.perturb(stream[i], items, rng)
    return baseline, stream, n_pre
