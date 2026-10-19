"""Loaders that turn labeled hallucination datasets into (context, output, hallucinated) examples.

RAGTruth and HaluEval are read from local files (clone their repos first);
LLM-AggreFact can be read from a local export or fetched from the Hugging Face
hub with the datasets package. Nothing is downloaded unless a loader is called
without a local path.

    RAGTruth:  https://github.com/ParticleMedia/RAGTruth (dataset/response.jsonl, source_info.jsonl)
    HaluEval:  https://github.com/RUCAIBox/HaluEval (data/qa_data.json, summarization_data.json, dialogue_data.json)
    AggreFact: https://huggingface.co/datasets/lytang/LLM-AggreFact
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
from dataclasses import dataclass

from benchmarks.config import DATASETS_DIR


@dataclass(frozen=True)
class GroundingExample:
    source: str
    context: str
    output: str
    hallucinated: bool


def _jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_local() -> list[GroundingExample]:
    """The labeled RAG and summarization rows shipped in benchmarks/datasets."""
    examples = []
    for name in ("rag_retrieval", "summarization"):
        for row in _jsonl(DATASETS_DIR / f"{name}.jsonl"):
            examples.append(
                GroundingExample(
                    source=f"local/{name}",
                    context=row.get("context") or row["input"],
                    output=row["output"],
                    hallucinated=row["label"] == "bad",
                )
            )
    return examples


def _ragtruth_context(info) -> str:
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and "passages" in info:
        passages = info["passages"]
        return passages if isinstance(passages, str) else "\n\n".join(map(str, passages))
    return json.dumps(info, ensure_ascii=False)


def load_ragtruth(directory: str | pathlib.Path, split: str = "test") -> list[GroundingExample]:
    """RAGTruth responses; a response is hallucinated when it has any annotated span."""
    directory = pathlib.Path(directory)
    sources = {r["source_id"]: r for r in _jsonl(directory / "source_info.jsonl")}
    examples = []
    for r in _jsonl(directory / "response.jsonl"):
        if split and r.get("split") != split:
            continue
        source = sources.get(r["source_id"])
        if source is None:
            continue
        examples.append(
            GroundingExample(
                source=f"ragtruth/{source.get('task_type', 'unknown')}",
                context=_ragtruth_context(source["source_info"]),
                output=r["response"],
                hallucinated=bool(r.get("labels")),
            )
        )
    return examples


_HALUEVAL_FIELDS = {
    "qa": ("knowledge", "right_answer", "hallucinated_answer"),
    "summarization": ("document", "right_summary", "hallucinated_summary"),
    "dialogue": ("knowledge", "right_response", "hallucinated_response"),
}


def load_halueval(path: str | pathlib.Path, task: str = "qa") -> list[GroundingExample]:
    """Each HaluEval row gives one faithful and one hallucinated example over the same context."""
    if task not in _HALUEVAL_FIELDS:
        raise ValueError(f"unknown HaluEval task '{task}' (valid: {', '.join(_HALUEVAL_FIELDS)})")
    ctx_key, right_key, wrong_key = _HALUEVAL_FIELDS[task]
    examples = []
    for row in _jsonl(pathlib.Path(path)):
        context = row[ctx_key]
        if task == "dialogue":
            context = f"{context}\n\n{row.get('dialogue_history', '')}".strip()
        examples.append(GroundingExample(f"halueval/{task}", context, row[right_key], False))
        examples.append(GroundingExample(f"halueval/{task}", context, row[wrong_key], True))
    return examples


def load_aggrefact(path: str | pathlib.Path | None = None) -> list[GroundingExample]:
    """LLM-AggreFact claims (label 1 = supported). Reads a local .jsonl/.csv export, else the hub."""
    if path is not None:
        path = pathlib.Path(path)
        if path.suffix == ".csv":
            with open(path, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        else:
            rows = _jsonl(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets not installed — run: pip install spaniq[benchmark], "
                "or pass a local export with aggrefact=PATH"
            ) from exc
        rows = list(load_dataset("lytang/LLM-AggreFact", split="test"))
    return [
        GroundingExample(
            source=f"aggrefact/{r.get('dataset', 'unknown')}",
            context=r["doc"],
            output=r["claim"],
            hallucinated=int(r["label"]) == 0,
        )
        for r in rows
    ]


def sample(examples: list[GroundingExample], limit: int | None, seed: int = 0):
    """Up to limit examples, half hallucinated where possible, in a fixed order."""
    if limit is None or len(examples) <= limit:
        return list(examples)
    rng = random.Random(seed)
    bad = [e for e in examples if e.hallucinated]
    good = [e for e in examples if not e.hallucinated]
    rng.shuffle(bad)
    rng.shuffle(good)
    n_bad = min(len(bad), limit // 2)
    n_good = min(len(good), limit - n_bad)
    n_bad = min(len(bad), limit - n_good)
    return bad[:n_bad] + good[:n_good]


def load(name: str, path: str | None = None) -> list[GroundingExample]:
    """Dispatch by dataset name: local, ragtruth, halueval_qa, halueval_summarization,
    halueval_dialogue, aggrefact."""
    if name == "local":
        return load_local()
    if name == "aggrefact":
        return load_aggrefact(path)
    if path is None:
        raise ValueError(f"dataset '{name}' needs a local path: {name}=PATH")
    if name == "ragtruth":
        return load_ragtruth(path)
    if name.startswith("halueval_"):
        return load_halueval(path, name.removeprefix("halueval_"))
    raise ValueError(
        f"unknown dataset '{name}' (valid: local, ragtruth, halueval_qa, "
        "halueval_summarization, halueval_dialogue, aggrefact)"
    )
