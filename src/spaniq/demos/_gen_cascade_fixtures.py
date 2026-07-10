"""Generate offline fixtures for the cascade attribution demo.
Run once: python -m spaniq.demos._gen_cascade_fixtures
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

SEED = 42
N_HEALTHY = 100
N_BROKEN = 100
BREAK_TRACE = 101
GENERATION_LAG = 7

OUTPUT_DIR = Path(__file__).parent / "fixtures" / "cascade"


def _ts(base: datetime, offset_seconds: int) -> str:
    return (base + timedelta(seconds=offset_seconds)).isoformat()


def _make_trace(
    i: int,
    base: datetime,
    retrieval_output: str,
    generation_output: str,
) -> dict:
    return {
        "trace_id": str(uuid4()),
        "input": "What is the capital of France?",
        "output": generation_output,
        "timestamp": _ts(base, i * 5),
        "components": [
            {
                "name": "retrieval",
                "kind": "retrieval",
                "output": retrieval_output,
                "latency_ms": random.gauss(120, 10),
                "error": False,
            },
            {
                "name": "search_tool",
                "kind": "execute_tool",
                "output": "{'result': 'Paris is the capital of France.'}",
                "latency_ms": random.gauss(300, 20),
                "error": False,
            },
            {
                "name": "generation",
                "kind": "chat",
                "output": generation_output,
                "latency_ms": random.gauss(900, 50),
                "error": False,
            },
        ],
    }


def generate(output_dir: Path = OUTPUT_DIR) -> Path:
    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    traces = []

    healthy_ctx = [
        "Paris is the capital and largest city of France.",
        "France's capital city is Paris, located in northern France.",
        "The capital of France is Paris, home to the Eiffel Tower.",
    ]
    healthy_gen = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "France's capital is Paris.",
    ]

    broken_ctx = [
        "",
        "N/A",
        "No relevant documents found.",
        "[empty context]",
    ]
    broken_gen_early = [
        "Based on my knowledge, the capital of France is Paris.",
        "I believe it's Paris, though I'm not entirely certain.",
        "Paris is likely the answer here.",
    ]
    broken_gen_late = [
        "I cannot determine the answer from the provided context.",
        "The context does not contain relevant information.",
        "I'm unable to answer with confidence given the context.",
    ]

    for i in range(N_HEALTHY + N_BROKEN):
        broken = i >= BREAK_TRACE
        generation_broken = i >= BREAK_TRACE + GENERATION_LAG

        retrieval_out = random.choice(broken_ctx) if broken else random.choice(healthy_ctx)

        if generation_broken:
            generation_out = random.choice(broken_gen_late)
        elif broken:
            generation_out = random.choice(broken_gen_early)
        else:
            generation_out = random.choice(healthy_gen)

        traces.append(_make_trace(i, base, retrieval_out, generation_out))

    out_path = output_dir / "traces.jsonl"
    with open(out_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    print(f"Generated {len(traces)} traces -> {out_path}")
    return out_path


if __name__ == "__main__":
    generate()
