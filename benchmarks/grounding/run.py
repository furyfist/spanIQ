"""Score grounding datasets and write summary.md and predictions.csv.

    python -m benchmarks.grounding.run --datasets local
    python -m benchmarks.grounding.run --datasets ragtruth=../RAGTruth/dataset --scorers unsupported_claims,nli
    python -m benchmarks.grounding.run --datasets halueval_qa=../HaluEval/data/qa_data.json --limit 1000

Every scorer returns how hallucinated an output looks, higher is worse.
novel_words is reported to show why it is a drift signal and not a
hallucination score: a faithful paraphrase uses new words too.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
from collections.abc import Callable

from benchmarks.analysis.metrics import (
    NEGATIVE,
    POSITIVE,
    best_f1_threshold,
    confusion,
    f1,
    roc_auc,
)
from benchmarks.config import GROQ_MODEL, GROUNDING_RESULTS_DIR
from benchmarks.grounding.loaders import GroundingExample, load, sample
from spaniq.signals.grounding import grounding_signals

Scorer = Callable[[GroundingExample], float]


def _unsupported_claims(e: GroundingExample) -> float:
    s = grounding_signals(e.output, e.context)
    return max(s["unsupported_numbers"], s["unsupported_names"])


def _novel_words(e: GroundingExample) -> float:
    return grounding_signals(e.output, e.context)["novel_words"]


def _nli_scorer() -> Scorer:
    from spaniq.signals.nli import check_groundedness, get_nli

    model = get_nli()
    return lambda e: 1.0 - check_groundedness(e.output, e.context, model).score


_JUDGE_PROMPT = """\
You check whether a response is supported by its source.

Source:
{context}

Response:
{output}

On a scale from 1 to 10, how fully is every claim in the response supported by the source?
Respond with ONLY a single integer from 1 to 10."""


def _judge_scorer() -> Scorer:
    from benchmarks.runners._judge import score_from_ten, with_retry

    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("groq SDK not installed — run: pip install spaniq[groq]") from exc
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise OSError("GROQ_API_KEY not set in environment")
    client = Groq(api_key=api_key)

    def score(e: GroundingExample) -> float:
        prompt = _JUDGE_PROMPT.format(context=e.context[:6000], output=e.output)
        resp = with_retry(
            lambda: client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
        )
        return 1.0 - score_from_ten(resp.choices[0].message.content or "")

    return score


SCORERS: dict[str, Callable[[], Scorer]] = {
    "unsupported_claims": lambda: _unsupported_claims,
    "novel_words": lambda: _novel_words,
    "nli": _nli_scorer,
    "judge": _judge_scorer,
}
DEFAULT_SCORERS = ("unsupported_claims", "novel_words")


def evaluate(examples: list[GroundingExample], scores: list[float]) -> dict:
    """AUC and best-threshold F1 with hallucinated as the positive class."""
    labels = [POSITIVE if e.hallucinated else NEGATIVE for e in examples]
    goodness = [1.0 - s for s in scores]
    threshold = best_f1_threshold(labels, goodness)
    return {
        "n": len(examples),
        "hallucinated": sum(e.hallucinated for e in examples),
        "auc": roc_auc(labels, goodness),
        "best_f1": f1(confusion(labels, goodness, threshold)),
    }


def parse_datasets(spec: str) -> list[tuple[str, str | None]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, path = part.partition("=")
        out.append((name.strip(), path.strip() or None))
    return out


def render_summary(rows: list[dict]) -> str:
    lines = [
        "# Grounding benchmark",
        "",
        "Positive class: hallucinated. AUC 0.5 is chance. Best F1 uses the threshold that "
        "maximises F1 on the same data, so it is optimistic.",
        "",
        "| dataset | scorer | n | hallucinated | AUC | best F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['scorer']} | {r['n']} | {r['hallucinated']} | "
            f"{r['auc']:.3f} | {r['best_f1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(
    datasets: list[tuple[str, str | None]] | None = None,
    scorers: list[str] | None = None,
    limit: int | None = None,
    output_dir: str | pathlib.Path = GROUNDING_RESULTS_DIR,
) -> list[dict]:
    datasets = datasets or [("local", None)]
    scorers = scorers or list(DEFAULT_SCORERS)
    for name in scorers:
        if name not in SCORERS:
            raise ValueError(f"unknown scorer '{name}' (valid: {', '.join(SCORERS)})")

    built: dict[str, Scorer] = {}
    for name in scorers:
        try:
            built[name] = SCORERS[name]()
        except (ImportError, OSError) as exc:
            print(f"  skipping {name}: {exc}")

    rows, predictions = [], []
    for ds_name, path in datasets:
        examples = sample(load(ds_name, path), limit)
        for sc_name, scorer in built.items():
            scores = [scorer(e) for e in examples]
            row = {"dataset": ds_name, "scorer": sc_name, **evaluate(examples, scores)}
            rows.append(row)
            print(f"  [{sc_name}] {ds_name}: n={row['n']} auc={row['auc']:.3f}")
            predictions += [
                {
                    "dataset": ds_name,
                    "scorer": sc_name,
                    "source": e.source,
                    "hallucinated": int(e.hallucinated),
                    "score": round(s, 6),
                }
                for e, s in zip(examples, scores, strict=True)
            ]

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(render_summary(rows), encoding="utf-8")
    if predictions:
        with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    print(f"wrote {out / 'summary.md'}")
    return rows


def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m benchmarks.grounding.run")
    p.add_argument("--datasets", default="local", help="name or name=PATH, comma-separated")
    p.add_argument("--scorers", default=",".join(DEFAULT_SCORERS))
    p.add_argument("--limit", type=int, default=None, help="max examples per dataset")
    p.add_argument("--output", default=str(GROUNDING_RESULTS_DIR))
    a = p.parse_args()
    main(
        datasets=parse_datasets(a.datasets),
        scorers=[s.strip() for s in a.scorers.split(",") if s.strip()],
        limit=a.limit,
        output_dir=a.output,
    )


if __name__ == "__main__":
    _cli()
