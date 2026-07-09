# Benchmark v2 — Accuracy Contract

This is the specification the accuracy benchmark obeys. Every column of the final
results table should be predictable from this document alone. It supersedes the
determinism framing in `BENCHMARK_METHODOLOGY.md`, which measured a tautology (a
metric that makes no LLM call scores identically every run) and whose headline was
falsified by a live run where the Groq judge was also deterministic.

## The question this benchmark answers

> Given an output and a reference, does the tool correctly flag it as **bad** when
> it is bad, and pass it when it is good?

Determinism and cost are still reported, but as secondary facts. The headline is
now **accuracy at catching bad outputs**.

## Task framing

Binary detection. Each dataset item carries a ground-truth `label`:

- `label = "good"` — the `output` is an acceptable answer given the reference.
- `label = "bad"`  — the `output` contains a real failure (see taxonomy below).

Every tool emits a continuous `score` in `[0, 1]` per item. A per-tool decision
**threshold** turns that score into a predicted label:

```
predicted = "good" if score >= threshold else "bad"
```

Thresholds are calibrated by a fixed, disclosed rule (see "Threshold selection"),
not hand-picked per tool — that is the difference between an honest benchmark and
a vendor benchmark.

## Positive class

The **positive class is `bad`** — the event we want to detect. This choice makes
the metric names read the way an operator thinks:

- **Recall** = of all genuinely bad outputs, what fraction did the tool catch?
  This is the money metric. A monitor that misses bad outputs is useless.
- **Precision** = of everything the tool flagged as bad, what fraction really was?
  Low precision means alert fatigue.
- **F1** = harmonic mean of the two, the single-number summary.

Because a tool's `score` is "how *good* the output looks," a **low** score means
"likely bad." The decision above flags `bad` when `score < threshold`, so a
bad-positive prediction corresponds to a low score, as expected.

## Reported metrics

| Metric | Definition | Threshold-dependent? |
|---|---|---|
| Precision (bad) | TP / (TP + FP) | yes |
| Recall (bad)    | TP / (TP + FN) | yes |
| F1 (bad)        | 2·P·R / (P + R) | yes |
| AUC (ROC)       | ranking quality of `score` vs `label`, threshold-free | no |
| Average precision | area under precision–recall curve, threshold-free | no |

Where, with `bad` as positive:

- **TP** — labeled `bad`, predicted `bad`
- **FP** — labeled `good`, predicted `bad`
- **FN** — labeled `bad`, predicted `good`
- **TN** — labeled `good`, predicted `good`

The two threshold-free metrics matter because they let a reader compare tools
without trusting any threshold choice: AUC / AP measure whether the tool's score
*ranks* bad below good, independent of where the line is drawn.

## Failure taxonomy (for the `bad` outputs)

Negatives must be realistic failure modes, not gibberish, or the benchmark tests
nothing interesting. `failure_kind` on each bad row is one of:

| `failure_kind` | What it is | Example |
|---|---|---|
| `hallucination` | invented fact absent from reference/context | "Paris, population 40 million" |
| `wrong_entity` | right shape, wrong specific answer | "The capital of France is Berlin." |
| `omission` | drops a key required element of the reference | summary that skips the 45% figure |
| `contradiction` | asserts the opposite of the reference | "solar panels emit more CO2 than coal" |
| `unfaithful_to_context` | (RAG) answer not supported by the retrieved passage | claim absent from the provided context |

`good` rows carry `failure_kind: null`.

## Dataset shape

Existing fields are preserved; two are added. Every row:

```json
{
  "input": "...",
  "reference_output": "...",
  "context": "...",           // rag_retrieval only
  "output": "...",
  "label": "good" | "bad",
  "failure_kind": null | "hallucination" | "wrong_entity" | "omission" |
                  "contradiction" | "unfaithful_to_context",
  "category": "...",
  "source": "..."
}
```

Each dataset is **class-balanced** (~50/50 good/bad) and expanded to a size large
enough that precision/recall are stable (target ≥ 100 rows per dataset; the v4
seed data of 20/8/8 correct rows is the good half's starting point, perturbed into
matching bad rows). The provenance of every negative is recorded in
`benchmarks/datasets/LABELING.md` so labels are auditable by hand.

## Threshold selection (the fairness rule)

Applied identically to every tool:

1. Split each dataset into a **calibration** fold and a **test** fold with a fixed,
   committed random seed.
2. On the calibration fold only, pick the threshold that maximizes F1.
3. Report precision / recall / F1 on the **test** fold, using that threshold.
4. Also report AUC / average precision on the test fold (no threshold involved).

No tool — including spanIQ — is allowed a threshold tuned on the data it is scored
on. The seed and the rule are documented; changing the seed changes the specific
fold, never the method.

## What stays from v1

- **Cost** — real token-usage cost per tool, from `_cost.py`. spanIQ = $0.00.
- **Determinism** — still computed and reported, as a secondary sidebar, honestly
  (including where LLM judges were deterministic).
- **Graceful skip** — competitor runners still skip cleanly on missing dep / key.
- **Reproducibility** — synthetic committed datasets; anyone can rerun and get the
  same inputs.

## The two honest claims this enables

1. **Cost & determinism by construction** — spanIQ is free per trace and
   deterministic *because it makes no LLM call*; judges are deterministic only
   conditionally and are never free in production.
2. **Accuracy, stated plainly** — here is exactly how well each tool catches bad
   outputs (precision / recall / F1 / AUC), and therefore where each tool is the
   right choice. If spanIQ's embedding metric loses on subtle failures, the table
   says so — a losing-but-true number beats a winning-but-false one.

## Out of scope (still)

- Human-annotation accuracy correlation (would need a human panel).
- Large-scale throughput (this stays a first-run-sized benchmark).
- Per-vendor price quotes (the flat $0.27/1M proxy stays, to avoid pricing drift).
