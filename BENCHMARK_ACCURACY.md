# Accuracy Benchmark

This is the primary benchmark trust document. It supersedes the determinism
framing in `BENCHMARK_METHODOLOGY.md` (kept as a historical record). Where the old
benchmark asked "does a tool score the same every run?", this one asks the
question that matters: **does the tool actually catch bad outputs?**

Everything here is derived from the code in `benchmarks/`. The contract it obeys
is `docs/plans/benchmark_v2_accuracy.md`.

## 1. The question and the claim

> Given an output and its reference, does the tool flag it as **bad** when it is
> bad, and pass it when it is good?

Two honest claims this benchmark supports:

1. **Cost & determinism by construction.** spanIQ is free per trace and
   deterministic *because it makes no LLM call*. LLM judges are deterministic only
   conditionally (a live run showed the Groq judge at `temperature=0` was
   deterministic too) and are never free in production.
2. **Accuracy, stated plainly.** Here is exactly how well each tool catches bad
   outputs — precision, recall, F1, and threshold-free AUC / average precision.
   Where spanIQ's embedding metric loses to an LLM judge on subtle failures, the
   table says so.

## 2. Task framing and positive class

Binary detection. Each dataset item carries a ground-truth `label` (`good` /
`bad`) and, for bad rows, a `failure_kind`. Every tool emits a continuous `score`
in `[0, 1]` ("how good the output looks"); a per-tool threshold turns it into a
decision:

```
predicted = "good" if score >= threshold else "bad"
```

The **positive class is `bad`** — the failure we want to catch. So:

- **Recall (bad)** — of all genuinely bad outputs, how many did the tool catch?
  This is the money metric.
- **Precision (bad)** — of everything flagged bad, how much really was? (alert
  fatigue if low)
- **F1** — the harmonic mean, single-number summary.
- **ROC-AUC / Average precision** — threshold-free ranking quality, so the
  comparison doesn't hinge on any one cutoff.

## 3. Tools

Same five runners as the determinism benchmark, now returning per-item
predictions through one shared interface (`run_*_predictions` →
`LabeledResult`). The good/bad label always comes from the dataset, never from the
tool (`predictions_from_scores`), so the comparison is fair by construction.

| Tool | What it scores | Judge model |
|---|---|---|
| spaniq | embedding cosine similarity (deterministic, no LLM) | none |
| groq | LLM-as-judge, 1–10 normalized | `openai/gpt-oss-120b` via Groq |
| deepeval | G-Eval correctness | `openai/gpt-oss-120b` via Groq |
| ragas | faithfulness (needs context; rag_retrieval only) | `openai/gpt-oss-120b` via Groq |
| langfuse | Langfuse-style LLM-as-judge (local replica) | `openai/gpt-oss-120b` via Groq |

The judge model is `benchmarks.config.GROQ_MODEL`; override it with
`SPANIQ_JUDGE_MODEL`.

## 4. Datasets (labeled)

Built by `benchmarks/datasets/build_labeled.py` from the immutable `*_seed.jsonl`
sources; provenance of every negative is in `benchmarks/datasets/LABELING.md`.

| Dataset | Rows | Good / Bad | Failure kinds used |
|---|---|---|---|
| qa_factual | 100 | 50 / 50 | wrong_entity, contradiction |
| summarization | 16 | 8 / 8 | omission, hallucination |
| rag_retrieval | 16 | 8 / 8 | unfaithful_to_context, contradiction |

qa_factual is the primary set (large enough for stable P/R). summarization and
rag_retrieval are smaller and their per-metric numbers are directional.

## 5. Threshold selection (the fairness rule)

Applied identically to every tool, in `benchmarks/analysis/calibrate.py`:

1. Split each dataset into a calibration fold and a test fold with a fixed,
   committed seed (`CALIBRATION_SEED = 20260710`, 50/50 split).
2. Pick the F1-maximizing threshold on the **calibration** fold only.
3. Report precision / recall / F1 on the **test** fold with that threshold.
4. Also report AUC / average precision on the test fold (threshold-free).

No tool — including spanIQ — is scored with a threshold tuned on the data it is
graded on.

## 6. How to reproduce

```bash
pip install -e ".[benchmark]"
export GROQ_API_KEY="your-key"   # free at console.groq.com; competitors only

# spanIQ only — no key needed, verifies the accuracy path in seconds
spaniq benchmark --tool spaniq --runs 5 --metric accuracy

# full suite
spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 5 --metric accuracy

# artifacts
cat benchmarks/results/accuracy_summary.md
```

Competitor runners skip with a printed reason if their dependency or
`GROQ_API_KEY` is missing, or if a judge call still fails after retrying (for
example when the daily token quota runs out). A skipped tool has no row; it never
gets placeholder scores. The legacy determinism benchmark is still available via
`--metric variance`.

**ragas dependency note.** `ragas>=0.4` imports
`langchain_community.chat_models.vertexai`, which `langchain-community>=0.4`
removed. The `benchmark` extra therefore pins `langchain-community<0.4`. Without
that pin, ragas fails to import and its row silently disappears from the table.

## 7. Results

Live run on 2026-10-19 (Python 3.12, Windows), 2 runs per tool per dataset, judge
`openai/gpt-oss-120b` via Groq. Source: `benchmarks/results_accuracy/`.

**What changed since the 2026-07-10 run, and why the old summarization and
rag_retrieval rows were withdrawn.** Two defects made those rows meaningless:

- *The judge rows were not judgments.* The July run hit Groq's free-tier daily
  token limit after qa_factual. Every runner turned a failed call into a neutral
  `0.5`, so 47 of the 48 judge scores on those two sets were fallbacks, which is
  why all three judges showed exactly F1 0.667 / AUC 0.500. Runners now retry
  transient failures and raise when a call still fails, so the tool is skipped
  with a visible reason instead.
- *spanIQ's 1.0 was a label leak.* Every good row was the reference answer copied
  verbatim, so similarity to the reference was exactly 1.0 for every good row.
  Good rows are now hand-written paraphrases (see `LABELING.md`).

The re-run below hit the same daily token limit after qa_factual, and this time
the judges on summarization and rag_retrieval were **skipped, not faked**. Their
rows are pending a run with fresh quota:
`spaniq benchmark --tool groq,deepeval,ragas,langfuse --dataset summarization,rag_retrieval --runs 2`.
Groq also retired `llama-3.3-70b-versatile` (calls now return 404), so the judge
model changed; qa_factual judge numbers are not comparable with the July ones.

### Table 1 — Accuracy (held-out test fold, positive class = bad)

| Tool | Dataset | Threshold | Precision | Recall | F1 | ROC-AUC | Avg Prec |
|---|---|---|---|---|---|---|---|
| spaniq | qa_factual | 0.6133 | 0.806 | 0.967 | **0.879** | 0.857 | 0.850 |
| groq-llm-judge | qa_factual | 0.55 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| deepeval | qa_factual | 0.30 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| langfuse | qa_factual | 0.60 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| spaniq | summarization | 0.9639 | 0.500 | 1.000 | **0.667** | 0.250 | 0.566 |
| spaniq | rag_retrieval | 0.9946 | 0.500 | 1.000 | **0.667** | 0.063 | 0.378 |
| judges | summarization, rag_retrieval | — | — | — | pending | — | — |

**How to read this:**

- **On qa_factual every LLM judge beats spanIQ.** All three caught every bad answer
  (F1 1.0). spanIQ's embedding similarity scored F1 0.879 / AUC 0.857: it misses
  wrong-entity answers that are worded like the reference ("The capital of France
  is Berlin" embeds close to the "Paris" answers).
- **On summarization and rag_retrieval spanIQ does worse than chance** (AUC 0.25
  and 0.06). The bad rows are the reference plus an invented clause, or the
  reference clipped short, so they share more words with the reference than a
  correct paraphrase does. Similarity to a reference rewards copying, and cannot
  see an added hallucination. These sets are small (n=8 per class), but the
  direction is unambiguous.

The honest conclusion: spanIQ's metrics are not a substitute for an LLM judge at
deciding whether an individual answer is right. What they offer is a
deterministic, free signal of *change* against a baseline (drift, structural
shifts, a component breaking in a pipeline), which is what the monitoring and
attribution layers use them for.

### Table 2 — Determinism sidebar (std dev of per-run mean score)

| Tool | qa_factual | summarization | rag_retrieval |
|---|---|---|---|
| spaniq | 0.0000 | 0.0000 | 0.0000 |
| groq-llm-judge | 0.0000 | pending | pending |
| deepeval | 0.0495 | pending | pending |
| langfuse | 0.0000 | pending | pending |

spaniq is `0.0000` everywhere by construction. The judges are mostly
deterministic at `temperature=0` but not always (deepeval on qa_factual), which is
why this benchmark leads with accuracy rather than determinism.

## 8. Audit trail

Every run writes to the results directory:

- `accuracy.csv` — one row per tool+dataset: threshold, precision, recall, F1,
  ROC-AUC, average precision, fold sizes, determinism std dev.
- `predictions.csv` — every item's true label, failure kind, and mean score, so
  any confusion matrix (and therefore any P/R number) can be rebuilt by hand.
- `accuracy_summary.md` — the human-readable tables.

The `predictions.csv` is the "show your work" artifact: nothing in the accuracy
table is computed from anything not in that file.

## 9. Fairness and limitations

0. **Neutral fallbacks faked data, and no longer exist.** Runners used to turn a
   failed call into a neutral `0.5`; under rate limiting that produced whole
   columns of fake scores (the withdrawn July summarization / rag rows). Every
   runner now retries with backoff and raises `JudgeError` if a call still fails,
   and the tool is skipped. Treat any tool whose scores are uniformly `0.5` as
   broken, not as evidence.
1. **Default configs.** Competitors run on defaults; a tuned setup may differ.
2. **Same judge model for all LLM tools** (`openai/gpt-oss-120b` via Groq), to
   isolate the framework's contribution from the model's.
3. **Small datasets** for two of three sets — read summarization / rag_retrieval
   as directional.
4. **Synthetic negatives** built by disclosed rules (`build_labeled.py`), not
   human-labeled production failures. This measures detection of constructed
   failure modes, not correlation with human quality judgment.
5. **spanIQ measures similarity, not semantics.** On failures that are lexically
   close to the reference but semantically wrong, an LLM judge may win. The table
   is where that shows up honestly.
6. **Complementary, not interchangeable.** If the question is "is this answer
   correct?", an LLM judge is the right tool, and this benchmark shows it. If it
   is "has this output distribution changed against its baseline,
   deterministically and for free?", spanIQ is.

## 10. Version history

| Date | Change |
|---|---|
| 2026-07-10 | Initial accuracy benchmark: 5 tools, labeled 100/16/16 datasets, calibration/test split, precision/recall/F1/AUC. Replaces the determinism headline. |
| 2026-10-19 | Withdrew the summarization / rag_retrieval rows (judge rows were rate-limit fallbacks, spaniq's 1.0 was a label leak). Runners fail loudly, good rows are paraphrases, judge model is `openai/gpt-oss-120b`. Judge rows for the two small sets pending fresh quota. |
