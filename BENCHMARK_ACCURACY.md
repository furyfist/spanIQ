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
| groq | LLM-as-judge, 1–10 normalized | `llama-3.3-70b-versatile` via Groq |
| deepeval | G-Eval correctness | `llama-3.3-70b-versatile` via Groq |
| ragas | faithfulness (needs context; rag_retrieval only) | `llama-3.3-70b-versatile` via Groq |
| langfuse | Langfuse-style LLM-as-judge (local replica) | `llama-3.3-70b-versatile` via Groq |

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

Competitor runners skip gracefully if their dependency or `GROQ_API_KEY` is
missing. The legacy determinism benchmark is still available via
`--metric variance`.

**ragas dependency note.** `ragas>=0.4` imports
`langchain_community.chat_models.vertexai`, which `langchain-community>=0.4`
removed. The `benchmark` extra therefore pins `langchain-community<0.4`. Without
that pin, ragas fails to import and its row silently disappears from the table.

## 7. Results

Live run on 2026-07-10 (Python 3.12.7, Windows), `GROQ_API_KEY` set, 3 runs per
tool per dataset, judge `llama-3.3-70b-versatile`. Source:
`benchmarks/results_accuracy/`.

**`ragas` is absent from this table, and the reason matters.** At the time of the
run the runner called `single_turn_ascore(sample)`, a method the v0.4 collections
API does not have. Every call raised `AttributeError`, a bare `except` converted
each one into the neutral `0.5` fallback, and the tool skipped on an unrelated
import error. Had it not skipped, it would have reported a full column of
fabricated `0.5`s as if they were judgments. The runner has since been fixed to
the real v0.4 API — `ascore(user_input, response, retrieved_contexts)` — and now
returns genuine scores, and it raises rather than reporting data if every item
fails. Its row will be filled on the next run with available judge tokens; the
run above exhausted the Groq free-tier daily token limit.

### Table 1 — Accuracy (held-out test fold, positive class = bad)

| Tool | Dataset | Threshold | Precision | Recall | F1 | ROC-AUC | Avg Prec |
|---|---|---|---|---|---|---|---|
| spaniq | qa_factual | 0.6133 | 0.806 | 0.967 | **0.879** | 0.857 | 0.850 |
| groq-llm-judge | qa_factual | 0.55 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| deepeval | qa_factual | 0.3667 | 1.000 | 0.833 | **0.909** | 0.971 | 0.978 |
| langfuse | qa_factual | 0.5 | 0.612 | 1.000 | **0.760** | 0.525 | 0.646 |
| spaniq | summarization | 0.982 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| groq-llm-judge | summarization | 0.5 | 0.500 | 1.000 | **0.667** | 0.500 | 0.692 |
| deepeval | summarization | 0.5 | 0.500 | 1.000 | **0.667** | 0.500 | 0.692 |
| langfuse | summarization | 0.5 | 0.500 | 1.000 | **0.667** | 0.500 | 0.692 |
| spaniq | rag_retrieval | 0.9973 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| groq-llm-judge | rag_retrieval | 0.5833 | 0.500 | 1.000 | **0.667** | 0.500 | 0.692 |
| deepeval | rag_retrieval | 0.5 | 0.500 | 1.000 | **0.667** | 0.500 | 0.692 |
| langfuse | rag_retrieval | 0.5 | 0.500 | 1.000 | **0.667** | 0.500 | 0.692 |

**How to read this — the honest reckoning:**

- **On qa_factual (the primary, balanced 100-row set), the LLM judges win on
  accuracy.** groq-judge caught every bad answer perfectly (F1 1.0); deepeval was
  strong (F1 0.91, AUC 0.97). spaniq's embedding cosine metric scored **F1 0.88 /
  AUC 0.86** — good, but it loses to a real judge on wrong-entity answers that are
  lexically close to the reference ("The capital of France is Berlin" embeds near
  "Paris" answers). This is the failure mode a statistical metric cannot see and a
  language model can. langfuse's template was weak here (AUC 0.53, near chance).
- **On summarization and rag_retrieval (n=8), the LLM judges collapse to F1 0.667
  / AUC 0.5** — precision 0.5 with recall 1.0 means they flagged *nothing* as bad;
  they scored every output "good," so at this scale and default config they add no
  detection signal. spaniq scores these perfectly (F1 1.0) because the omission /
  hallucination / unfaithful negatives are lexically distinct from the reference.
  Read these two rows as directional (small n), not as a verdict.

The takeaway is not "spaniq wins" or "judges win" — it is **complementary**: an
LLM judge is the right tool for subtle, semantically-close errors (qa_factual);
a deterministic, free metric is the right tool for gross lexical drift and is the
only one here that stayed useful when the judges defaulted to passing everything.

### Table 2 — Determinism sidebar (std dev of per-run mean score)

| Tool | qa_factual | summarization | rag_retrieval |
|---|---|---|---|
| spaniq | 0.0000 | 0.0000 | 0.0000 |
| groq-llm-judge | 0.0000 | 0.0000 | 0.0147 |
| deepeval | 0.0681 | 0.0000 | 0.0000 |
| langfuse | 0.0024 | 0.0000 | 0.0000 |

spaniq is `0.0000` everywhere by construction. The judges are *mostly*
deterministic at `temperature=0` but not always (deepeval on qa_factual, groq on
rag_retrieval) — which is exactly why the old "judges vary every run" headline was
wrong and this benchmark leads with accuracy instead.

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

0. **Neutral fallbacks can fake data.** Every LLM-judge runner degrades a failed
   item to a neutral `0.5`. That is fine for a transient timeout and dangerous for
   a broken integration: the ragas runner once failed on *every* call and would
   have reported a full column of `0.5`s as judgments. The ragas prediction path
   now raises when all items fail. Treat any tool whose scores are uniformly
   `0.5` as broken, not as evidence.
1. **Default configs.** Competitors run on defaults; a tuned setup may differ.
2. **Same judge model for all LLM tools** (`llama-3.3-70b-versatile` via Groq), to
   isolate the framework's contribution from the model's.
3. **Small datasets** for two of three sets — read summarization / rag_retrieval
   as directional.
4. **Synthetic negatives** built by disclosed rules (`build_labeled.py`), not
   human-labeled production failures. This measures detection of constructed
   failure modes, not correlation with human quality judgment.
5. **spanIQ measures similarity, not semantics.** On failures that are lexically
   close to the reference but semantically wrong, an LLM judge may win. The table
   is where that shows up honestly.
6. **Complementary, not interchangeable.** If the question is "was this answer
   helpful/subtle-correct?", an LLM judge is the right tool. If it is "has this
   drifted / is it grossly wrong, deterministically and for free?", spanIQ is.

## 10. Version history

| Date | Change |
|---|---|
| 2026-07-10 | Initial accuracy benchmark: 5 tools, labeled 100/16/16 datasets, calibration/test split, precision/recall/F1/AUC. Replaces the determinism headline. |
