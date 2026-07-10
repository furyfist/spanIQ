# Benchmark Methodology (determinism — superseded)

> **Superseded by the accuracy benchmark.** This document describes the
> *determinism* benchmark. A live run showed that LLM judges are frequently
> deterministic too (the Groq judge scored std dev 0.0000 on all three datasets
> at `temperature=0`), so the "LLM judges produce different scores every run"
> framing is not supported by the data. It is retained here as a historical
> record, with its Section 8 tables updated to the **actual** committed results
> rather than `[PENDING]`.
>
> **The primary benchmark is now accuracy** — precision / recall / F1 / AUC on
> labeled good-vs-bad outputs. See **[`BENCHMARK_ACCURACY.md`](BENCHMARK_ACCURACY.md)**
> for the trust document and results, and `docs/plans/benchmark_v2_accuracy.md`
> for the contract. The durable, true facts from this benchmark carry over: spanIQ
> is deterministic **and $0/trace by construction** because it makes no LLM call.

This document describes exactly how the spanIQ determinism benchmark is run, what it measures, what it deliberately does not measure, and how anyone can reproduce the numbers themselves.

## Why this file exists

Without a methodology document, the benchmark table is just numbers. With it, the numbers become evidence you can check. This document serves three readers:

1. The developer who sees the claim and asks "how did you configure the competitors?"
2. The reproducer who wants to clone the repo and verify the numbers.
3. The reviewer who wants to judge whether the claims are honest.

Everything here is derived from the code in `benchmarks/`. Where a number comes from a live run, the run is described. Where a number requires an API key we did not have at publish time, it is marked `[PENDING]` rather than guessed.

## 1. What this benchmark measures (and what it doesn't)

This is the most important section. The claim is narrow on purpose.

**What it measures:**

- **Score variance** across N identical runs on the same dataset. The question is "if you run the same evaluation 5 times, do you get the same number?" This is a measurement-instrument stability test, not an accuracy test.
- **Execution time** per evaluation run, wall-clock, measured with `time.perf_counter()`.
- **Benchmark cost** with Groq, which is $0.00 on the free tier for every tool in this suite.
- **Estimated production cost** extrapolated from the actual token counts observed during the Groq runs (see the cost model in Section 6).

**What it does NOT measure:**

- Which tool catches more bugs, hallucinations, or quality issues. That is a different benchmark and requires human-labeled ground truth.
- Accuracy correlation with human judgment. Every LLM-judge tool would need a human annotation baseline to claim that.
- Performance at scale (thousands of traces). This benchmark uses small datasets by design, for reproducibility.
- The optimal configuration of any competitor. All tools run on their default settings.

The point of this framing: here is what the numbers prove and here is the boundary. A deterministic statistical metric and an LLM judge measure different things — this benchmark compares their *stability*, not their *insight*. See Section 10 for where each kind of tool is the right choice.

## 2. The claim (as originally stated — the second half is now known false)

> ~~spanIQ produces identical scores on identical inputs, every time, at $0 cost. LLM-as-judge tools produce different scores on every run.~~

The first sentence holds (Section 7). The second does **not**: the live run in
Section 8 shows LLM judges are frequently deterministic at `temperature=0`. The
supportable claim is now: *spanIQ is deterministic and $0/trace by construction;
LLM judges are deterministic only conditionally and are never free.* Accuracy —
not variance — is the axis that actually distinguishes the tools; see
[`BENCHMARK_ACCURACY.md`](BENCHMARK_ACCURACY.md).

## 3. Tools benchmarked

Five runners live in `benchmarks/runners/`. Versions below come from `pyproject.toml` (the `benchmark` optional-dependency group); judge model and scoring come from each runner file.

| Tool | Version constraint | What it evaluates | Judge model | How it scores |
|---|---|---|---|---|
| **spaniq** | 0.4.0 | `SemanticSimilarityMetric` (embedding cosine similarity, deterministic) | None | 0–1 similarity to the expected output |
| **groq** | `groq>=1.0.0` | Custom LLM-as-judge prompt | `llama-3.3-70b-versatile` via Groq | 1–10 integer, normalized to 0–1 |
| **deepeval** | `deepeval>=1.0.0` | G-Eval correctness | `llama-3.3-70b-versatile` via Groq | 0–1 (G-Eval score) |
| **ragas** | `ragas>=0.4.0,<1.0.0` | Faithfulness (collections API) | `llama-3.3-70b-versatile` via Groq | 0–1 (supported claims / total claims) |
| **langfuse** | N/A (custom) | LLM-as-judge prompt replicating Langfuse's methodology | `llama-3.3-70b-versatile` via Groq | 1–10, JSON output, normalized to 0–1 |

Notes from reading the runners:

- The spaniq runner (`spaniq_runner.py`) runs `SemanticSimilarityMetric` through `spaniq.core.evaluate.evaluate`. It makes no LLM calls and reports `cost_usd=0.0` for every run.
- The groq, deepeval, ragas, and langfuse runners all set judge `temperature=0.0` and all point at the same Groq model so the only variable between them is the framework.
- **Langfuse runner caveat:** this runner replicates the LLM-as-a-Judge evaluation *methodology* that Langfuse provides in its platform. It does not call Langfuse servers and does not use the Langfuse SDK. The prompt template (structured input, 1–10 scale, JSON score + reasoning) and the judge call match what a Langfuse user configures in the UI. We benchmark the evaluation methodology, not the platform integration.

## 4. Datasets

Three JSONL datasets live in `benchmarks/datasets/`. All are synthetic and committed to the repo, so a reproducer runs against the exact same inputs. Record counts and field names below were read directly from the files.

**qa_factual.jsonl** — 20 records

- Fields: `input` (question), `reference_output` (ground truth), `output` (the answer being scored), `category`, `source`
- Used by: spaniq, groq, deepeval, langfuse
- Source: synthetic, committed to repo

**summarization.jsonl** — 8 records

- Fields: `input` (source text), `reference_output` (reference summary), `output` (summary being scored), `category`, `source`
- Used by: spaniq, groq, deepeval, langfuse
- Source: synthetic, committed to repo

**rag_retrieval.jsonl** — 8 records

- Fields: `input` (question), `context` (retrieved passage), `reference_output` (ground truth), `output` (answer being scored), `category`, `source`
- Used by: ragas (the only runner that requires `context`, for its faithfulness metric); also usable by spaniq, groq, deepeval, langfuse
- Source: synthetic, committed to repo

The ragas runner explicitly errors if asked to score a dataset with no `context` field, which is why it is paired with `rag_retrieval.jsonl`.

## 5. Benchmark configuration

```
Python version:   3.12.7
OS (publish run): Windows 11
Judge model:      llama-3.3-70b-versatile (via Groq, OpenAI-compatible endpoint)
Judge temperature: 0.0 (set explicitly in every LLM-judge runner)
Runs per tool:    5 (each tool evaluates the full dataset 5 times)
Groq API tier:    free
```

Parallelism, read from the runners:

- **spaniq** — sequential, runs the metric over all items per run.
- **groq** — sequential, with a 0.05s courtesy sleep between items to respect rate limits.
- **deepeval** — sequential, one G-Eval `measure()` per item.
- **ragas** — sequential per item, inside an `asyncio.run` per run.
- **langfuse** — concurrent: all items in a run are judged together via `asyncio.gather`.

Every LLM-judge runner wraps its per-item call in a try/except that falls back to a neutral score of `0.5` on error, so a transient API failure degrades one item rather than aborting the run.

## 6. Metric definitions

These are computed in `benchmarks/runners/spaniq_runner.py` (`BenchmarkResult`) and rendered in `benchmarks/analysis/report.py`. Definitions match the code exactly.

**Score variance (the primary metric).** For each run, take the mean of that run's per-item scores. Collect the N per-run means. The reported variance is the population variance of those means, and the reported std dev is its square root:

```
Given N runs, each producing a mean score μ_i:
  M = (1/N) Σ μ_i               # mean of the run means
  variance = (1/N) Σ (μ_i − M)²  # population variance
  std_dev  = sqrt(variance)
```

spanIQ's std dev is exactly `0.0` because it makes no LLM calls and the embedding similarity is deterministic. LLM-judge tools produce a non-zero std dev driven by sampling and floating-point non-determinism in the judge model.

**Execution time.** Wall-clock per full-dataset run, measured with `time.perf_counter()` around each run, reported as the mean across the N runs.

**Benchmark cost (Groq).** $0.00 for every tool — this is the actual cost incurred, because the runs use Groq's free tier. The non-zero numbers in the cost column are the *estimated production cost* defined next, not money spent.

**Estimated production cost.** The LLM-judge runners track token usage and apply a single flat rate, defined once in `benchmarks/runners/_cost.py`:

```python
_RATE_PER_MILLION = 0.27  # USD per 1M tokens, applied to both input and output

def token_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens + completion_tokens) * _RATE_PER_MILLION / 1_000_000
```

So the estimate is intentionally simple and symmetric: every token, input or output, costs $0.27 per million. The langfuse and groq runners read real `usage.prompt_tokens` / `usage.completion_tokens` from the Groq response; the ragas runner estimates tokens from sample text (~4 chars per token) because its faithfulness decomposition does not surface a single usage object. spaniq's cost is `0.0` because it never calls an LLM.

This flat rate is the number the code actually computes and the number a reproducer will see. It is not a vendor-specific quote. For reference, mainstream hosted judge models are in the same order of magnitude (e.g. GPT-4o-mini lists $0.15/1M input and $0.60/1M output), but we deliberately do not bake a third-party price into the benchmark output, because that price drifts and the methodology should not. Treat the cost column as "production-equivalent token cost at a representative rate," not as a quote for any specific provider.

## 7. How to reproduce

```bash
# 1. Clone the repo
git clone https://github.com/<username>/spaniq.git
cd spaniq

# 2. Install with benchmark dependencies
pip install -e ".[benchmark]"

# 3. (Optional) Set a Groq API key for the LLM-judge tools — free at console.groq.com
export GROQ_API_KEY="your-key-here"

# 4. Run spanIQ only — no API key needed, finishes in seconds
spaniq benchmark --tool spaniq --runs 5

# 5. Run the full suite (all five tools, 5 runs each)
spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 5

# 6. View results
cat benchmarks/results/summary.md
```

The same suite can be run without the installed entry point via `python -m benchmarks.run_benchmark --tool spaniq --runs 5`.

Notes:

- The spaniq-only run requires no API key and is the fastest way to verify the std-dev = 0.0 claim.
- Competitor runners skip gracefully if their dependency or `GROQ_API_KEY` is missing — you will see a `skipping <tool>: <reason>` line on stderr and that tool simply will not appear in the results table.
- The ragas runner only accepts a dataset that has a `context` field; point it at `rag_retrieval` (the default `--dataset all` handles this).
- Groq offers a free tier at console.groq.com with no credit card required.

## 8. Results

The rows below are the **actual committed results** from a live run on the
configuration in Section 5 (Python 3.12.7, Windows), source
`benchmarks/results/results.csv`. The run used **3 runs per tool per dataset**
(not the 5 stated as the default in Section 5 — noted here so the doc matches the
artifact). `GROQ_API_KEY` was set for this run, so the competitor rows are real,
not `[PENDING]`. The `ragas` runner did not appear in the output (skipped — its
live v0.4 + Groq `llm_factory` path remains unverified; see Section 10).

### Table 1 — Score variance (and why the headline does not hold)

| Tool | Dataset | Mean Score | Std Dev | Runs |
|---|---|---|---|---|
| spaniq | qa_factual | 0.5890 | **0.0000** | 3 |
| spaniq | summarization | 1.0000 | **0.0000** | 3 |
| spaniq | rag_retrieval | 1.0000 | **0.0000** | 3 |
| groq-llm-judge | qa_factual | 1.0000 | **0.0000** | 3 |
| groq-llm-judge | summarization | 1.0000 | **0.0000** | 3 |
| groq-llm-judge | rag_retrieval | 1.0000 | **0.0000** | 3 |
| deepeval | qa_factual | 0.6533 | 0.0309 | 3 |
| deepeval | summarization | 1.0000 | **0.0000** | 3 |
| deepeval | rag_retrieval | 1.0000 | **0.0000** | 3 |
| langfuse | qa_factual | 0.5583 | 0.0312 | 3 |
| langfuse | summarization | 0.6250 | 0.0884 | 3 |
| langfuse | rag_retrieval | 0.5833 | 0.0295 | 3 |

spanIQ's std dev is `0.0000` everywhere — that part holds. But so is
`groq-llm-judge`'s, on all three datasets, and `deepeval`'s on two of three. At
`temperature=0` with a tight prompt, an LLM judge is often deterministic. Only
`langfuse` (and `deepeval` on qa_factual) shows the non-zero spread the original
claim assumed for *all* judges. **The correct reading is not "spanIQ is
deterministic and judges are not" but "spanIQ is deterministic *and free by
construction*; judges are deterministic only conditionally and cost money."** The
determinism-of-spaniq fact is real; the determinism-*comparison* framing is what
the successor accuracy benchmark replaces.

### Table 2 — Cost comparison

Benchmark cost is the money actually spent (Groq free tier: $0.00 for all). The
estimated production cost is the token-based estimate the runners recorded during
this run, at the flat $0.27/1M rate from Section 6.

| Tool | Benchmark Cost (Groq) | Est. Production Cost (this run, all datasets) |
|---|---|---|
| spaniq | $0.00 | $0.0000 (no LLM calls) |
| groq-llm-judge | $0.00 | $0.0044 |
| deepeval | $0.00 | $0.0000 (usage not surfaced by this path) |
| langfuse | $0.00 | $0.0011 |

spanIQ's production cost is structurally $0.00: it never calls an LLM, so there
are no tokens to bill. This is the durable, defensible advantage — independent of
the variance framing.

### Table 3 — Execution time

Mean wall-clock per full-dataset run, from the same live run.

| Tool | Dataset | Mean Time |
|---|---|---|
| spaniq | qa_factual | 8.80s |
| spaniq | summarization | 0.15s |
| spaniq | rag_retrieval | 0.14s |
| groq-llm-judge | qa_factual | 26.65s |
| deepeval | qa_factual | 61.61s |
| langfuse | qa_factual | 3.94s |

The first spaniq dataset carries the one-time cost of loading the embedding
model, which is why qa_factual is slower than the two datasets evaluated after it
in the same process. The LLM-judge rows are network-bound and dominated by
per-item API latency.

## 9. Raw run logs

Every run writes its artifacts to `benchmarks/results/`:

```
benchmarks/results/
├── results.csv          # one row per tool+dataset: mean, std dev, variance, time, cost, per-run means
├── results.json         # same data, machine-readable
├── summary.md           # human-readable comparison table
└── variance_chart.html  # per-run mean score, plotted per tool
```

`results.csv` and `results.json` both include a `run_scores` field — the list of per-run mean scores — so anyone can recompute the std dev from the raw inputs and confirm the variance numbers in Section 8. The CSV is the "show your work" artifact: the variance column is derived from `run_scores` using the formula in Section 6, nothing hidden.

## 10. Known limitations and fairness statement

This is what separates an honest benchmark from a vendor benchmark.

**1. Default configurations.** Every competitor ran on its default settings. A user who tunes ragas's prompt or deepeval's G-Eval parameters might see lower variance. We benchmark the out-of-the-box experience, not the optimized one.

**2. Same judge model for all LLM tools.** Every LLM-judge runner used `llama-3.3-70b-versatile` via Groq. In production, ragas and deepeval commonly default to GPT-4o. Different judge models have different variance characteristics. We hold the judge model constant on purpose, to isolate the *framework's* contribution to variance from the *model's*.

**3. Small datasets.** The benchmark uses 20 / 8 / 8 items across the three datasets. At larger scales, LLM-judge variance on the *mean* may shrink (law of large numbers), but per-item variance remains. We benchmark at a scale typical of a developer's first evaluation run, not a production fleet.

**4. Groq-specific inference.** Groq's engine may have different variance characteristics than OpenAI's. We chose Groq because it is free and accessible. The determinism claim generalizes regardless of provider: LLM non-determinism is inherent to the inference (GPU floating-point non-associativity, sampling), not specific to one vendor.

**5. Different score scales.** spanIQ scores are embedding similarities (0–1); LLM-judge scores are quality ratings (0–1 or 1–10 normalized). The comparison is about *variance*, not absolute scores. A deterministic metric on a different scale is still deterministic. Do not read the spaniq mean and a judge mean as measuring the same thing.

**6. Langfuse runner is a local replica.** The `langfuse` runner replicates Langfuse's LLM-as-a-Judge methodology locally. It does not call Langfuse servers or use the Langfuse SDK. The prompt template and scoring match what a Langfuse user configures. We benchmark the evaluation methodology, not the platform.

**7. What spanIQ cannot do that LLM-judges can.** LLM-as-judge tools assess subjective quality — helpfulness, tone, reasoning correctness — that a statistical metric cannot. spanIQ's determinism comes from measuring distributional and embedding properties, not semantic understanding. If your question is "was this answer helpful?", you need an LLM judge. If your question is "has this output drifted from baseline?", spanIQ gives you a deterministic, free answer. They are complementary, not interchangeable.

## 11. Version history

| Date | Change |
|---|---|
| 2026-06-25 | Initial benchmark: 5 tools, 20/8/8-item datasets, Groq `llama-3.3-70b-versatile` judge. spaniq rows live; competitor rows pending a keyed run. |
| 2026-07-10 | Reconciled §8 with the actual committed 3-run results. The live run showed the Groq judge (and deepeval on 2/3 datasets) are also deterministic, so the "judges produce different scores every run" headline is not supported. Determinism benchmark marked as being superseded by an accuracy benchmark; the falsified tweet card was removed from the repo. |

This methodology document is versioned alongside the code. If the benchmark changes — new tools, new datasets, a new judge model — this document is updated in the same commit.

## Appendix — pricing reference

The cost column uses one rate, defined in `benchmarks/runners/_cost.py` and reproduced here:

```
Benchmark judge (llama-3.3-70b-versatile via Groq free tier):
  Actual cost: $0.00

Production-equivalent estimate (used for the cost column):
  $0.27 per 1M tokens, applied symmetrically to input and output
  per_trace_cost   = (prompt_tokens + completion_tokens) * 0.27 / 1_000_000
  cost_per_100     = per_trace_cost * 100
```

We do not hardcode a third-party provider's price into the benchmark output, because those prices drift and would silently make the numbers wrong. The flat rate is a stable, reproducible proxy. For orientation only, hosted small-judge models at the time of writing are in the same order of magnitude (e.g. GPT-4o-mini around $0.15/1M input and $0.60/1M output) — but verify current pricing at the provider before quoting a real production figure.
