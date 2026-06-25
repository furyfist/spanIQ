# Benchmark Methodology

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

## 2. The claim

> spanIQ produces identical scores on identical inputs, every time, at $0 cost. LLM-as-judge tools produce different scores on every run.

This benchmark exists to let you verify that claim yourself. The determinism half is provable from a single run with no API key (Section 7). The variance half requires running the LLM-judge tools against a real model.

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

The spaniq rows below are from a live run on the configuration in Section 5 (Python 3.12.7, Windows, 5 runs per dataset). The competitor rows require `GROQ_API_KEY`, which was not set at publish time, so they are marked `[PENDING]` and will be filled in from a live judge run.

### Table 1 — Score variance (the proof)

| Tool | Dataset | Mean Score | Std Dev | Runs |
|---|---|---|---|---|
| spaniq | qa_factual | 0.5890 | **0.0000** | 5 |
| spaniq | summarization | 1.0000 | **0.0000** | 5 |
| spaniq | rag_retrieval | 1.0000 | **0.0000** | 5 |
| groq | qa_factual | [PENDING] | [PENDING] | 5 |
| deepeval | qa_factual | [PENDING] | [PENDING] | 5 |
| ragas | rag_retrieval | [PENDING] | [PENDING] | 5 |
| langfuse | qa_factual | [PENDING] | [PENDING] | 5 |

spanIQ's std dev is `0.0000` on every dataset across 5 runs: identical scores on identical inputs. The pending rows are expected to show non-zero std dev, since the judge model samples non-deterministically even at `temperature=0.0`.

### Table 2 — Cost comparison

Benchmark cost is the money actually spent (Groq free tier: $0.00 for all). Estimated production cost applies the flat $0.27/1M token rate from Section 6 to the tokens each tool consumes, scaled to 100 traces.

| Tool | Benchmark Cost (Groq) | Est. Production Cost / 100 traces |
|---|---|---|
| spaniq | $0.00 | $0.00 (no LLM calls) |
| groq | $0.00 | [PENDING] |
| deepeval | $0.00 | [PENDING] |
| ragas | $0.00 | [PENDING] |
| langfuse | $0.00 | [PENDING] |

spanIQ's production cost is structurally $0.00: it never calls an LLM, so there are no tokens to bill. The pending estimates come straight from the token counts the runners record during a live judge run.

### Table 3 — Execution time

Mean wall-clock per full-dataset run, from the same live spaniq run. Per-trace is mean time divided by the dataset's item count (qa_factual = 20, summarization = 8, rag_retrieval = 8).

| Tool | Dataset | Mean Time | Per-trace |
|---|---|---|---|
| spaniq | qa_factual | 3.23s | 0.162s |
| spaniq | summarization | 0.12s | 0.015s |
| spaniq | rag_retrieval | 0.11s | 0.014s |
| groq | qa_factual | [PENDING] | [PENDING] |
| deepeval | qa_factual | [PENDING] | [PENDING] |
| ragas | rag_retrieval | [PENDING] | [PENDING] |
| langfuse | qa_factual | [PENDING] | [PENDING] |

The first spaniq dataset carries the one-time cost of loading the embedding model, which is why qa_factual is slower than the two datasets evaluated after it in the same process. The LLM-judge rows are network-bound and will be dominated by per-item API latency.

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
