# BENCHMARK_METHODOLOGY.md — Plan for Claude Code

> **Goal:** Create a `BENCHMARK_METHODOLOGY.md` file in the spanIQ repo root that documents exactly how the benchmark was run, what it measures, what it doesn't measure, and how anyone can reproduce it. This is the trust document that stops a skeptical HN commenter from dismissing the results.

---

## Why this file exists

Without a methodology doc, the benchmark table in the README is just numbers. With it, the numbers become evidence. The doc serves three audiences:

1. **The skeptical developer** who sees the tweet and thinks "yeah but how did you configure the competitors"
2. **The reproducer** who wants to `git clone` and verify the numbers themselves
3. **The reviewer** (YC partner, potential employer, open-source contributor) who evaluates whether the claims are honest

---

## What Claude Code should produce

A single file: `BENCHMARK_METHODOLOGY.md` in the repo root (NOT inside `benchmarks/` or `docs/` -- it needs to be visible at the top level alongside README.md).

---

## Document Structure

### Section 1: What This Benchmark Measures (and What It Doesn't)

This is the most important section. Lead with the narrow claim and the explicit non-claims.

**What it measures:**
- **Score variance** across N identical runs on the same dataset. "If you run the same eval 5 times, do you get the same number?" This is a measurement-instrument stability test, not an accuracy test.
- **Execution time** per evaluation run (wall-clock, `time.perf_counter()`)
- **Benchmark cost** with Groq (free tier, $0.00 for all tools)
- **Estimated production cost** extrapolated to GPT-4o-mini pricing ($0.15/1M input, $0.60/1M output) based on actual token counts from the Groq runs

**What it does NOT measure (state explicitly):**
- Which tool catches more bugs, hallucinations, or quality issues -- that's a different benchmark requiring human-labeled ground truth
- Accuracy correlation with human judgment -- all LLM-judge tools would need human annotation baselines
- Performance at scale (>1000 traces) -- this benchmark uses small datasets by design for reproducibility
- Optimal configuration of any competitor -- all tools run on default settings

This framing is critical. Write it in plain language, not legalese. The tone should be "here's what the numbers prove and what they don't, we're being upfront about the boundary."

### Section 2: The Claim

State the claim in one sentence, exactly as it will appear in marketing:

> **spanIQ produces identical scores on identical inputs, every time, at $0 cost. LLM-as-judge tools produce different scores on every run.**

Then: "This benchmark exists to let you verify that claim yourself."

### Section 3: Tools Benchmarked

A table with pinned versions for all 5 tools:

| Tool | Version | What it evaluates | Judge model | How it scores |
|---|---|---|---|---|
| **spanIQ** | 0.4.0 | Statistical metrics (PSI, cosine similarity, JS divergence, KS test) | None (deterministic) | Distributional distance, 0 = no drift |
| **Groq LLM-judge** | groq SDK [pin version] | Custom LLM-as-judge prompt | llama-3.3-70b-versatile via Groq | 1-10 scale, normalized to 0-1 |
| **deepeval** | [pin version] | G-Eval faithfulness | llama-3.3-70b-versatile via Groq | 0-1 (G-Eval score) |
| **ragas** | [pin version, >=0.4.0] | Faithfulness (claims vs context) | llama-3.3-70b-versatile via Groq | 0-1 (supported claims / total claims) |
| **Langfuse-style** | N/A (custom) | LLM-as-judge prompt (replicates Langfuse methodology) | llama-3.3-70b-versatile via Groq | 1-10 scale, normalized to 0-1 |

**IMPORTANT instructions for Claude Code:**
- Read the actual `pyproject.toml` and `pip freeze` output to fill in exact pinned versions
- Read each runner file to confirm the judge model string and scoring method
- The Langfuse runner needs a footnote: "This runner replicates the LLM-as-a-Judge evaluation methodology that Langfuse provides in its platform. It does not call Langfuse servers. The prompt template, scoring rubric, and LLM judge call are identical to what a Langfuse user configures in their UI. We benchmark the evaluation methodology, not the platform integration overhead."

### Section 4: Datasets

Describe each dataset with field names, sample counts, and source:

**qa_factual.jsonl**
- [N] factual question-answer pairs
- Fields: `input` (question), `output` (LLM answer), `reference_output` (ground truth)
- Used by: spaniq, groq, deepeval, langfuse runners
- Source: synthetic, committed to repo

**summarization.jsonl**
- [N] summarization tasks
- Fields: [read the file and list actual fields]
- Used by: [which runners]
- Source: synthetic, committed to repo

**rag_retrieval.jsonl**
- [N] RAG scenarios with retrieved context
- Fields: `input`, `output`, `context`, `reference_output` [verify actual field names]
- Used by: ragas runner (requires `context` for faithfulness metric)
- Source: synthetic, committed to repo

**IMPORTANT:** Claude Code must read each JSONL file, count the actual records, and list the actual field names. Do not guess.

### Section 5: Benchmark Configuration

Pin everything:

```
Python version: [read from CI or pyproject.toml]
OS: [state what the benchmark was run on]
Judge model: llama-3.3-70b-versatile (via Groq API, OpenAI-compatible endpoint)
Judge temperature: 0.0 (set explicitly in all LLM-judge runners)
Runs per tool: 5 (each tool evaluates the full dataset 5 times)
Parallelism: [read from runners -- asyncio.gather for langfuse, sequential for others?]
Groq API tier: free
```

### Section 6: Metrics Definitions

Define each metric precisely so there's no ambiguity about what the numbers mean:

**Score variance (the primary metric):**
- For each tool, compute mean score per run across all dataset items
- Collect the N per-run means into a list
- Variance = standard deviation of that list
- spanIQ's variance should be exactly 0.0000 (deterministic)
- LLM-judge tools will show non-zero variance

Write the formula explicitly:

```
Given N runs, each producing a mean score μ_i:
  Mean of means: M = (1/N) Σ μ_i
  Std Dev: σ = sqrt((1/N) Σ (μ_i - M)²)
```

**Execution time:**
- Wall-clock time per full dataset evaluation, measured with `time.perf_counter()`
- Reported as mean across N runs

**Benchmark cost (Groq):**
- All tools: $0.00 (Groq free tier)
- This is the actual cost incurred during the benchmark

**Estimated production cost (GPT-4o-mini):**
- Extrapolated from token counts observed during Groq runs
- Applied to GPT-4o-mini published pricing: $0.15/1M input tokens, $0.60/1M output tokens (source: openai.com/api/pricing, verified June 2026)
- spanIQ: $0.00 (no LLM calls, ever)
- For each LLM-judge tool: estimate = (input_tokens × $0.15/1M) + (output_tokens × $0.60/1M)
- Per-trace cost = total cost / number of dataset items
- Note: "These are estimates based on token counts from our Groq runs extrapolated to published OpenAI pricing. Actual costs may vary based on prompt caching, batch API discounts, and token counting differences between providers."

**How to compute the production cost estimate from Groq runs:**
The runners should already track token usage from Groq responses (the `usage` field in the completion response). If they don't, Claude Code should add token tracking to each LLM-judge runner:

```python
# After each LLM call, accumulate:
total_input_tokens += response.usage.prompt_tokens
total_output_tokens += response.usage.completion_tokens

# At end of run:
estimated_gpt4o_mini_cost = (
    (total_input_tokens / 1_000_000) * 0.15 +
    (total_output_tokens / 1_000_000) * 0.60
)
```

If the existing runners don't track tokens, Claude Code MUST add this tracking before generating the methodology doc. This is what makes the "production cost" column defensible without actually spending money on OpenAI.

### Section 7: How to Reproduce

Step-by-step, copy-paste-able:

```bash
# 1. Clone the repo
git clone https://github.com/[username]/spaniq.git
cd spaniq

# 2. Install with benchmark dependencies
pip install -e ".[benchmark]"

# 3. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your-key-here"

# 4. Run the full benchmark (all 5 tools, 5 runs each)
spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 5

# 5. Run spanIQ only (no API key needed)
spaniq benchmark --tool spaniq --runs 5

# 6. View results
cat benchmarks/results/summary.md
```

Add notes:
- "The spaniq-only benchmark requires no API key and completes in seconds. This is the fastest way to verify the variance=0.0 claim."
- "Competitor runners gracefully skip if their dependencies or API keys are missing. You will see 'skipped' in the results table for unavailable tools."
- "Groq offers a free API tier at console.groq.com. No credit card required."

### Section 8: Results

**IMPORTANT:** Claude Code must run the actual benchmark before writing this section. The flow:

1. Run `spaniq benchmark --tool spaniq --runs 5` and capture output
2. If GROQ_API_KEY is available, run `spaniq benchmark --all --runs 5` and capture output
3. Paste the actual numbers into this section
4. If no GROQ_API_KEY, write this section with placeholder `[PENDING]` markers and a note: "Results will be populated after live benchmark run"

The results section should include:

**Table 1: Score Variance (the proof)**

```
| Tool          | Mean Score | Std Dev | Runs |
|---------------|-----------|---------|------|
| spaniq        | X.XXXX    | 0.0000  | 5    |
| groq          | X.XXXX    | X.XXXX  | 5    |
| deepeval      | X.XXXX    | X.XXXX  | 5    |
| ragas         | X.XXXX    | X.XXXX  | 5    |
| langfuse      | X.XXXX    | X.XXXX  | 5    |
```

**Table 2: Cost Comparison**

```
| Tool          | Benchmark Cost (Groq) | Est. Production Cost per 100 traces (GPT-4o-mini) |
|---------------|----------------------|--------------------------------------------------|
| spaniq        | $0.00                | $0.00 (no LLM calls)                             |
| groq          | $0.00                | $X.XX                                            |
| deepeval      | $0.00                | $X.XX                                            |
| ragas         | $0.00                | $X.XX                                            |
| langfuse      | $0.00                | $X.XX                                            |
```

**Table 3: Execution Time**

```
| Tool          | Mean Time (N items) | Per-trace |
|---------------|--------------------|-----------| 
| spaniq        | X.XXs              | X.XXXs    |
| groq          | X.XXs              | X.XXXs    |
| deepeval      | X.XXs              | X.XXXs    |
| ragas         | X.XXs              | X.XXXs    |
| langfuse      | X.XXs              | X.XXXs    |
```

### Section 9: Raw Run Logs

State that individual per-run scores for every tool are available in `benchmarks/results/` as CSV and JSON files. Link to the directory. This is the "show your work" section.

```
benchmarks/results/
├── spaniq_runs.csv        # per-item scores for each of 5 runs
├── groq_runs.csv
├── deepeval_runs.csv
├── ragas_runs.csv
├── langfuse_runs.csv
├── summary.json           # machine-readable summary
└── summary.md             # human-readable report
```

"Anyone can inspect the individual scores to verify our variance calculations. The CSV files contain one row per dataset item per run."

### Section 10: Known Limitations and Fairness Statement

This is what separates an honest benchmark from a vendor benchmark. State clearly:

1. **Default configurations:** All competitor tools were run with their default settings. A user who tunes ragas's prompt or deepeval's G-Eval parameters might get lower variance. "We benchmark the out-of-the-box experience, not the optimized experience."

2. **Same judge model:** All LLM-judge tools used the same model (llama-3.3-70b-versatile via Groq). In production, ragas defaults to GPT-4o and deepeval defaults to GPT-4o. Different judge models may produce different variance characteristics. "We control for the judge model to isolate the framework's contribution to variance."

3. **Small dataset:** The benchmark uses [N] items. At larger scales, LLM-judge variance may average out (law of large numbers on the mean), but per-item variance remains. "We benchmark at a scale typical of a developer's first evaluation run."

4. **Groq-specific behavior:** Groq's inference engine may produce different variance characteristics than OpenAI's. "We chose Groq because it's free and accessible. The variance claim generalizes to any LLM provider because the non-determinism is inherent to LLM inference (GPU floating-point non-associativity, sampling), not provider-specific."

5. **Different score scales:** spanIQ scores are distributional distances (PSI, cosine, JS, KS), while LLM-judge scores are quality ratings (0-1 or 1-10). "The variance comparison is the point, not the absolute scores. A deterministic metric with a different scale is still deterministic."

6. **Langfuse runner caveat:** "The 'langfuse' runner replicates Langfuse's LLM-as-a-Judge evaluation methodology locally. It does not call Langfuse servers or use the Langfuse SDK. The prompt template and scoring approach match what a Langfuse user would configure. We benchmark the evaluation methodology, not the platform."

7. **What spanIQ cannot do that LLM-judges can:** "LLM-as-judge tools can assess subjective quality (helpfulness, tone, reasoning correctness) that statistical metrics cannot. spanIQ's determinism comes from measuring distributional properties, not semantic understanding. If your evaluation requires 'was this answer helpful?', you need an LLM judge. If your evaluation requires 'has this output drifted from baseline?', spanIQ gives you a deterministic, free answer."

### Section 11: Version History

```
| Date | Change |
|------|--------|
| [date of first publish] | Initial benchmark: 5 tools, [N]-item datasets, Groq judge |
```

"This methodology document is versioned alongside the code. If we update the benchmark (new tools, new datasets, new configurations), the methodology doc is updated in the same commit."

---

## Checklist for Claude Code

### Before writing the file:
- [ ] Read every file in `benchmarks/runners/` to confirm tool names, versions, judge models, scoring methods
- [ ] Read every file in `benchmarks/datasets/` to count records and list field names
- [ ] Read `benchmarks/run_benchmark.py` to confirm CLI tool names match
- [ ] Read `benchmarks/analysis/report.py` to understand how variance/cost/time are computed
- [ ] Read `pyproject.toml` for pinned dependency versions
- [ ] Check if runners track token usage from Groq responses (the `usage` field)
- [ ] If runners don't track tokens, add token tracking to each LLM-judge runner FIRST, then write the methodology doc

### Writing the file:
- [ ] Create `BENCHMARK_METHODOLOGY.md` in the repo root
- [ ] Use actual field names, record counts, and version numbers from the code (no guessing)
- [ ] Include the production cost estimate methodology (token counts from Groq runs × GPT-4o-mini pricing)
- [ ] Include the reproduction steps (copy-paste-able bash commands)
- [ ] Include the fairness/limitations section (all 7 points)
- [ ] If benchmark results are available, include actual numbers; if not, use `[PENDING]` placeholders

### After writing:
- [ ] Verify the reproduction steps actually work: `spaniq benchmark --tool spaniq --runs 5`
- [ ] Verify the file renders correctly as GitHub markdown
- [ ] Add to git: `git add BENCHMARK_METHODOLOGY.md`
- [ ] Commit: `docs(benchmark): add benchmark methodology document`

---

## Pricing Reference (for production cost estimates)

These are the published prices Claude Code should use for the "estimated production cost" column. Do not web-search for these; they're pinned here:

```
GPT-4o-mini (as of June 2026):
  Input:  $0.15 per 1M tokens ($0.00000015 per token)
  Output: $0.60 per 1M tokens ($0.00000060 per token)
  Source: openai.com/api/pricing

GPT-4o (for reference, not used in estimates):
  Input:  $2.50 per 1M tokens
  Output: $10.00 per 1M tokens

Groq (llama-3.3-70b-versatile):
  Free tier: $0.00
  This is what the benchmark actually uses.
```

The production cost column formula:

```
per_trace_cost = (avg_input_tokens × 0.00000015) + (avg_output_tokens × 0.00000060)
cost_per_100_traces = per_trace_cost × 100
```

If the runners don't currently track `response.usage.prompt_tokens` and `response.usage.completion_tokens`, Claude Code should add this tracking. The token counts are available in every OpenAI-compatible API response (including Groq).