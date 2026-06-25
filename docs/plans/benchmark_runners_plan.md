# spanIQ V4 — Benchmark Runners Plan: ragas + Langfuse

> **Goal:** Complete the two missing benchmark runners so the determinism benchmark covers all planned competitors. Pass this file to Claude Code for implementation.

## Context

spanIQ V4's benchmark suite already has working runners for:
- `spaniq_runner.py` — deterministic, variance = 0.0, cost = $0.00
- `groq_runner.py` — Groq-backed LLM-as-judge (smart deviation from the original promptfoo plan)
- `deepeval_runner.py` — G-Eval faithfulness + answer relevancy

Missing:
- `ragas_runner.py` — ragas faithfulness + context precision (the most expensive per-trace tool, strongest cost contrast)
- `langfuse_runner.py` — Langfuse LLM-as-judge evaluation

All runners implement the same interface and return the same `BenchmarkResult` dataclass. Read the existing runners before writing new ones to match field names, error handling, and skip logic exactly.

---

## Important: Read These Files First

Before writing any code, Claude Code MUST read:
1. The existing `BenchmarkResult` and `RunResult` dataclasses (likely in `benchmarks/config.py` or `benchmarks/__init__.py`)
2. `benchmarks/runners/spaniq_runner.py` — the reference implementation
3. `benchmarks/runners/groq_runner.py` — shows the LLM-judge pattern
4. `benchmarks/runners/deepeval_runner.py` — shows the competitor-framework pattern
5. `benchmarks/datasets/rag_retrieval.jsonl` — the ragas runner MUST use this dataset (it has context fields)
6. `benchmarks/analysis/report.py` — to verify the runner output format is consumed correctly
7. `benchmarks/run_benchmark.py` — the CLI entry point, to register the new runners

Match the existing patterns exactly. Do not invent new interfaces.

---

## Runner 1: ragas

### What ragas is

ragas (v0.4.3, latest as of Jan 2026) is a RAG evaluation framework. Its core metrics (faithfulness, context_precision, answer_relevancy) are all LLM-as-judge internally. It decomposes an LLM answer into atomic claims, checks each claim against retrieved context, and returns a 0-1 score. This makes it expensive (~$0.05-0.15 per trace on GPT-4o) and non-deterministic.

### API to use (v0.4 collections-based API, NOT the legacy API)

ragas v0.4 deprecated the old `from ragas.metrics import faithfulness` pattern and introduced a collections-based API. The plan MUST use the new API:

```python
# NEW API (v0.4+) — USE THIS
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate as ragas_evaluate

# Setup LLM (judge model)
client = AsyncOpenAI()  # or use Groq client
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric scorer
scorer = Faithfulness(llm=llm)

# Option A: Score a single sample
sample = SingleTurnSample(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL-NFL World Championship Game was an American football game played on January 15, 1967."
    ]
)
score = await scorer.single_turn_ascore(sample)  # returns float 0-1

# Option B: Evaluate a full dataset
dataset = EvaluationDataset(samples=[sample1, sample2, ...])
result = ragas_evaluate(dataset=dataset, metrics=[Faithfulness(llm=llm)])
# result is EvaluationResult with .scores, .to_pandas(), .total_cost()
```

### Key fields in SingleTurnSample

```python
SingleTurnSample(
    user_input="...",           # the question (REQUIRED)
    response="...",             # the LLM's answer (REQUIRED for faithfulness)
    retrieved_contexts=["..."], # list of context strings (REQUIRED for faithfulness)
    reference="...",            # ground truth answer (optional, used by context_recall)
)
```

### Implementation plan

**File:** `benchmarks/runners/ragas_runner.py`

**Step 1: Dependency check and graceful skip**

```python
def _has_ragas_deps() -> bool:
    """Check if ragas and Groq API key are available."""
    try:
        import ragas
        from ragas.metrics.collections import Faithfulness
    except ImportError:
        return False
    import os
    return bool(os.environ.get("GROQ_API_KEY"))
```

If deps missing, return `BenchmarkResult.skipped("ragas", reason="ragas not installed or no LLM API key")` — match exactly how deepeval_runner handles this.

**Step 2: Dataset loading**

Use `rag_retrieval.jsonl` specifically (NOT qa_factual.jsonl or summarization.jsonl). ragas's faithfulness metric requires `retrieved_contexts`, which only exists in the RAG dataset.

Load the JSONL and map fields to `SingleTurnSample`:

```python
def _load_ragas_dataset(dataset_path: str) -> list[SingleTurnSample]:
    """Load JSONL and convert to ragas SingleTurnSample objects."""
    from ragas.dataset_schema import SingleTurnSample
    import json

    samples = []
    with open(dataset_path) as f:
        for line in f:
            record = json.loads(line)
            samples.append(SingleTurnSample(
                user_input=record["input"],
                response=record["output"],
                retrieved_contexts=[record["context"]],  # wrap in list
                reference=record.get("reference_output", ""),
            ))
    return samples
```

IMPORTANT: Check the actual field names in `rag_retrieval.jsonl` before coding. The field names above (`input`, `output`, `context`, `reference_output`) are assumed from the V4 plan. Read the file first.

**Step 3: LLM provider setup (Groq only)**

Use Groq exclusively via its OpenAI-compatible endpoint. This keeps cost at $0 and stays consistent with the existing groq_runner:

```python
def _get_ragas_llm():
    """Get Groq LLM for ragas judge via OpenAI-compatible endpoint."""
    from ragas.llms import llm_factory
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    return llm_factory("llama-3.3-70b-versatile", client=client)
```

IMPORTANT: Verify that ragas `llm_factory` works with Groq's OpenAI-compatible client. If it errors, try passing the client differently (e.g. wrapping in `LangchainLLMWrapper` or using `ChatOpenAI` from langchain with the Groq base_url). Check the existing groq_runner.py for how it handles this. Test before committing.

**Step 4: The run_eval function**

```python
import asyncio
import time

def run_eval(dataset_path: str, n_runs: int = 5) -> BenchmarkResult:
    """Run ragas faithfulness on the dataset N times, measure variance, cost, and time."""
    if not _has_ragas_deps():
        return BenchmarkResult.skipped("ragas", reason="ragas not installed or no LLM API key")

    samples = _load_ragas_dataset(dataset_path)
    llm = _get_ragas_llm()

    from ragas.metrics.collections import Faithfulness
    from ragas.dataset_schema import EvaluationDataset
    from ragas import evaluate as ragas_evaluate

    results = []
    for run_idx in range(n_runs):
        dataset = EvaluationDataset(samples=samples)
        scorer = Faithfulness(llm=llm)

        start = time.perf_counter()
        # ragas evaluate is async internally, use asyncio.run if needed
        eval_result = ragas_evaluate(dataset=dataset, metrics=[scorer])
        elapsed = time.perf_counter() - start

        # Extract scores
        df = eval_result.to_pandas()
        scores = df["faithfulness"].tolist()
        mean_score = float(df["faithfulness"].mean())

        # Cost: $0.00 (Groq is free)
        cost = 0.0

        results.append(RunResult(
            scores=scores,
            mean_score=mean_score,
            time=elapsed,
            cost=cost,
        ))

    return BenchmarkResult(
        tool="ragas",
        variance=compute_variance(results),
        mean_time=mean([r.time for r in results]),
        total_cost=sum(r.cost for r in results),
        runs=results,
    )
```

IMPORTANT NOTES:
- `RunResult` and `BenchmarkResult` field names MUST match the existing dataclass exactly. Read the dataclass definition first.
- `compute_variance` and `mean` are helper functions that should already exist in `benchmarks/analysis/variance.py` or similar. Find and reuse them.
- ragas's `evaluate()` may run async internally. If it complains about event loops, wrap in `asyncio.run()` or use the sync wrapper if available.

**Step 5: Register in CLI**

Add `"ragas"` to the tool registry in `benchmarks/run_benchmark.py` so `spaniq benchmark --tool ragas` works.

**Step 6: Tests**

Add to `tests/test_benchmark_runners.py`:

```python
def test_ragas_runner_skips_without_deps(monkeypatch):
    """ragas runner should return skipped result when deps are missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = ragas_runner.run_eval("benchmarks/datasets/rag_retrieval.jsonl", n_runs=1)
    assert result.skipped is True  # or however skip is represented
```

**Step 7: Update pyproject.toml**

Add ragas to the benchmark optional dependency group:

```toml
[project.optional-dependencies]
benchmark = [
    "deepeval>=1.0.0",
    "ragas>=0.4.0",   # ADD THIS
]
```

---

## Runner 2: Langfuse

### The honest situation with Langfuse eval

After researching Langfuse's current SDK (v3+, OTel-based), here's the key finding: **Langfuse does NOT have a standalone local evaluation function you can call like `langfuse.evaluate(input, output)`**. Their evaluation model works differently:

1. **LLM-as-a-Judge evaluators** are configured in the Langfuse UI and run server-side on ingested traces
2. **Code evaluators** run custom Python/TypeScript in Langfuse's environment (not locally)
3. **Scores via SDK** let you push scores TO Langfuse, but you compute them yourself
4. **Experiments via SDK** let you run a task function + evaluator function locally, but the evaluator is your own code, not a Langfuse-provided one

This means a true "Langfuse eval runner" has two honest approaches:

### Approach A (RECOMMENDED): Langfuse-style LLM-as-judge prompt

Replicate what Langfuse's LLM-as-a-Judge does internally: take the same kind of evaluation prompt template that Langfuse users configure, run it against an LLM, parse the score. This is what a Langfuse user's eval workflow actually looks like in practice.

This is honest because:
- It's the same evaluation methodology Langfuse provides
- It uses the same kind of LLM judge call
- The cost and variance characteristics are identical to what a real Langfuse user experiences
- It doesn't require a running Langfuse server

### Approach B: Skip Langfuse entirely

If the team decides Langfuse isn't comparable enough (since it's really a tracing platform, not an eval library like deepeval/ragas), replace it with a different competitor or drop it from the benchmark. This is defensible.

### Implementation plan (Approach A)

**File:** `benchmarks/runners/langfuse_runner.py`

**Step 1: Dependency check and graceful skip**

This runner needs GROQ_API_KEY. No Langfuse SDK needed.

```python
def _has_langfuse_deps() -> bool:
    """Check if we have Groq API key for Langfuse-style evaluation."""
    import os
    return bool(os.environ.get("GROQ_API_KEY"))
```

**Step 2: The Langfuse-style evaluation prompt**

Langfuse's LLM-as-a-Judge uses a configurable prompt template. Here's a standard one (faithfulness/correctness style):

```python
LANGFUSE_JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response.

## Input
Question: {input}
Expected Answer: {expected}
Actual Answer: {output}

## Task
Rate the actual answer on a scale of 1-10 based on:
- Factual correctness compared to the expected answer
- Completeness of the response
- Relevance to the question

Respond with ONLY a JSON object:
{{"score": <number 1-10>, "reasoning": "<brief explanation>"}}
"""
```

**Step 3: The judge call**

```python
import json

async def _judge_single(client, model: str, input_text: str, output_text: str, expected: str) -> float:
    """Run a single Langfuse-style LLM-as-judge evaluation."""
    prompt = LANGFUSE_JUDGE_PROMPT.format(
        input=input_text,
        expected=expected,
        output=output_text,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # attempt determinism (won't fully work)
    )
    text = response.choices[0].message.content.strip()

    # Parse score from JSON response
    try:
        parsed = json.loads(text)
        return float(parsed["score"]) / 10.0  # normalize to 0-1
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: try to extract a number
        import re
        match = re.search(r'\d+', text)
        if match:
            return min(float(match.group()) / 10.0, 1.0)
        return 0.5  # default if parsing fails
```

**Step 4: The run_eval function**

```python
import asyncio
import time

def run_eval(dataset_path: str, n_runs: int = 5) -> BenchmarkResult:
    """Run Langfuse-style LLM-as-judge on dataset N times."""
    if not _has_langfuse_deps():
        return BenchmarkResult.skipped("langfuse", reason="No LLM API key for Langfuse-style judge")

    records = _load_jsonl(dataset_path)
    client, model, cost_per_call = _get_client_and_model()

    results = []
    for run_idx in range(n_runs):
        start = time.perf_counter()
        scores = asyncio.run(_judge_batch(client, model, records))
        elapsed = time.perf_counter() - start

        mean_score = sum(scores) / len(scores)
        cost = len(records) * cost_per_call

        results.append(RunResult(
            scores=scores,
            mean_score=mean_score,
            time=elapsed,
            cost=cost,
        ))

    return BenchmarkResult(
        tool="langfuse",
        variance=compute_variance(results),
        mean_time=mean([r.time for r in results]),
        total_cost=sum(r.cost for r in results),
        runs=results,
    )


async def _judge_batch(client, model: str, records: list) -> list[float]:
    """Judge all records concurrently."""
    tasks = [
        _judge_single(client, model, r["input"], r["output"], r.get("reference_output", ""))
        for r in records
    ]
    return await asyncio.gather(*tasks)


def _get_client_and_model():
    """Get async Groq client via OpenAI-compatible endpoint."""
    import os
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
    return client, "llama-3.3-70b-versatile", 0.0  # Groq is free
```

**Step 5: Dataset choice**

Unlike ragas, the Langfuse-style judge doesn't require `retrieved_contexts`. Use `qa_factual.jsonl` as the primary dataset (same as the Groq runner, for apples-to-apples comparison of LLM-judge variance).

**Step 6: Register in CLI**

Add `"langfuse"` to the tool registry in `benchmarks/run_benchmark.py`.

**Step 7: Tests**

```python
def test_langfuse_runner_skips_without_deps(monkeypatch):
    """Langfuse runner should return skipped result when no Groq API key."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = langfuse_runner.run_eval("benchmarks/datasets/qa_factual.jsonl", n_runs=1)
    assert result.skipped is True
```

**Step 8: Add a note in the benchmark README**

Document that the "Langfuse" runner replicates Langfuse's LLM-as-a-Judge evaluation pattern locally, not calling Langfuse servers. This is honest and defensible: "We replicate the evaluation methodology, not the platform integration."

---

## Checklist for Claude Code

### Before writing any code:
- [ ] Read `BenchmarkResult` and `RunResult` dataclass definitions
- [ ] Read `spaniq_runner.py` for the reference interface
- [ ] Read `groq_runner.py` for the LLM-judge pattern
- [ ] Read `deepeval_runner.py` for the competitor-framework pattern
- [ ] Read `rag_retrieval.jsonl` to verify field names
- [ ] Read `qa_factual.jsonl` to verify field names
- [ ] Read `run_benchmark.py` to see how runners are registered
- [ ] Read `analysis/report.py` to verify output format consumption

### ragas runner:
- [ ] Create `benchmarks/runners/ragas_runner.py`
- [ ] Use ragas v0.4 collections API (`SingleTurnSample`, `EvaluationDataset`, `Faithfulness` from `ragas.metrics.collections`)
- [ ] Use `rag_retrieval.jsonl` dataset (has context fields)
- [ ] Graceful skip when ragas not installed or no API key
- [ ] Match `BenchmarkResult`/`RunResult` field names exactly
- [ ] Register in `run_benchmark.py`
- [ ] Add skip test to `test_benchmark_runners.py`
- [ ] Add `ragas>=0.4.0` to `pyproject.toml` benchmark extras

### Langfuse runner:
- [ ] Create `benchmarks/runners/langfuse_runner.py`
- [ ] Use Langfuse-style LLM-as-judge prompt (NOT Langfuse SDK server calls)
- [ ] Use `qa_factual.jsonl` dataset
- [ ] Graceful skip when no API key
- [ ] Match `BenchmarkResult`/`RunResult` field names exactly
- [ ] Register in `run_benchmark.py`
- [ ] Add skip test to `test_benchmark_runners.py`
- [ ] Document the approach in benchmark README ("replicates Langfuse eval methodology")

### After both runners:
- [ ] Run full test suite: `python -m pytest tests/ -v` (all V1+V2+V3+V4 tests green)
- [ ] Run benchmark with spanIQ only: `spaniq benchmark --tool spaniq` (verify still works)
- [ ] If API keys available, run: `spaniq benchmark --tool spaniq,ragas,langfuse --runs 3`
- [ ] Verify the report table includes all 5 tools (spaniq, groq, deepeval, ragas, langfuse)
- [ ] Commit: `feat(benchmark): add ragas and langfuse benchmark runners`

---

## Dependency Versions (pinned)

```
ragas>=0.4.0,<1.0.0       # v0.4 collections API
openai>=1.0.0              # AsyncOpenAI client (Groq's OpenAI-compatible endpoint)
groq>=0.4.0                # Groq API (shared across all LLM-judge runners)
```

No Langfuse SDK dependency needed (the runner replicates the eval pattern locally).
No OpenAI API key needed (all runners use Groq).

---

## What success looks like

After these two runners, the benchmark table should show 5 rows:

```
| Tool     | Mean Score | Std Dev | Cost/100 | Time/100 |
|----------|-----------|---------|----------|----------|
| spaniq   | 7.31      | 0.0000  | $0.00    | 2.1s     |
| groq     | 7.42      | 0.83    | $0.00    | 12.4s    |
| deepeval | 7.18      | 0.91    | $0.00    | 18.7s    |
| ragas    | 0.82      | 0.07    | $0.00    | 24.1s    |
| langfuse | 7.55      | 0.79    | $0.00    | 14.2s    |
```

(Numbers are illustrative. All costs show $0.00 because all runners use Groq. The key proof: spanIQ's std dev is 0.0000, everyone else's is non-zero. The cost story for the README/blog should note: "all competitors were run on free Groq API for fairness, but in production these tools default to GPT-4o at $X/trace.")

Note: ragas scores are 0-1 (faithfulness), while others are on different scales. The variance comparison is the point, not the absolute scores. Document this scale difference in the benchmark output.