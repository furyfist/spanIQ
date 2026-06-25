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
