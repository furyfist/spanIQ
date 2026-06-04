# Why spanIQ Exists

## The Cost Problem

Every team evaluating LLM outputs today uses LLM-as-judge. deepeval, ragas, and braintrust all call GPT-4o or Claude to score outputs. The math compounds fast:

| Scale | API calls | Cost per run | Daily CI |
|---|---|---|---|
| 100 test cases × 3 metrics | 900 | ~$1.50 | ~$45/month |
| 1,000 test cases × 3 metrics | 9,000 | ~$15 | ~$450/month |
| 10,000 test cases × 3 metrics | 90,000 | ~$150 | ~$4,500/month |

That is the cost of knowing whether your LLM is still working correctly.

## The Non-Determinism Problem

Same input, different score across runs. LLM judges are stochastic. This means:

- You cannot reliably detect regressions. Did the score drop because your prompt changed, or because the judge was noisy?
- You cannot gate CI/CD on eval results. A test that passes 70% of the time is not a test.
- You cannot compare scores across time. The distribution of judge scores shifts as the judge model is updated.

## The Insight

LLM output is a distribution. When your model is working correctly, its outputs cluster around a known region in embedding space. They have consistent vocabulary, length, and structure. When something breaks — a prompt change, a model update, a data shift — the distribution moves.

You do not need an LLM to detect a distribution shift. You need statistics.

PSI detects vocabulary drift. KS test detects distributional shape changes. JS divergence detects structural instability. Cosine similarity detects semantic drift. These are proven, fast, and completely deterministic.

## What spanIQ Does Not Replace

Statistical methods work for a specific class of evaluations. Be honest about the limits:

- **Subjective quality** — "is this response helpful?" requires judgment, not statistics
- **Factual accuracy** — PSI cannot tell you if a claim is true, only if it differs from baseline
- **Safety and toxicity** — requires semantic understanding beyond distributional analysis
- **Zero-shot evaluation** — if you have no baselines, there is nothing to compare against
- **Complex reasoning** — "did the model follow the chain of thought?" needs an LLM

spanIQ is for regression detection, consistency monitoring, and CI gating. For everything else, use an LLM judge — but only where you actually need one.
