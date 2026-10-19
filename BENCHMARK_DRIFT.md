# Drift and Grounding Benchmarks

spanIQ 0.5 answers a different question from 0.4. It no longer asks "is this one output correct?" (an LLM judge does that better, see [BENCHMARK_ACCURACY.md](BENCHMARK_ACCURACY.md)). It asks: did the pipeline change, which component changed, and which traces should a judge look at. These two benchmarks test that claim. Both were set up with a go/no-go bar before the numbers existed, and this page reports where each one stands, including the parts that are not done.

## 1. Drift benchmark

**Question.** After a pipeline change, how many traces pass before an alarm? How often does it alarm when nothing changed? Does it blame the right component?

**Setup.** Streams of retrieval → generation traces. A detector sees 100 baseline traces, then 200 unchanged traces, then 300 traces with one scenario switched on. The control scenario never changes, so every alarm on it is a false alarm.

| scenario | what changes | cause |
|---|---|---|
| retriever_degraded | retriever returns other items' passages; the model refuses or invents facts | retrieval |
| context_truncated | context cut to 80 characters; answers get short and hedged | retrieval |
| partial_retriever | 30% of retrievals return an unrelated passage; answers unchanged | retrieval |
| prompt_format_change | answers become bullet lists | generation |
| model_swap | terse, hedged answers, sometimes with invented facts | generation |
| verbose_drift | long, repetitive answers | generation |
| fact_swap | half the answers keep their shape but get wrong numbers and names | generation |

**Detectors.**
- **legacy:** the 0.4 approach. Each component output is scored on its own against the baseline (OutputStabilityMetric), and an incident starts after 3 failures in a row.
- **window:** the 0.5 approach. Each component's [signals](src/spaniq/signals) are collected in a 100-trace window and checked against its baseline every 10 traces ([spaniq.drift](src/spaniq/drift/detector.py)). Blame goes to the most upstream component that shifted after the estimated change point ([drift_attribution](src/spaniq/attribution/drift_attribution.py)).

**Results.** Synthetic streams, 10 seeds. `python -m benchmarks.drift.run --runs 10`; raw rows in [benchmarks/results_drift](benchmarks/results_drift).

| scenario | legacy detected | legacy delay | legacy root | window detected | window delay | window root |
|---|---|---|---|---|---|---|
| control (false alarms / 1k) | 0.0 | | | 0.0 | | |
| retriever_degraded | 100% | 2 | 0% | 100% | 29 | 90% |
| context_truncated | 100% | 2 | 100% | 100% | 29 | 100% |
| partial_retriever | **0%** | n/a | n/a | 100% | 94 | **0%** |
| prompt_format_change | 100% | 2 | 100% | 100% | 29 | 100% |
| model_swap | 100% | 2 | 100% | 100% | 29 | 100% |
| verbose_drift | 100% | 2 | 100% | 100% | 29 | 100% |
| fact_swap | **0%** | n/a | n/a | 90% | 119 | 100% |

**Reading it.**
- **Legacy is faster when a change shows in the shape of the output.** It needs 2 traces where the window needs 29. Three failures in a row cost less evidence than a window test.
- **Legacy never sees changes to content alone.** Swapped facts and a partly broken retriever leave length and structure alone. The grounding signals catch both.
- **Legacy blames the wrong component when the retriever breaks.** It fires on whichever component trips first, and that is usually generation.
- **The window detector still cannot tell a partly broken retriever from a generation problem (0% correct root).** Only 30% of retrievals are wrong, and half the items are summaries whose question has no content words to check coverage against. So the retriever's shift is real but too small to confirm.

**Against a sampled judge.** The table below is arithmetic, not a run. It assumes a judge notices a changed output with probability p and that 3 flags are needed, the same as alert_after=3. Cost assumes $0.000165 per call ($0.27 per 1M tokens).

| judge samples | cost / 1k traces | traces to detect, p=0.3 | p=0.6 | p=0.9 |
|---|---|---|---|---|
| 1% | $0.0016 | 1000 | 500 | 333 |
| 5% | $0.0082 | 200 | 100 | 67 |
| 20% | $0.0329 | 50 | 25 | 17 |

The window detector needs 29 to 119 traces at no per-trace cost. A 5% judge needs about 67 to 200, and only if it notices the change at all. Judge calls stay useful for deciding whether a detected change is a problem, which is what escalation spends them on.

**Go/no-go.**
- **Detection: met, on synthetic streams only.** The bar was at least 4 of 5 changes within 200 traces and under 5% false alarms. The window detector caught all 7 changes in 69 of 70 runs within 200 traces (one fact_swap run was missed), with no false alarms in 5,000 control traces.
- **Root cause: met overall, missed on one scenario.** The bar was the right component first in at least 70% of detections. It was right in 58 of 69 (84%), and in none on partial_retriever.

**Caveats.** These streams are synthetic and we wrote the scenarios, so the results show that the mechanics work. They are not evidence about real traffic. The same scenarios run against a real model with `python -m benchmarks.drift.run --live` ([live.py](benchmarks/drift/live.py)); that needs `GROQ_API_KEY` and has not been run yet.

## 2. Grounding benchmark

**Question.** Do the grounding signals separate hallucinated outputs from faithful ones, using no reference answer and only the context the output was given?

**Scorers.**
- **unsupported_claims:** the share of numbers and names in the output that the context never mentions. Lexical, no model.
- **novel_words:** the share of content words not found in the context. It is reported to show why it is *not* a hallucination score.
- **nli:** 1 minus the mean sentence entailment from a local NLI cross-encoder (`cross-encoder/nli-deberta-v3-small` by default).
- **judge:** an LLM judge scoring support on a 1 to 10 scale, for comparison.

**Results so far.** Only the local labeled rows (`python -m benchmarks.grounding.run`):

| dataset | scorer | n | AUC |
|---|---|---|---|
| local (RAG + summarization rows) | unsupported_claims | 32 | 0.734 |
| local | novel_words | 32 | 0.131 |

**Reading it.**
- On the same rows where the 0.4 similarity metrics scored worse than chance (AUC 0.25 and 0.06), `unsupported_claims` scores 0.73, because the invented claims carry new numbers and names.
- `novel_words` is strongly inverted because faithful paraphrases use new words. It stays a drift signal, not a hallucination score.
- 32 rows is too few to conclude anything.

**Go/no-go: pending.** The bar is NLI AUC of at least 0.75 on RAGTruth, or else the NLI tier gets dropped and only the lexical signals stay. Not run yet: it needs the RAGTruth and HaluEval files and a one-time download of the NLI model. To run it:

```bash
git clone https://github.com/ParticleMedia/RAGTruth ../RAGTruth
git clone https://github.com/RUCAIBox/HaluEval ../HaluEval
spaniq benchmark --suite grounding --scorers unsupported_claims,nli,judge \
  --dataset ragtruth=../RAGTruth/dataset,halueval_qa=../HaluEval/data/qa_data.json,aggrefact --limit 2000
```
