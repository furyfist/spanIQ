# spanIQ

[![CI](https://github.com/furyfist/spaniq/actions/workflows/ci.yml/badge.svg)](https://github.com/furyfist/spaniq/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Did your LLM pipeline change, which component changed, and which traces should a judge look at?**

spanIQ records cheap, deterministic signals for every trace and every pipeline component: length and structure, refusals, hedging, and whether the answer's numbers and names appear in its own retrieved context. It compares windows of those signals against a baseline. A model swap, prompt edit or broken retriever shifts them, even when no single output looks wrong. spanIQ then names the most upstream component that shifted, and can send the traces that moved most to an LLM judge within a daily budget.

It is not an LLM-judge replacement. For deciding whether one answer is right a judge is better ([BENCHMARK_ACCURACY.md](BENCHMARK_ACCURACY.md)). spanIQ runs on 100% of traffic for $0 and tells you where to spend judge calls.

---

## Catch regressions in CI

Replay the same eval inputs on `main` and on your branch, then:

```bash
pip install "spaniq @ git+https://github.com/furyfist/spanIQ"   # not on PyPI yet
spaniq diff baseline.jsonl candidate.jsonl --markdown report.md
```

```
drift in generation; most upstream drifted component: generation (200 baseline vs 300 candidate traces)
  retrieval: stable
generation: drifted
| words_per_sentence | 26.9 -> 7.92 (d=-5.06, PSI 13.72) | 1.2e-137 |
| hedges             | 0.00139 -> 0.038 (d=+0.99)        | 4.7e-18  |
...
review first
  t000542  generation.unsupported_numbers=1 (baseline mean 0); ...
```

It exits 1 when the candidate drifted, so a pull request fails. Traces are JSONL with `input` and `output`, plus optional `components` (retrieval, tool and generation spans) and `metadata.context`. Add `--judge groq --judge-budget 20` to have a judge score the traces listed for review. A ready workflow is in [examples/ci/spaniq-diff.yml](examples/ci/spaniq-diff.yml); details are in [docs/CI.md](docs/CI.md).

## Watch production

```bash
spaniq collect-otel --store-only                     # OTLP on localhost:4317 / 4318, writes otel_traces.jsonl
spaniq pipeline run --name rag --path otel_traces.jsonl
spaniq attribute --pipeline rag --signals            # change points on every stored signal series
spaniq dashboard
```

`spaniq pipeline run` uses each component's first `--warmup` traces as its baseline. It then compares a sliding window of each component's signals against that baseline. When one trips, every component is re-tested on the traces since the estimated change, and the most upstream one that shifted is blamed. `spaniq monitor run --judge groq` escalates the traces that moved furthest at the start of each incident.

---

## What it measures

| Signals | Needs | Examples | Catches |
|---|---|---|---|
| Reference-free | the output | length, sentences, list lines, JSON validity, refusals, hedges, repetition | format, style and behaviour changes |
| Grounding | the trace's own context | unsupported numbers and names, novel wording, question coverage of the retrieval | invented facts, wrong passages |
| NLI (optional) | context and a local cross-encoder | sentence entailment, contradiction | reworded false claims |
| Reference metrics | a baseline or reference answer | `SemanticSimilarityMetric`, `ResponseDriftMetric`, ... | offline eval against known answers |

Windows are compared with a KS test per signal, Benjamini-Hochberg across signals, and an effect-size or PSI gate, so that a large window does not turn a negligible wobble into an alarm. Embedding MMD is available for shifts in meaning. More in [docs/SIGNALS.md](docs/SIGNALS.md).

---

## Benchmarks

**Drift** ([BENCHMARK_DRIFT.md](BENCHMARK_DRIFT.md)). Seven injected pipeline changes, 10 seeds. These are synthetic streams written by us: they show the mechanics work, not real-traffic accuracy.

| | legacy (0.4 per-output streak) | window (0.5) |
|---|---|---|
| changes detected | 5 of 7 scenarios | 7 of 7 (69 of 70 runs) |
| median delay | 2 traces | 29 to 119 traces |
| false alarms (5,000 unchanged traces) | 0 | 0 |
| right component blamed | 80% of detections (40 of 50) | 84% of detections (58 of 69) |

Legacy is faster when the output's shape changes, but it is blind to wrong facts and partly broken retrievers. The window detector still cannot attribute a retriever that fails on only 30% of queries. The live-model run of the same scenarios is wired but not run yet.

**Grounding.** On the 32 local labeled RAG/summarization rows, where the 0.4 similarity metrics scored worse than chance (AUC 0.25 and 0.06), `unsupported_claims` scores AUC 0.73. RAGTruth, HaluEval and LLM-AggreFact runs, including the NLI tier, are pending.

**Accuracy** ([BENCHMARK_ACCURACY.md](BENCHMARK_ACCURACY.md)). At judging one QA answer, LLM judges score F1 1.000 and spanIQ's embedding similarity scores 0.879. That table also records the correction of an earlier, invalid claim.

---

## Architecture

```
 collectors              signals                   detection                  action
 ──────────              ───────                   ─────────                  ──────
 OTLP gRPC/HTTP ──┐      per trace and span:       window per unit vs         drift alert (JSONL + SQLite)
 JSONL file ──────┼──▶   text, grounding, NLI ──▶  baseline: KS + FDR +  ──▶  upstream attribution
 Langfuse, SDK ───┘      stored as signal.*        effect size, MMD           judge escalation (budgeted)
                                │                         │                    spaniq diff exit code
                                └──────── SQLite: timeline, alerts, baselines, judge budget ──▶ dashboard
```

---

## Install

```bash
# Not on PyPI yet. From GitHub (add extras in brackets):
pip install "spaniq @ git+https://github.com/furyfist/spanIQ"           # signals, drift, diff, monitoring
pip install "spaniq[otel] @ git+https://github.com/furyfist/spanIQ"     # OTel receiver
pip install "spaniq[dashboard] @ git+https://github.com/furyfist/spanIQ" # local dashboard
pip install "spaniq[groq] @ git+https://github.com/furyfist/spanIQ"     # Groq judge and baseline collection
pip install "spaniq[viz] @ git+https://github.com/furyfist/spanIQ"      # PNG chart export
pip install "spaniq[all] @ git+https://github.com/furyfist/spanIQ"      # everything

# Benchmarks need a source checkout (datasets are not in the wheel)
git clone https://github.com/furyfist/spanIQ && cd spanIQ && pip install -e ".[benchmark]"
```

The NLI model (`cross-encoder/nli-deberta-v3-small`, set `SPANIQ_NLI_MODEL` to change it) and the embedding model download on first use. Nothing needs them unless you pass `--nli`, `--embeddings` or use `GroundednessMetric` / the similarity metrics.

---

## CLI reference

```bash
# Compare two runs (CI)
spaniq diff baseline.jsonl candidate.jsonl [--embeddings] [--nli] [--json] [--markdown PATH] [--no-fail]
spaniq diff baseline.jsonl candidate.jsonl --judge groq --judge-budget 20

# Pipelines and attribution
spaniq pipeline run --name rag --path traces.jsonl [--warmup 20] [--drift-window 100]
spaniq pipeline status --name rag
spaniq attribute --pipeline rag --signals [--json] [--export chart.png]

# Single-output monitoring against a stored baseline
spaniq baseline collect --name my_baseline --prompt "..." --source file --path outputs.jsonl
spaniq monitor run --baseline my_baseline --path traces.jsonl [--judge groq --judge-budget 50]
spaniq collect-otel --baseline my_baseline        # or --store-only --traces-path otel_traces.jsonl

# Inspect
spaniq timeline summary --metric signal.refusal
spaniq dashboard --db spaniq.db

# Benchmarks (source checkout)
spaniq benchmark --suite drift [--detectors legacy,window] [--live]
spaniq benchmark --suite grounding --dataset local,ragtruth=DIR --scorers unsupported_claims,nli
spaniq benchmark --suite accuracy --tool spaniq,groq --runs 5
```

---

## Comparison

| | spanIQ | deepeval | ragas | Phoenix | Langfuse |
|---|---|---|---|---|---|
| Runs on every trace at $0 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Distribution drift with FDR control | ✅ | ❌ | ❌ | partial | ❌ |
| Blames a pipeline component | ✅ | ❌ | ❌ | ❌ | ❌ |
| CI regression gate on traces | ✅ | ✅ (judge-based) | ❌ | ❌ | ❌ |
| Budgeted judge escalation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Judges single answers well | via escalation | ✅ | ✅ | ✅ | ✅ |
| OTel ingestion | ✅ | ❌ | ❌ | ✅ | ✅ |

---

## Documentation

- [docs/CI.md](docs/CI.md): `spaniq diff` in a pull request workflow
- [docs/SIGNALS.md](docs/SIGNALS.md): signals, drift tests, attribution and escalation
- [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md): change point attribution over stored series
- [docs/otel-integration.md](docs/otel-integration.md): connect any OTel app
- [docs/dashboard.md](docs/dashboard.md): the Streamlit UI
- [docs/METRICS.md](docs/METRICS.md): the reference-based metrics

---

## License

Apache 2.0, see [`LICENSE`](LICENSE).
