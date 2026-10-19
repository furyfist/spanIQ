# Changelog

## 0.5.0 (unreleased): detect changes, blame components, escalate

spanIQ stops claiming to judge single outputs and instead detects when a pipeline changes, names the component, and points a judge at the traces that moved. See `BENCHMARK_DRIFT.md` for what is measured and what is still pending.

### Added

- `spaniq.signals`: reference-free text signals, grounding signals against the trace's own context (unsupported numbers and names, novel wording, question coverage of retrieval), and optional sentence-level NLI groundedness with a lazily loaded local cross-encoder (`SPANIQ_NLI_MODEL`).
- `GroundednessMetric` (NLI) and `UnsupportedClaimsMetric` (lexical), usable in `evaluate()`, `Monitor` and `--metrics`.
- `spaniq.drift`: window comparison with KS tests, Benjamini-Hochberg control, effect-size and PSI gating, embedding MMD, and `WindowDriftDetector` for online use.
- `spaniq diff BASELINE CANDIDATE`: per-component drift report, most upstream drifted component, traces to review first, Markdown output, exit code 1 on drift for CI. Example workflow in `examples/ci/`.
- `Monitor` and `PipelineMonitor` store every trace's signals as `signal.<name>` series and raise one `drift:<unit>` alert per incident (`--drift-window`, 0 turns it off).
- Drift attribution: when a component trips, the others are re-tested on the traces since the estimated change point and the most upstream one that shifted is blamed. `spaniq attribute --signals` runs change point detection on the stored signal series.
- `spaniq.escalation`: budgeted LLM-judge escalation (`JudgeBudget` shared through SQLite, `GroqJudge`, ranked selection with optional exploration). `spaniq diff --judge groq` and `spaniq monitor run --judge groq`.
- Benchmarks: `spaniq benchmark --suite drift` (injected pipeline changes, legacy vs window detector, live-model mode) and `--suite grounding` (RAGTruth, HaluEval, LLM-AggreFact loaders and the local labeled rows).
- `Trace.from_dict()`, shared by the file collector and the new commands.

## Correctness and honesty pass (unreleased, ships with 0.5.0)

### Benchmark

- **The summarization and RAG results published with v0.4.0 were invalid.** The LLM-judge rows were not judgments: under Groq rate limiting every failed call was recorded as a neutral `0.5`, so 47 of the 48 judge scores on those two sets were fallbacks. spanIQ's F1 1.0 was a label leak: every good row was the reference answer copied verbatim, so similarity scored it exactly 1.0. Both are fixed and the tables were re-run (see `BENCHMARK_ACCURACY.md`).
- Judge runners retry transient failures with backoff and raise instead of fabricating a score; a tool whose calls keep failing is skipped with a visible message. Permanent 4xx errors are not retried.
- Good summarization / RAG rows are hand-written paraphrases of the reference.
- Groq retired `llama-3.3-70b-versatile`; the judge model is now `openai/gpt-oss-120b`, overridable with `SPANIQ_JUDGE_MODEL`.

### Fixed

- Alerts are written to SQLite, so the dashboard Alert Log shows them; a failure streak raises one alert instead of one per trace (`repeat_every` re-alerts during long incidents). Importing the alerting module no longer rewraps `sys.stdout`.
- SQLite connections are closed, use WAL and a busy timeout; `BaselineStore.update` is atomic.
- OTel receiver: thread-safe trace assembly, real OTel trace IDs, ISO timestamps, OTLP/HTTP protobuf and gzip, 4xx on bad payloads, bind errors raised at start, localhost bind by default (`--host`), late child spans kept via a short root grace period, current `gen_ai.input/output.messages` attributes read.
- Langfuse collector reads every page and no longer repeats the boundary observation; message lists become text instead of a Python repr. File collector in tail mode waits for partial lines.
- `ConsistencyMetric` uses a one-sided KS test with a sample-size-aware critical value (the old fixed 0.05 failed nearly every output). `OutputStabilityMetric` grades how far each structural feature falls outside the baseline range instead of a near-binary JS value. Baseline embeddings are cached; the model loads once under concurrency and is configurable via `SPANIQ_EMBEDDING_MODEL`. `evaluate()` keeps duplicate test case objects.
- Attribution aligns component breaks by trace ID and filters by pipeline; online CUSUM is calibrated from observed scores. The unused rolling window (`window_size`, `--window`) is removed.
- CLI: `--store-only` stores traces as JSONL, `--baseline` is optional with it, `attribute --pipeline` and `pipeline status --name` are honoured, unknown `--metrics` names are rejected, `--db` works after the subcommand, and common failures print one-line errors instead of tracebacks.
- Dashboard Overview no longer sleeps and reruns forever; auto-refresh is a toggle.

### Packaging

- Added the Apache-2.0 `LICENSE` file.
- Generated demo outputs moved to `docs/demo_output` and excluded from the wheel; demos write to `./spaniq_demo_output`.
- `matplotlib` moved to the new `viz` extra. `opentelemetry-proto` added to the `otel` extra.
- CI tests Python 3.10, 3.11 and 3.12 and checks the wheel contents.
- `spaniq` exports `BaseMetric`, `Monitor`, `TimelineStore`, the collectors, `PipelineMonitor` and `attribute` (lazily).

## v0.4.0 — OTel collector, Streamlit dashboard, determinism benchmark

### New

- **OTelCollector** — embedded OTLP/gRPC + HTTP receiver that maps OTel GenAI semantic convention spans to spanIQ `Trace` objects. Accepts both GenAI semconv (auto-mapped) and generic `spaniq.*` attribute spans. `spaniq collect-otel --baseline <name>` starts the receiver.
- **Streamlit dashboard** — local dashboard with 4 pages: Overview (health badges, KPIs), Drift Timeline (interactive Plotly charts), Attribution (V3 cascade visualization), Alert Log (filterable table with CSV export). `spaniq dashboard` launches it.
- **Accuracy benchmark** — reproducible suite comparing spanIQ vs Groq LLM judge / deepeval / ragas / langfuse on precision / recall / F1 / AUC at catching bad outputs, with a calibrated threshold rule and a per-item audit trail. `spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 5 --metric accuracy`. The legacy determinism metric remains via `--metric variance`. See `BENCHMARK_ACCURACY.md`.
- **Labeled benchmark datasets** — good/bad JSONL datasets (QA, summarization, RAG retrieval) with a documented failure taxonomy, generated from immutable `*_seed.jsonl` sources by `build_labeled.py`; provenance in `datasets/LABELING.md`.
- **`alerts` SQLite table** — `TimelineStore` now creates and writes to an `alerts` table (in addition to JSONL). `AlertEngine` dual-writes when `db_path` is set. Dashboard reads from this table.
- **`record_alert` / `query_alerts` / `alert_count`** methods on `TimelineStore`.
- Optional dependency groups: `otel`, `dashboard`, `benchmark`, `all`.

### Changed

- `pyproject.toml` version bumped to `0.4.0`.
- CLI extended with `collect-otel`, `dashboard`, and `benchmark` subcommands.
- `AlertEngine` gains optional `db_path` field for SQLite alert persistence.

### Tests added

- `test_otel_genai_mapping.py` (8 tests)
- `test_otel_generic_mapping.py` (6 tests)
- `test_otel_collector.py` (6 tests)
- `test_dashboard_overview.py` (4 tests)
- `test_dashboard_drift.py` (4 tests)
- `test_dashboard_attribution.py` (4 tests)
- `test_dashboard_alerts.py` (4 tests)
- `test_benchmark_runners.py` (5 tests)
- `test_e2e_otel_to_dashboard.py` (1 test)
- `test_e2e_benchmark.py` (2 tests)

## v0.3.0 — Per-component attribution (PELT + CUSUM cascade)

Root-cause attribution for multi-component pipelines.

## v0.2.0 — Production monitoring

Stream-based monitoring with baseline comparison, CUSUM alerts, and SQLite timeline.

## v0.1.0 — Deterministic evaluation

Core metrics: SemanticSimilarity, OutputStability, Consistency, ResponseDrift.
