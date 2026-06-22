# Changelog

## v0.4.0 — OTel collector, Streamlit dashboard, determinism benchmark

### New

- **OTelCollector** — embedded OTLP/gRPC + HTTP receiver that maps OTel GenAI semantic convention spans to spanIQ `Trace` objects. Accepts both GenAI semconv (auto-mapped) and generic `spaniq.*` attribute spans. `spaniq collect-otel --baseline <name>` starts the receiver.
- **Streamlit dashboard** — local dashboard with 4 pages: Overview (health badges, KPIs), Drift Timeline (interactive Plotly charts), Attribution (V3 cascade visualization), Alert Log (filterable table with CSV export). `spaniq dashboard` launches it.
- **Determinism benchmark** — reproducible benchmark suite comparing spanIQ vs Groq LLM judge / deepeval / ragas on variance, cost, and speed. `spaniq benchmark --tool spaniq,groq --runs 5`.
- **Benchmark datasets** — 3 JSONL datasets committed to the repo (TriviaQA-style QA, summarization, RAG retrieval) with full `fetch.py` for downloading real HuggingFace data.
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
