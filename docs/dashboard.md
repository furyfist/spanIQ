# Dashboard Guide

spanIQ V4 ships a local Streamlit dashboard — no account, no server, no cloud.
It reads directly from your `spaniq.db` SQLite file.

---

## Launch

```bash
pip install "spaniq[dashboard]"
spaniq dashboard
# Opens http://localhost:8501
```

Custom DB path or port:
```bash
spaniq dashboard --db /path/to/spaniq.db --port 8502
```

> **First launch note:** Streamlit downloads frontend assets on first run (~30s).
> Subsequent launches are instant.

---

## Pages

### Overview

The landing page. Shows at a glance:

- **Total traces ingested** — count from the timeline store
- **Components monitored** — distinct pipeline components with data
- **Active alerts** — total alerts in the alerts table
- **Per-component health badges** — 🟢 healthy / 🟡 mild drift / 🔴 significant drift
- **Ingestion sparkline** — recent trace activity

### Drift Timeline

Interactive time-series charts for each metric, per component.

- **Component selector** — filter to one component or view all
- **Metric selector** — ResponseDrift, SemanticSimilarity, OutputStability, Consistency
- **Last N traces slider** — control how much history to show (20-1000)
- **Threshold lines** — dashed lines show where alerts fire
- **Auto-refresh toggle** — re-queries SQLite every 10 seconds when monitoring is live

### Attribution

Runs V3's changepoint/cascade analysis interactively.

- Set **Traces to analyse**, **PELT penalty**, and **Warmup traces**
- Click **Run Attribution**
- See the cascade timeline chart: 🔴 root cause, 🟡 cascade, 🟢 healthy
- Component details table shows break index and lead gap

### Alert Log

Historical table of all fired alerts.

- Filter by component, metric, and max rows
- All columns sortable in the Streamlit dataframe
- **Export CSV** button — download the full alert history

---

## Configuration

`DashboardConfig` in `src/spaniq/dashboard/config.py`:

```python
@dataclass
class DashboardConfig:
    db_path: str = "spaniq.db"        # SQLite file location
    refresh_interval: int = 10         # seconds between auto-refreshes
    page_title: str = "spanIQ Dashboard"
    max_rows: int = 500                # max rows per query
```

---

## Architecture

The dashboard is **read-only**. It uses the same `TimelineStore` API as the V2/V3 CLI.
No new storage, no new API layer, no ports.

```
Streamlit app (browser)
    ↓ reads
TimelineStore (SQLite) ← written by Monitor / OTelCollector
```

This means the dashboard works even if monitoring is stopped — it's a view of historical data.

---

## Running alongside monitoring

```bash
# Terminal 1: collect and monitor
spaniq collect-otel --baseline my_baseline

# Terminal 2: view live data
spaniq dashboard --db spaniq.db
```

Enable **Auto-refresh** in the sidebar to see live updates as traces arrive.
