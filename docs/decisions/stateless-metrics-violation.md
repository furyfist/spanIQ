# Decision: Metrics must be stateless — no internal window

Removed `self._window` from `ResponseDriftMetric` and `OutputStabilityMetric`. Both metrics now score directly against `test_case.baseline_outputs` and `test_case.actual_output`, with no mutable state between calls.

**The bug:** both metrics had grown an internal `deque` to maintain a rolling window of recent outputs. This violated the V1 `BaseMetric` contract, which treats `measure()` as a pure function: same inputs → same output, nothing stored between calls. The violation went unnoticed until V3, when `PipelineMonitor` began calling the same shared metric instances for multiple components. Because the single `self._window` deque accumulated outputs from all components interleaved, `search_tool` (a healthy control with a constant output string) was being scored against a window that contained retrieval and generation outputs — causing it to register as drifted even though it had never changed.

**What we chose:** remove `self._window` entirely from both metrics. `PipelineMonitor` already maintains a per-component `warmup_outputs` baseline and processes one output at a time — it is the correct and only owner of rolling-window state. Metrics receive the baseline corpus and the current output through `LLMTestCase` and return a score. No memory.

**What we rejected:** keeping the internal window and instantiating fresh metric objects per component on each `PipelineMonitor` init. This would have fixed the contamination bug but preserved a hidden contract violation — any caller that passed in shared metric instances (the documented API) would silently get the wrong behavior again.

**What would make us change this:** if a metric genuinely needs multi-output context that cannot be supplied through `baseline_outputs` (e.g. a stateful anomaly detector with a Kalman filter). In that case, the metric should be redesigned as an explicit `StatefulMetric` subclass with a documented component-binding requirement, not a hidden instance field on the base class.

**Commit:** `ba3589a` — fix(metrics): make ResponseDriftMetric and OutputStabilityMetric stateless — remove shared _window
