# Signals, drift, attribution, escalation

spanIQ 0.5 is built in four layers. Each one is usable on its own from Python.

## 1. Signals (`spaniq.signals`)

`trace_signals(trace)` returns one dictionary of numbers per unit: the final output (`"output"`) and every component span.

| group | signals | computed for |
|---|---|---|
| text | chars, words, sentences, words_per_sentence, lines, list_lines, headers, code_blocks, json_valid, refusal, hedges, questions, digits, uppercase, non_ascii, lexical_diversity, repetition | every unit |
| grounding | unsupported_numbers, unsupported_names, novel_words | outputs with a context |
| relevance | question_coverage | retrieval and tool spans |
| span | error, latency_ms | component spans |
| nli | entailment, unsupported_sentences, contradicted_sentences | with `nli=get_nli()` |

**Signals are descriptions, not verdicts.**
- `unsupported_numbers` and `unsupported_names` are the closest to a per-output hallucination score.
- `novel_words` also rises for honest paraphrases, so use it as a drift signal only ([BENCHMARK_DRIFT.md](../BENCHMARK_DRIFT.md)).

**Metrics built on the signals:**
- `GroundednessMetric` is the NLI version.
- `UnsupportedClaimsMetric` is the lexical version.

Both work in `evaluate()` and `Monitor` like the other metrics.

## 2. Drift (`spaniq.drift`)

`compare(baseline_rows, candidate_rows)` tests every signal present on both sides:

- a two-sample KS test, with Benjamini-Hochberg adjustment across signals
- a standardised mean difference
- a PSI that handles flags and counts as categories

A signal drifts when it is significant at `alpha`, and its effect is at least `min_effect` or its PSI is at least `min_psi`. The gate matters because with thousands of traces almost any difference is significant. Pass embeddings to add an MMD permutation test.

`WindowDriftDetector(baseline_rows, window=100, stride=25)` is the online version. `update()` returns a report on the trace where an incident starts, and the incident ends at the first clean check. `Monitor` and `PipelineMonitor` run one per unit and raise one `drift:<unit>` alert per incident.

`diff_traces(baseline, candidate)` is the offline version that `spaniq diff` uses.

## 3. Attribution (`spaniq.attribution.drift_attribution`)

When one component's window trips, the other components are re-tested, but only on the traces since the estimated change point. The change point comes from a single-split binary segmentation of the tripping component's strongest signal.

Without that cut, a weak upstream shift is diluted by the pre-change traces still in the window. The usual case is a retriever that returns wrong passages while the generation step visibly starts refusing. The blamed component is the most upstream one that shifted.

This is ordering, not causal proof; see [decisions/temporal-not-causal.md](decisions/temporal-not-causal.md). The PELT and CUSUM attribution over stored series still works, and with `spaniq attribute --signals` it also runs on every `signal.*` series. That method is described in [ATTRIBUTION.md](ATTRIBUTION.md).

## 4. Escalation (`spaniq.escalation`)

`drift_priorities()` ranks traces by how far they moved in the direction of each drifted signal (summed, capped z-scores). `select()` takes the top ones, optionally mixing in a random share so the judge also sees ordinary traffic. `escalate()` then sends the selected traces to a `Judge`:

- It stays within a `JudgeBudget`, a daily cap kept in SQLite and shared by every process using the same database.
- Every verdict is stored as the `judge` series.
- A judge call that keeps failing is skipped. It is never recorded as a placeholder score.
