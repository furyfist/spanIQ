# spanIQ diff in CI

`spaniq diff` compares two runs of the same pipeline and exits 1 when the second one drifted. In CI that means: run your eval inputs through the pipeline on the base branch and on the pull request, then diff.

## Trace format

One JSON object per line, the same format `spaniq pipeline run` and `FileCollector` read:

```json
{"trace_id": "q17", "input": "What causes auroras?", "output": "Charged solar particles ...",
 "metadata": {"context": "The Northern Lights are caused by ..."},
 "components": [
   {"name": "retrieval", "kind": "retrieval", "output": "The Northern Lights are caused by ..."},
   {"name": "generation", "kind": "chat", "output": "Charged solar particles ..."}
 ]}
```

- **`input` and `output`** are required.
- **`metadata.context`** turns on the grounding signals for the final output. It can be a string or a list of passages.
- **`components`** gives each span its own signals and lets the report blame one. A `retrieval` or `execute_tool` span is checked against the question. Other spans are checked against the retrieval and tool outputs before them.
- **When the final output just repeats a span's output**, it is reported once, under that span.

`spaniq collect-otel --store-only` writes this format from OTel GenAI spans.

## What the report says

- **Each component is reported as stable or drifted.** For each drifted component, the report lists the signals that moved, with:
  - the baseline and candidate means
  - the standardised shift `d`
  - the PSI
  - the false-discovery-adjusted p-value

  A signal counts as drifted only when it is both significant (`--alpha`, default 0.01) and large enough: `|d| >= --min-effect` (default 0.5) or `PSI >= --min-psi` (default 0.2).
- **The most upstream drifted component is the suggested place to look.** A diff has no time axis, so this rests on pipeline order alone.
- **"Review first" lists candidate traces in order of how far they moved** in the direction of the drift. These are the ones to read, or to send to a judge with `--judge groq`.

## Choosing sizes

- Both runs need at least 5 traces for a signal to be tested at all.
- With 100 traces per run, a shift of about half a standard deviation is reliably found.
- Fewer traces only find larger shifts.
- Use the same inputs on both sides when you can, so input mix does not show up as drift.

## Optional checks

- `--embeddings` adds an MMD test on the final output embeddings. It catches a change in meaning when length and wording statistics stay put. It loads the sentence-transformers model.
- `--nli` adds sentence-level entailment against the context. It loads a local NLI cross-encoder on first use.

## Workflow

See [examples/ci/spaniq-diff.yml](../examples/ci/spaniq-diff.yml). It writes the Markdown report to the job summary.
