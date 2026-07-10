# spanIQ Benchmark

Compares spanIQ against competitor evaluators on the same labeled datasets. The
**primary metric is accuracy** — precision / recall / F1 / AUC at catching bad
outputs (positive class = `bad`). The legacy determinism metric (score variance
across identical runs) is still available via `--metric variance`.

For the full trust document — question, positive class, datasets, the fairness
threshold rule, reproduction steps, results, and the fairness statement — see
**[BENCHMARK_ACCURACY.md](../BENCHMARK_ACCURACY.md)** in the repo root. The
superseded determinism benchmark lives in
[BENCHMARK_METHODOLOGY.md](../BENCHMARK_METHODOLOGY.md). The contract both obey is
`docs/plans/benchmark_v2_accuracy.md`.

## Running

```bash
# accuracy (default) — spaniq-only needs no API key
spaniq benchmark --tool spaniq --runs 5 --metric accuracy
spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 3 --metric accuracy

# legacy determinism / variance
spaniq benchmark --tool spaniq --runs 5 --metric variance

spaniq benchmark --setup    # download full HF datasets
```

Datasets are labeled good/bad by `benchmarks/datasets/build_labeled.py` from the
immutable `*_seed.jsonl` sources; see `benchmarks/datasets/LABELING.md`.

Competitor runners need `GROQ_API_KEY`. Missing deps or key are skipped
gracefully (the CLI prints `skipping <tool>: ...` and continues).

The benchmark tests run without any API key — they assert the graceful-skip path,
the score parser, and the accuracy metrics / calibration / reporting on synthetic
predictions. Run them with:

```bash
python -m pytest tests/test_benchmark_runners.py tests/test_benchmark_metrics.py \
  tests/test_benchmark_calibrate.py tests/test_benchmark_report_accuracy.py -q
```

## Runners

Each runner exposes a `run_*_predictions` function returning a `LabeledResult`
(per-item predictions for accuracy) and a legacy `run_*_eval` (variance). The
good/bad label always comes from the dataset, never the tool.

| Tool       | Method                                    | Dataset        | Cost / determinism |
|------------|-------------------------------------------|----------------|--------------------|
| `spaniq`   | semantic similarity, no LLM               | any            | $0, det. by construction |
| `groq`     | Groq LLM-as-judge, 1-10 scale             | any            | paid, usually det. at temp 0 |
| `deepeval` | G-Eval correctness via Groq               | any            | paid, usually det. at temp 0 |
| `ragas`    | v0.4 collections Faithfulness via Groq    | rag_retrieval  | paid, usually det. at temp 0 |
| `langfuse` | Langfuse-style LLM-as-judge via Groq      | any            | paid, usually det. at temp 0 |

## Notes on the runners

**ragas** uses the v0.4 collections API (`SingleTurnSample`, `Faithfulness`
from `ragas.metrics.collections`), scoring each sample with
`single_turn_ascore`. Faithfulness needs retrieved context, so it runs on
`rag_retrieval.jsonl` only. Scores are 0-1 faithfulness, a different scale from
the 1-10-derived scores of the other judges — the variance comparison is the
point, not the absolute numbers.

**langfuse** replicates Langfuse's LLM-as-a-Judge *methodology*, not the
platform. Langfuse has no standalone local eval function — its judges run
server-side on ingested traces. This runner uses the same kind of judge prompt
template a Langfuse user configures, run against Groq locally. No Langfuse SDK
or server is required. The cost and variance characteristics match what a real
Langfuse LLM-judge user experiences.

## Cost

All competitor runners execute on the free Groq API, so real spend is $0. The
reported `cost_usd` is a token-based estimate of the production-equivalent cost
(at $0.27/1M tokens) so the table reflects the non-zero cost these tools incur
when run on a paid model in production.
