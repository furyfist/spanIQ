# spanIQ Determinism Benchmark

Compares spanIQ against competitor evaluators on the same datasets, measuring
score variance across N identical runs. The headline claim: spanIQ's std dev is
exactly 0.0000 (deterministic, no LLM judge), while every LLM-as-judge tool
shows non-zero spread.

## Running

```bash
spaniq benchmark --tool spaniq --runs 5
spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 3
spaniq benchmark --setup    # download full HF datasets
```

Competitor runners need `GROQ_API_KEY`. Missing deps or key are skipped
gracefully (the CLI prints `skipping <tool>: ...` and continues).

## Runners

| Tool       | Method                                    | Dataset        | Deterministic |
|------------|-------------------------------------------|----------------|---------------|
| `spaniq`   | semantic similarity, no LLM               | any            | yes (0.0000)  |
| `groq`     | Groq LLM-as-judge, 1-10 scale             | qa_factual     | no            |
| `deepeval` | G-Eval correctness via Groq               | qa_factual     | no            |
| `ragas`    | v0.4 collections Faithfulness via Groq    | rag_retrieval  | no            |
| `langfuse` | Langfuse-style LLM-as-judge via Groq      | qa_factual     | no            |

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
