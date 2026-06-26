# spanIQ Determinism Benchmark Results

| Tool | Dataset | Mean Score | Std Dev | Cost ($) | Time (s) |
|------|---------|-----------|---------|---------|---------|
| spaniq | qa_factual | 0.589 | **0.0** | $0.0000 | 8.8s |
| groq-llm-judge | qa_factual | 1.0 | **0.0** | $0.0019 | 26.65s |
| deepeval | qa_factual | 0.6533 | **0.0309** | $0.0000 | 61.61s |
| langfuse | qa_factual | 0.5583 | **0.0312** | $0.0003 | 3.94s |
| spaniq | summarization | 1.0 | **0.0** | $0.0000 | 0.15s |
| groq-llm-judge | summarization | 1.0 | **0.0** | $0.0014 | 18.59s |
| deepeval | summarization | 1.0 | **0.0** | $0.0000 | 26.11s |
| langfuse | summarization | 0.625 | **0.0884** | $0.0005 | 3.34s |
| spaniq | rag_retrieval | 1.0 | **0.0** | $0.0000 | 0.14s |
| groq-llm-judge | rag_retrieval | 1.0 | **0.0** | $0.0011 | 19.48s |
| deepeval | rag_retrieval | 1.0 | **0.0** | $0.0000 | 26.76s |
| langfuse | rag_retrieval | 0.5833 | **0.0295** | $0.0002 | 3.34s |

> spanIQ std dev is 0.0000 — fully deterministic, no LLM judge needed.