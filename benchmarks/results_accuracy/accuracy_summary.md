# spanIQ Accuracy Benchmark

Positive class = **bad** (the failure we want to catch). Thresholds are chosen by F1 on a calibration fold; metrics below are on the held-out test fold. AUC and average precision are threshold-free.

## Table 1 — Accuracy (test fold)

| Tool | Dataset | Threshold | Precision | Recall | F1 | ROC-AUC | Avg Prec |
|------|---------|-----------|-----------|--------|----|---------|----------|
| spaniq | qa_factual | 0.6133 | 0.8056 | 0.9667 | **0.8788** | 0.8567 | 0.85 |
| groq-llm-judge | qa_factual | 0.55 | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| deepeval | qa_factual | 0.3667 | 1.0 | 0.8333 | **0.9091** | 0.9708 | 0.9784 |
| langfuse | qa_factual | 0.5 | 0.6122 | 1.0 | **0.7595** | 0.525 | 0.6455 |
| spaniq | summarization | 0.982 | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| groq-llm-judge | summarization | 0.5 | 0.5 | 1.0 | **0.6667** | 0.5 | 0.6917 |
| deepeval | summarization | 0.5 | 0.5 | 1.0 | **0.6667** | 0.5 | 0.6917 |
| langfuse | summarization | 0.5 | 0.5 | 1.0 | **0.6667** | 0.5 | 0.6917 |
| spaniq | rag_retrieval | 0.9973 | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| groq-llm-judge | rag_retrieval | 0.5833 | 0.5 | 1.0 | **0.6667** | 0.5 | 0.6917 |
| deepeval | rag_retrieval | 0.5 | 0.5 | 1.0 | **0.6667** | 0.5 | 0.6917 |
| langfuse | rag_retrieval | 0.5 | 0.5 | 1.0 | **0.6667** | 0.5 | 0.6917 |

## Table 2 — Determinism sidebar (std dev of per-run mean score)

Reported honestly: determinism is a property spanIQ has *by construction* (no LLM call), but LLM judges at `temperature=0` are often deterministic too.

| Tool | Dataset | Score Std Dev |
|------|---------|---------------|
| spaniq | qa_factual | 0.0 |
| groq-llm-judge | qa_factual | 0.0 |
| deepeval | qa_factual | 0.0681 |
| langfuse | qa_factual | 0.0024 |
| spaniq | summarization | 0.0 |
| groq-llm-judge | summarization | 0.0 |
| deepeval | summarization | 0.0 |
| langfuse | summarization | 0.0 |
| spaniq | rag_retrieval | 0.0 |
| groq-llm-judge | rag_retrieval | 0.0147 |
| deepeval | rag_retrieval | 0.0 |
| langfuse | rag_retrieval | 0.0 |

> Recall (bad) is the money metric: of all genuinely bad outputs, how many did the tool catch? See `predictions.csv` to rebuild any number.