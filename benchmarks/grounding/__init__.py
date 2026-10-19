"""Grounding benchmark: do spanIQ's grounding signals separate hallucinated outputs from faithful ones.

Runs on public labeled sets (RAGTruth, HaluEval, LLM-AggreFact) so the test
data is not ours, plus the small labeled RAG and summarization sets in
benchmarks/datasets. Reports ROC AUC and best-threshold F1 per scorer.
"""
