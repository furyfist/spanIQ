"""Drift benchmark: does a detector notice a pipeline change, how fast, and does it blame the right component.

Streams of retrieval -> generation traces are replayed through a detector. The
first part of every stream is unchanged, then a scenario (retriever degraded,
model swap, prompt change, ...) is switched on. The control scenario never
changes, so every alarm on it is a false alarm.
"""
