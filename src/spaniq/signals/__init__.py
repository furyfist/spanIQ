"""Per-output signals recorded for every trace and compared across runs by spaniq.drift."""

from spaniq.signals.extract import (
    OUTPUT_UNIT,
    context_for,
    extract,
    output_repeats_span,
    trace_signals,
)
from spaniq.signals.grounding import (
    GROUNDING_SIGNALS,
    RELEVANCE_SIGNALS,
    grounding_signals,
    relevance_signals,
)
from spaniq.signals.nli import NLI_SIGNALS, CrossEncoderNLI, check_groundedness, get_nli
from spaniq.signals.text import TEXT_SIGNALS, split_sentences, text_signals

__all__ = [
    "CrossEncoderNLI",
    "GROUNDING_SIGNALS",
    "NLI_SIGNALS",
    "OUTPUT_UNIT",
    "RELEVANCE_SIGNALS",
    "TEXT_SIGNALS",
    "check_groundedness",
    "context_for",
    "extract",
    "get_nli",
    "grounding_signals",
    "output_repeats_span",
    "relevance_signals",
    "split_sentences",
    "text_signals",
    "trace_signals",
]
