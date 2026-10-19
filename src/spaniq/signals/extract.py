"""Turn a trace into signal dictionaries, one per unit (the final output and each component span)."""

from __future__ import annotations

from spaniq.attribution.component import ComponentKind
from spaniq.monitor.trace import Trace
from spaniq.signals.grounding import grounding_signals, relevance_signals
from spaniq.signals.nli import NLIModel, nli_signals
from spaniq.signals.text import text_signals

OUTPUT_UNIT = "output"
_SOURCE_KINDS = (ComponentKind.RETRIEVAL, ComponentKind.TOOL)


def context_for(trace: Trace) -> str | None:
    """The context the answer should be grounded in.

    metadata["context"] (or "retrieval_context") when the caller recorded it,
    otherwise the outputs of the trace's retrieval and tool spans.
    """
    meta = trace.metadata or {}
    ctx = meta.get("context") or meta.get("retrieval_context")
    if isinstance(ctx, (list, tuple)):
        ctx = "\n\n".join(str(c) for c in ctx)
    if ctx:
        return str(ctx)
    sources = [c.output for c in trace.components or [] if c.kind in _SOURCE_KINDS and c.output]
    return "\n\n".join(sources) or None


def extract(
    output: str, context: str | None = None, nli: NLIModel | None = None
) -> dict[str, float]:
    """Reference-free signals, plus grounding signals when a context is known.

    With an NLI model the sentence-level entailment signals are added as well.
    """
    signals = text_signals(output)
    if context:
        signals.update(grounding_signals(output, context))
        if nli is not None and output.strip():
            signals.update(nli_signals(output, context, nli))
    return signals


def trace_signals(trace: Trace, nli: NLIModel | None = None) -> dict[str, dict[str, float]]:
    """Signals for the final output and for every component span of one trace.

    Retrieval and tool spans are checked against the question (does what came
    back mention what was asked); other spans are checked against the outputs
    of the retrieval and tool spans before them.
    """
    units = {OUTPUT_UNIT: extract(trace.output, context_for(trace), nli)}
    upstream: list[str] = []
    for span in trace.components or []:
        if span.kind in _SOURCE_KINDS:
            signals = text_signals(span.output)
            if trace.input:
                signals.update(relevance_signals(trace.input, span.output))
            upstream.append(span.output)
        else:
            signals = extract(span.output, "\n\n".join(upstream) or context_for(trace), nli)
        signals["error"] = 1.0 if span.error else 0.0
        if span.latency_ms is not None:
            signals["latency_ms"] = float(span.latency_ms)
        units[span.name] = signals
    return units
