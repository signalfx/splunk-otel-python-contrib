from __future__ import annotations

from typing import Any, Tuple

from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext

from .types import GenAI


def extract_span_context(span: Span | Any | None) -> SpanContext | None:
    """Return a SpanContext from the given span-like object if available."""

    if span is None:
        return None

    context = getattr(span, "context", None)
    if context is not None and getattr(context, "trace_id", 0):
        return context

    get_ctx = getattr(span, "get_span_context", None)
    if callable(get_ctx):
        try:
            context = get_ctx()
        except Exception:  # pragma: no cover - defensive
            context = None
    if context is not None and getattr(context, "trace_id", 0):
        return context
    return None


def store_span_context(target: GenAI, context: SpanContext | None) -> None:
    """Persist span context metadata on the GenAI invocation."""

    if context is None:
        return
    if not getattr(context, "trace_id", 0):
        return

    target.span_context = context
    target.trace_id = getattr(context, "trace_id", None)
    target.span_id = getattr(context, "span_id", None)
    trace_flags = getattr(context, "trace_flags", None)
    if trace_flags is not None:
        try:
            target.trace_flags = int(trace_flags)
        except Exception:  # pragma: no cover - defensive
            target.trace_flags = None


def span_context_hex_ids(
    context: SpanContext | None,
) -> Tuple[str | None, str | None]:
    """Return hexadecimal trace/span identifiers."""

    if context is None or not getattr(context, "trace_id", 0):
        return None, None
    trace_id = f"{context.trace_id:032x}"
    span_id = f"{context.span_id:016x}"
    return trace_id, span_id


def build_otel_context(
    span: Span | None,
    context: SpanContext | None,
) -> trace.Context | None:
    """Return an OpenTelemetry Context carrying a span or span_context."""

    if span is not None:
        try:
            return trace.set_span_in_context(span)
        except Exception:  # pragma: no cover - defensive
            pass
    if context is not None:
        try:
            non_recording = NonRecordingSpan(context)
            return trace.set_span_in_context(non_recording)
        except Exception:  # pragma: no cover - defensive
            return None
    return None
