"""Opt-in debug logging utilities for GenAI telemetry types.

The debug facility is disabled by default and activated when either
`OTEL_GENAI_DEBUG` or `OTEL_INSTRUMENTATION_GENAI_DEBUG` environment variable
is set to a truthy value (case-insensitive one of: 1, true, yes, on, debug).

Usage pattern (internal):

    from opentelemetry.util.genai.debug import genai_debug_log
    genai_debug_log("handler.start_llm.begin", invocation)

The helper auto-formats an object representation including span context IDs
when available.

This module intentionally avoids heavy imports and large content dumps.
Message bodies are NOT logged; counts only.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .span_context import (
    extract_span_context,
    span_context_hex_ids,
    store_span_context,
)

try:  # Local import guarded for namespace package variations
    from opentelemetry.util.genai.types import GenAI  # type: ignore
except Exception:  # pragma: no cover - fallback for edge import errors
    GenAI = object  # type: ignore

_TRUTHY = {"1", "true", "yes", "on", "debug"}


def _read_enabled_flag() -> bool:
    for var in ("OTEL_GENAI_DEBUG", "OTEL_INSTRUMENTATION_GENAI_DEBUG"):
        raw = os.environ.get(var)
        if raw is None:
            continue
        if raw.strip().lower() in _TRUTHY:
            return True
    return False


_ENABLED = _read_enabled_flag()

_LOGGER = logging.getLogger("opentelemetry.util.genai.debug")
if _ENABLED and not _LOGGER.handlers:  # configure minimal handler if none
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(message)s")  # raw message only (no ts/prefix)
    handler.setFormatter(fmt)
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.DEBUG)


def is_enabled() -> bool:
    """Return whether GenAI debug logging is enabled."""

    return _ENABLED


def _hex_trace(span_context: Any) -> str | None:
    try:
        if span_context and getattr(span_context, "is_valid", False):
            return f"{span_context.trace_id:032x}"  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return None
    return None


def _hex_span(span_context: Any) -> str | None:
    try:
        if span_context and getattr(span_context, "is_valid", False):
            return f"{span_context.span_id:016x}"  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return None
    return None


def summarize_genai(obj: Any) -> str:
    """Return a short representation for a GenAI object.

    Avoids printing message content; focuses on identity and context.
    """

    if obj is None:
        return "<None>"
    cls_name = obj.__class__.__name__
    parts: list[str] = [cls_name]
    # Common identifiers
    run_id = getattr(obj, "run_id", None)
    if run_id is not None:
        parts.append(f"run_id={run_id}")
    parent_run_id = getattr(obj, "parent_run_id", None)
    if parent_run_id is not None:
        parts.append(f"parent_run_id={parent_run_id}")
    provider = getattr(obj, "provider", None)
    if provider:
        parts.append(f"provider={provider}")
    model = getattr(obj, "request_model", None) or getattr(obj, "model", None)
    if model:
        parts.append(f"model={model}")
    # Span context
    span = getattr(obj, "span", None)
    span_context = getattr(obj, "span_context", None)
    if span_context is None and span is not None:
        try:
            span_context = extract_span_context(span)
        except Exception:  # pragma: no cover
            span_context = None
        else:
            store_span_context(obj, span_context)
    trace_hex, span_hex = span_context_hex_ids(span_context)
    if not trace_hex:
        trace_val = getattr(obj, "trace_id", None)
        if isinstance(trace_val, int) and trace_val:
            trace_hex = f"{trace_val:032x}"
    if not span_hex:
        span_val = getattr(obj, "span_id", None)
        if isinstance(span_val, int) and span_val:
            span_hex = f"{span_val:016x}"
    if trace_hex:
        parts.append(f"trace_id={trace_hex}")
    if span_hex:
        parts.append(f"span_id={span_hex}")
    # Token counts if present
    for attr in ("input_tokens", "output_tokens"):
        val = getattr(obj, attr, None)
        if isinstance(val, (int, float)):
            parts.append(f"{attr}={val}")
    # Message counts when lists
    inp_msgs = getattr(obj, "input_messages", None)
    if isinstance(inp_msgs, list):
        parts.append(f"input_messages={len(inp_msgs)}")
    out_msgs = getattr(obj, "output_messages", None)
    if isinstance(out_msgs, list):
        parts.append(f"output_messages={len(out_msgs)}")
    return "<" + " ".join(parts) + ">"


def genai_debug_log(event: str, obj: Any = None, **info: Any) -> None:
    """Conditionally emit a single structured debug log line.

    Parameters
    ----------
    event : str
        Event key/path (e.g., 'handler.start_llm.begin').
    obj : GenAI | None
        Related GenAI object for context representation.
    **info : Any
        Additional arbitrary key-value pairs (only simple scalar reprs recommended).
    """

    if not _ENABLED:
        return
    fields: list[str] = ["GENAIDEBUG", f"event={event}"]
    if obj is not None:
        fields.append(f"class={obj.__class__.__name__}")
        # Include summary after key-value list for readability
    for k, v in info.items():
        if v is None:
            continue
        try:
            if isinstance(v, (list, tuple, set)):
                fields.append(f"{k}_count={len(v)}")
            else:
                fields.append(f"{k}={v}")
        except Exception:  # pragma: no cover
            continue
    if obj is not None:
        fields.append("repr=" + summarize_genai(obj))
    _LOGGER.debug(" ".join(fields))


__all__ = [
    "is_enabled",
    "genai_debug_log",
    "summarize_genai",
]
