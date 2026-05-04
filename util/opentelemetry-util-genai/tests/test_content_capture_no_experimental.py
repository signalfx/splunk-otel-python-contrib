from __future__ import annotations

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    should_emit_event,
)


def _enable_capture(monkeypatch, mode: str | None = None) -> None:
    if mode is None:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "SPAN_AND_EVENT",
        )
    else:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", mode
        )
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE",
        raising=False,
    )


def test_event_only_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture mode to EVENT_ONLY yields EVENT_ONLY."""
    _enable_capture(monkeypatch, "EVENT_ONLY")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.EVENT_ONLY


def test_span_only_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture mode to SPAN_ONLY yields SPAN_ONLY."""
    _enable_capture(monkeypatch, "SPAN_ONLY")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.SPAN_ONLY


def test_span_and_event_default(monkeypatch):  # type: ignore[no-untyped-def]
    """Explicit SPAN_AND_EVENT when that mode is the configured value."""
    _enable_capture(monkeypatch, "SPAN_AND_EVENT")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.SPAN_AND_EVENT


def test_none_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture mode to NONE yields NO_CONTENT."""
    _enable_capture(monkeypatch, "NONE")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.NO_CONTENT


def test_invalid_mode_defaults_to_span_and_event(monkeypatch):  # type: ignore[no-untyped-def]
    """Invalid value on legacy true+MODE path falls back to SPAN_AND_EVENT."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE",
        "garbage-value",
    )
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.SPAN_AND_EVENT


def test_disabled_flag(monkeypatch):  # type: ignore[no-untyped-def]
    """Falsey capture flag disables content."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE",
        "SPAN_ONLY",
    )
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.NO_CONTENT


def test_should_emit_event_defaults_from_capture_mode(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    monkeypatch.delenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, raising=False)
    assert should_emit_event() is False

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "EVENT_ONLY"
    )
    assert should_emit_event() is True


def test_should_emit_event_explicit_override(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, "true")
    assert should_emit_event() is True

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, "false")
    assert should_emit_event() is False
