from __future__ import annotations

from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode


def _enable_capture(monkeypatch, mode: str | None = None) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
    )
    if mode is not None:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", mode
        )
    else:
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
    """Default capture mode when enabled is SPAN_AND_EVENT."""
    _enable_capture(monkeypatch, None)
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.SPAN_AND_EVENT


def test_none_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture mode to NONE yields NO_CONTENT."""
    _enable_capture(monkeypatch, "NONE")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.NO_CONTENT


def test_invalid_mode_defaults_to_span_and_event(monkeypatch):  # type: ignore[no-untyped-def]
    """Invalid capture mode falls back to SPAN_AND_EVENT."""
    _enable_capture(monkeypatch, "garbage-value")
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
