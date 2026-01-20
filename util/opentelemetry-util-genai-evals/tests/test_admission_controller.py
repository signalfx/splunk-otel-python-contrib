from __future__ import annotations

from opentelemetry.util.genai.evals import admission_controller


def test_admission_controller_allows_when_disabled(monkeypatch):
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS", raising=False
    )
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST", raising=False
    )

    controller = admission_controller.EvaluationAdmissionController()

    allowed, reason = controller.allow(None)

    assert allowed is True
    assert reason is None


def test_admission_controller_rate_limits(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS", "1"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST", "1"
    )

    monotonic_values = iter([1000.0, 1000.0, 1000.0, 1001.0])
    monkeypatch.setattr(
        admission_controller.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    controller = admission_controller.EvaluationAdmissionController()

    allowed_first, reason_first = controller.allow(None)
    allowed_second, reason_second = controller.allow(None)
    allowed_third, reason_third = controller.allow(None)

    assert (allowed_first, reason_first) == (True, None)
    assert (allowed_second, reason_second) == (False, "rate_limited")
    assert (allowed_third, reason_third) == (True, None)
