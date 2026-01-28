from __future__ import annotations

import asyncio

from opentelemetry.util.genai.evals import admission_controller


def test_admission_controller_allows_when_disabled(monkeypatch):
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS", raising=False
    )
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST", raising=False
    )

    controller = admission_controller.EvaluationAdmissionController()

    allowed, reason = controller.allow()

    assert allowed is True
    assert reason is None


def test_admission_controller_rate_limits(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_ENABLE", "true"
    )
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

    allowed_first, reason_first = controller.allow()
    allowed_second, reason_second = controller.allow()
    allowed_third, reason_third = controller.allow()

    assert (allowed_first, reason_first) == (True, None)
    assert (
        allowed_second,
        reason_second,
    ) == (False, "client_evaluation_rate_limited")
    assert (allowed_third, reason_third) == (True, None)


def test_admission_controller_allow_async(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS", "1"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST", "1"
    )

    # Provide constant value for asyncio event loop + rate limiter calls
    monotonic_value = 2000.0
    monkeypatch.setattr(
        admission_controller.time,
        "monotonic",
        lambda: monotonic_value,
    )

    controller = admission_controller.EvaluationAdmissionController()

    allowed, reason = asyncio.run(controller.allow_async())

    assert (allowed, reason) == (True, None)


def test_admission_controller_allow_async_rate_limits(monkeypatch):
    """Test that async rate limiting works correctly."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_ENABLE", "true"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS", "1"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST", "1"
    )

    # Use a mutable counter to track progression
    state = {"time": 3000.0, "checks": 0}

    def mock_monotonic():
        # Advance time after second check
        if state["checks"] >= 2:
            return state["time"] + 1.5
        return state["time"]

    monkeypatch.setattr(
        admission_controller.time,
        "monotonic",
        mock_monotonic,
    )

    controller = admission_controller.EvaluationAdmissionController()

    async def run_checks():
        state["checks"] = 0
        allowed_first, reason_first = await controller.allow_async()

        state["checks"] = 1
        allowed_second, reason_second = await controller.allow_async()

        state["checks"] = 2
        allowed_third, reason_third = await controller.allow_async()

        return (
            (allowed_first, reason_first),
            (allowed_second, reason_second),
            (allowed_third, reason_third),
        )

    results = asyncio.run(run_checks())

    assert results[0] == (True, None)
    assert results[1] == (False, "client_evaluation_rate_limited")
    assert results[2] == (True, None)
