#!/usr/bin/env python3
"""Emit a single synthetic evaluation result to sanity-check telemetry plumbing."""

from __future__ import annotations

from time import sleep

from opentelemetry import _logs as otel_logs

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telemetry import init_telemetry
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    EvaluationResult,
)


def main() -> None:
    handler = init_telemetry(service_name="eval-smoke-test")

    invocation = LLMInvocation(
        request_model="smoke-model",
        provider="smoke",
        operation="chat",
        input_messages=[InputMessage(role="user", parts=[Text(content="ping")])],
    )
    handler.start_llm(invocation)
    invocation.output_messages = [
        OutputMessage(role="assistant", parts=[Text(content="pong")], finish_reason="stop")
    ]
    handler.stop_llm(invocation)

    result = EvaluationResult(metric_name="toxicity", score=0.05, label="clean")
    handler.evaluation_results(invocation, [result])

    handler.wait_for_evaluations(timeout=5)
    provider = otel_logs.get_logger_provider()
    flush = getattr(provider, "force_flush", None)
    if callable(flush):
        flush()
    else:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    # Give the OTLP exporter a moment before exiting
    sleep(60)


if __name__ == "__main__":
    main()
