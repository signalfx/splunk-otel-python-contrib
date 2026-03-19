"""Test configuration for LlamaIndex instrumentation tests."""

import os

import pytest

from opentelemetry.util.genai import handler as genai_handler


@pytest.fixture(autouse=True)
def disable_deepeval():
    """Disable deepeval evaluators to prevent real API calls in CI."""
    original_evals = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS")

    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    setattr(genai_handler.get_telemetry_handler, "_default_handler", None)

    yield

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = original_evals

    setattr(genai_handler.get_telemetry_handler, "_default_handler", None)
