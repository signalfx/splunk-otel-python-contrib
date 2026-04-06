from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
GENAI_ROOT = TESTS_ROOT.parent
REPO_ROOT = GENAI_ROOT.parent
PROJECT_ROOT = REPO_ROOT.parent

for path in (
    PROJECT_ROOT / "opentelemetry-instrumentation" / "src",
    GENAI_ROOT / "src",
    PROJECT_ROOT / "util" / "opentelemetry-util-genai" / "src",
    REPO_ROOT / "openai_agents_lib",
    REPO_ROOT / "openai_lib",
    TESTS_ROOT / "stubs",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _reset_all_telemetry_state():
    """Reset every piece of shared singleton state that can leak between tests.

    Uses importlib.import_module for agents.tracing so we always operate on the
    *current* module instance in sys.modules (test_tracer.py pops and re-imports
    the stub module, which would otherwise leave conftest with a stale reference).
    """
    from opentelemetry.instrumentation.openai_agents import (
        OpenAIAgentsInstrumentor,
    )
    from opentelemetry.util.genai.handler import get_telemetry_handler

    # 1. Handler singleton (match langchain/crewai pattern: setattr to None)
    setattr(get_telemetry_handler, "_default_handler", None)
    # 2. Stub trace-provider processor list — resolve at call time
    tracing = importlib.import_module("agents.tracing")
    tracing.set_trace_processors([])
    # 3. OpenAIAgentsInstrumentor is a singleton via BaseInstrumentor.__new__;
    #    reset its instance-level flag and processor reference so re-instrument works.
    singleton = OpenAIAgentsInstrumentor()
    singleton._is_instrumented_by_opentelemetry = False
    singleton._processor = None


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    """Reset shared singletons before/after each test for isolation."""
    _reset_all_telemetry_state()
    yield
    _reset_all_telemetry_state()
