# Copyright Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BedrockAgentCoreInstrumentor."""

import os

import opentelemetry.instrumentation.bedrock_agentcore as agentcore_module
from opentelemetry.instrumentation.bedrock_agentcore import (
    BedrockAgentCoreInstrumentor,
    _iter_wrap_specs,
    _iter_wrap_targets,
)
from opentelemetry.instrumentation.bedrock_agentcore.utils import is_content_enabled


def test_instrumentor_initialization():
    """Test that the instrumentor can be initialized."""
    instrumentor = BedrockAgentCoreInstrumentor()
    assert instrumentor is not None


def test_instrument_uninstrument(tracer_provider, meter_provider):
    """Test basic instrument/uninstrument round-trip."""
    instrumentor = BedrockAgentCoreInstrumentor()

    # Instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Uninstrument
    instrumentor.uninstrument()

    # Should be able to instrument again
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    instrumentor.uninstrument()


def test_instrumentation_dependencies():
    """Test that instrumentation dependencies are correctly specified."""
    instrumentor = BedrockAgentCoreInstrumentor()
    deps = instrumentor.instrumentation_dependencies()
    assert "bedrock-agentcore" in str(deps)


def test_wrap_specs_match_unwrap_targets():
    """Wrap and unwrap should use the same Bedrock AgentCore target inventory."""
    wrap_targets = [
        (module, name) for module, name, _wrapper in _iter_wrap_specs(object(), False)
    ]
    unwrap_targets = list(_iter_wrap_targets())

    assert wrap_targets == unwrap_targets
    assert len(unwrap_targets) == len(set(unwrap_targets))
    assert (
        "bedrock_agentcore.tools.code_interpreter_client",
        "CodeInterpreter.upload_file",
    ) in unwrap_targets
    assert (
        "bedrock_agentcore.memory.session",
        "MemorySessionManager.add_turns",
    ) in unwrap_targets
    assert (
        "bedrock_agentcore.memory.session",
        "MemorySession.add_turns",
    ) not in unwrap_targets


def test_instrument_passes_logger_provider(
    monkeypatch, tracer_provider, meter_provider
):
    """Instrumentor passes logger_provider through to the util-genai handler."""
    calls = []
    logger_provider = object()

    def fake_get_telemetry_handler(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        agentcore_module, "get_telemetry_handler", fake_get_telemetry_handler
    )
    monkeypatch.setattr(
        agentcore_module,
        "wrap_function_wrapper",
        lambda module, name, wrapper: None,
    )

    instrumentor = BedrockAgentCoreInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    instrumentor.uninstrument()

    assert calls == [
        {
            "tracer_provider": tracer_provider,
            "meter_provider": meter_provider,
            "logger_provider": logger_provider,
        }
    ]


# ---------------------------------------------------------------------------
# is_content_enabled
# ---------------------------------------------------------------------------


def test_is_content_enabled_false_by_default():
    """is_content_enabled returns False when env var is unset."""
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
    assert is_content_enabled() is False


def test_is_content_enabled_true_when_set():
    """is_content_enabled returns True when env var is 'true'."""
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    try:
        assert is_content_enabled() is True
    finally:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


def test_is_content_enabled_case_insensitive():
    """is_content_enabled treats 'TRUE' and 'True' as enabled."""
    for value in ("TRUE", "True", "TrUe"):
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = value
        try:
            assert is_content_enabled() is True, f"Expected True for value={value!r}"
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


def test_is_content_enabled_false_for_non_true_values():
    """is_content_enabled returns False for '1', 'yes', 'on', 'false', empty string."""
    for value in ("1", "yes", "on", "false", "0", ""):
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = value
        try:
            assert is_content_enabled() is False, f"Expected False for value={value!r}"
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
