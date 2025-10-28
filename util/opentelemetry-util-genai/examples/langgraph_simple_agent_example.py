#!/usr/bin/env python3
"""
Mocked LangGraph Agent Example with Manual OpenTelemetry Instrumentation.

This example synthesizes the telemetry that a LangGraph agent (backed by
LangChain) would normally emit. Instead of depending on the LangChain runtime,
we construct the OpenTelemetry GenAI types directly and send them through the
`TelemetryHandler`. This keeps the example dependency-free while still showing
the full set of attributes that would be captured by a real callback.

Run with:
  python examples/langgraph_simple_agent_example.py
"""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Dict

from opentelemetry import _logs as logs
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

# Set environment variables for content capture so that spans and events
# include message payloads when emitters allow it.
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE", "SPAN_AND_EVENT"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event"
)

# Configure OpenTelemetry with OTLP exporters so the spans can be shipped to a
# collector if one is running locally.
trace.set_tracer_provider(TracerProvider())
span_processor = BatchSpanProcessor(OTLPSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

logs.set_logger_provider(LoggerProvider())
logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)


def _mock_langgraph_execution(question: str) -> Dict[str, Any]:
    """Return mocked LangGraph execution details for the supplied question."""

    question_key = question.strip().lower()
    capital_answers = {
        "what is the capital of france?": "Paris is the capital of France.",
        "what is the capital of japan?": "Tokyo is the capital city of Japan.",
        "what is the capital of brazil?": "Brasília is the capital of Brazil.",
        "what is the capital of australia?": "Canberra serves as Australia's capital.",
        "what is the capital of canada?": "Ottawa is the capital city of Canada.",
    }

    # Produce a deterministic answer if we know the question, otherwise fall back.
    answer = capital_answers.get(
        question_key,
        "I'm not sure, but checking an up-to-date atlas should help.",
    )

    requested_model = "gpt-4o-mini"
    response_model = "gpt-4o-mini-2024-07-18"
    invocation_params = {
        "temperature": 0.0,
        "max_tokens": 256,
        "top_p": 1.0,
        "top_k": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop_sequences": ["\n\n"],
        "service_tier": "standard",
    }

    # Estimate tokens based on simple heuristics to keep the example lightweight.
    input_tokens = max(8, len(question.split()) + 4)
    output_tokens = max(16, len(answer.split()) + 6)

    return {
        "answer": answer,
        "requested_model": requested_model,
        "response_model": response_model,
        "invocation_params": invocation_params,
        "finish_reason": "stop",
        "response_id": f"resp-{uuid.uuid4()}",
        "system_fingerprint": f"fp-{uuid.uuid4().hex[:8]}",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "agent_run_id": uuid.uuid4(),
        "llm_run_id": uuid.uuid4(),
        "agent_initialization_delay": random.uniform(0.01, 0.05),
        "llm_latency": random.uniform(0.05, 0.15),
    }


def run_mock_agent_with_telemetry(question: str) -> str:
    """Emit agent and LLM telemetry for a mocked LangGraph execution."""

    handler = get_telemetry_handler()
    mocked = _mock_langgraph_execution(question)

    print(f"\n{'=' * 80}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 80}\n")

    print(f"\n{'=' * 80}")
    print("create_agent span...")
    print(f"{'=' * 80}\n")

    agent_create = AgentCreation(
        name="simple_capital_agent",
        agent_type="qa",
        description="Simple agent that answers capital city questions from knowledge.",
        model=mocked["requested_model"],
        provider="openai",
        framework="langgraph",
        system_instructions=(
            "You are a helpful assistant that answers questions about capital "
            "cities using your knowledge."
        ),
    )
    agent_create.agent_name = agent_create.name
    agent_create.agent_id = str(agent_create.run_id)
    agent_create.attributes["agent.version"] = "1.0"
    agent_create.attributes["agent.temperature"] = mocked["invocation_params"][
        "temperature"
    ]

    handler.start_agent(agent_create)
    time.sleep(mocked["agent_initialization_delay"])
    handler.stop_agent(agent_create)

    print(f"\n{'=' * 80}")
    print("invoke_agent span")
    print(f"{'=' * 80}\n")

    agent_invoke = AgentInvocation(
        name="simple_capital_agent",
        agent_type="qa",
        model=mocked["requested_model"],
        provider="openai",
        framework="langgraph",
        input_context=question,
        run_id=mocked["agent_run_id"],
        parent_run_id=agent_create.run_id,
    )
    agent_invoke.agent_name = agent_invoke.name
    agent_invoke.agent_id = agent_create.agent_id
    agent_invoke.attributes["agent.temperature"] = mocked["invocation_params"][
        "temperature"
    ]

    handler.start_agent(agent_invoke)

    user_message = InputMessage(role="user", parts=[Text(content=question)])
    assistant_message = OutputMessage(
        role="assistant",
        parts=[Text(content=mocked["answer"])],
        finish_reason=mocked["finish_reason"],
    )

    llm_invocation = LLMInvocation(
        request_model=mocked["requested_model"],
        response_model_name=mocked["response_model"],
        provider="openai",
        framework="langgraph",
        input_messages=[user_message],
        output_messages=[assistant_message],
        response_id=mocked["response_id"],
        request_temperature=mocked["invocation_params"]["temperature"],
        request_top_p=mocked["invocation_params"]["top_p"],
        request_top_k=mocked["invocation_params"]["top_k"],
        request_frequency_penalty=mocked["invocation_params"][
            "frequency_penalty"
        ],
        request_presence_penalty=mocked["invocation_params"][
            "presence_penalty"
        ],
        request_stop_sequences=mocked["invocation_params"]["stop_sequences"],
        request_max_tokens=mocked["invocation_params"]["max_tokens"],
        request_choice_count=1,
        output_type="text",
        response_finish_reasons=[mocked["finish_reason"]],
        response_system_fingerprint=mocked["system_fingerprint"],
        request_service_tier=mocked["invocation_params"]["service_tier"],
        run_id=mocked["llm_run_id"],
        parent_run_id=agent_invoke.run_id,
        agent_name=agent_invoke.name,
        agent_id=str(agent_invoke.run_id),
    )
    llm_invocation.input_tokens = mocked["input_tokens"]
    llm_invocation.output_tokens = mocked["output_tokens"]

    handler.start_llm(llm_invocation)
    time.sleep(mocked["llm_latency"])
    handler.stop_llm(llm_invocation)

    print(f"{'=' * 80}")
    print(
        f"Token Usage (mocked): Input={llm_invocation.input_tokens}, "
        f"Output={llm_invocation.output_tokens}"
    )
    print(
        f"Model: {mocked['response_model']}, "
        f"Finish Reason: {mocked['finish_reason']}\n"
    )
    print(f"{'=' * 80}")

    agent_invoke.output_result = mocked["answer"]
    handler.stop_agent(agent_invoke)

    print(f"{'=' * 80}")
    print(f"FINAL ANSWER: {mocked['answer']}")
    print(f"{'=' * 80}\n")

    return mocked["answer"]


def main() -> None:
    """Main entry point for the example."""

    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
        "What is the capital of Australia?",
        "What is the capital of Canada?",
    ]
    question = random.choice(questions)

    run_mock_agent_with_telemetry(question)

    print(f"\n{'=' * 80}")
    print("\nWaiting for metrics export...")
    print(f"{'=' * 80}\n")
    time.sleep(6)


if __name__ == "__main__":
    main()
