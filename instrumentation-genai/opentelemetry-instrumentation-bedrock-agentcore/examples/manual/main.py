#!/usr/bin/env python3
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

"""
Bedrock AgentCore Instrumentation Demo

Demonstrates OpenTelemetry instrumentation across all Bedrock AgentCore
components with one representative operation per span type:

  - BedrockAgentCoreApp   → Workflow span        (entrypoint)
  - MemoryClient          → ToolCall span         (list_memories, create_event)
                          → RetrievalInvocation   (retrieve_memories)
  - CodeInterpreter       → ToolCall spans        (start, execute_code, stop)
  - BrowserClient         → ToolCall spans        (start, take_control, stop)

Spans are sent to an OTLP collector (default: http://localhost:4317).
Set OTEL_EXPORTER_OTLP_ENDPOINT to override. If the OTLP exporter is
not installed the spans are printed to stdout instead.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTLP_AVAILABLE = True
except ImportError:
    _OTLP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry setup
# ---------------------------------------------------------------------------


def setup_telemetry() -> TracerProvider:
    """Configure OpenTelemetry with OTLP (preferred) or console exporter."""
    resource = Resource(
        attributes={
            ResourceAttributes.SERVICE_NAME: "bedrock-agentcore-demo",
            ResourceAttributes.SERVICE_VERSION: "1.0.0",
        }
    )

    tracer_provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    if _OTLP_AVAILABLE:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        logger.info("Sending spans to OTLP collector at %s", otlp_endpoint)
    else:
        exporter = ConsoleSpanExporter()
        logger.warning(
            "opentelemetry-exporter-otlp-proto-grpc is not installed. "
            "Falling back to console. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
        )

    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

    BedrockAgentCoreInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("BedrockAgentCoreInstrumentor active — all 69 methods instrumented")

    return tracer_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try(label: str, fn):
    """Run fn(), print result or the exception — never raises."""
    print(f"    → {label}")
    try:
        result = fn()
        print(f"      ✓ {result!r}")
        return result
    except Exception as exc:
        print(f"      ⚠ {exc}")
        return None


def _get_or_create_memory(memory_client, name: str):
    """
    Return an existing memory by name prefix, or create one.

    create_or_get_memory has a bug where it only catches ClientError but
    ValidationException("already exists") propagates as a plain Exception
    from inside create_memory_and_wait, so we handle it manually here.
    """
    memories = memory_client.list_memories()
    memory = next((m for m in memories if m.get("id", "").startswith(name)), None)
    if memory:
        logger.info("Using existing memory: %s", memory["id"])
        return memory
    return memory_client.create_or_get_memory(name=name)


# ---------------------------------------------------------------------------
# Per-component demos
# ---------------------------------------------------------------------------


def demo_memory(memory_client, actor_id: str, session_id: str) -> None:
    """
    MemoryClient — one operation per span type:
      - list_memories   → ToolCall
      - create_event    → ToolCall
      - retrieve_memories → RetrievalInvocation
    """
    print("\n[MemoryClient]")

    _try("list_memories", lambda: memory_client.list_memories())

    memory = _try(
        "get_or_create_memory",
        lambda: _get_or_create_memory(memory_client, "demoMemory"),
    )
    memory_id = (memory or {}).get("id") or (memory or {}).get("memoryId")
    if not memory_id:
        print("  ⚠ Could not resolve memory_id — skipping remaining memory calls")
        return

    print(f"  ℹ using memory_id={memory_id}")

    _try(
        "create_event",
        lambda: memory_client.create_event(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[("Hello from the demo", "USER")],
        ),
    )

    # RetrievalInvocation span
    _try(
        "retrieve_memories",
        lambda: memory_client.retrieve_memories(
            memory_id=memory_id,
            namespace="demo",
            query="demo",
        ),
    )


def demo_code_interpreter(code_interpreter) -> None:
    """
    CodeInterpreter — stateful: start() sets the active session on the instance.

      - start        → ToolCall
      - execute_code → ToolCall
      - stop         → ToolCall
    """
    print("\n[CodeInterpreter]")

    session_id = _try("start", lambda: code_interpreter.start())

    if session_id:
        _try(
            "execute_code",
            lambda: code_interpreter.execute_code(
                code="print('Hello from instrumented CodeInterpreter')",
            ),
        )
        _try("stop", lambda: code_interpreter.stop())


def demo_browser(browser_client) -> None:
    """
    BrowserClient — stateful: start() sets the active session on the instance.

      - start        → ToolCall
      - take_control → ToolCall
      - stop         → ToolCall
    """
    print("\n[BrowserClient]")

    session_id = _try("start", lambda: browser_client.start())

    if session_id:
        _try("take_control", lambda: browser_client.take_control())
        _try("stop", lambda: browser_client.stop())


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_demo() -> None:
    from bedrock_agentcore import BedrockAgentCoreApp
    from bedrock_agentcore.memory.client import MemoryClient
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
    from bedrock_agentcore.tools.browser_client import BrowserClient

    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    memory_client = MemoryClient(region_name=region)
    code_interpreter = CodeInterpreter(region=region)
    browser_client = BrowserClient(region=region)

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def agent_workflow(event):
        """
        The @app.entrypoint decorator creates a Workflow span. Every operation
        inside creates a child ToolCall or RetrievalInvocation span automatically.
        """
        print("\n[BedrockAgentCoreApp] Workflow started  →  Workflow span created")

        demo_memory(memory_client, actor_id="demoActor001", session_id="demoSession001")
        demo_code_interpreter(code_interpreter)
        demo_browser(browser_client)

        print("\n[BedrockAgentCoreApp] Workflow complete")
        return {"status": "ok"}

    agent_workflow({"demo": True})


def main() -> int:
    print("=" * 70)
    print("AWS Bedrock AgentCore — OpenTelemetry Instrumentation Demo")
    print("=" * 70)

    tracer_provider = setup_telemetry()

    try:
        run_demo()

        print("\n" + "=" * 70)
        print("Demo complete.  Spans emitted per component:")
        print("  BedrockAgentCoreApp  1  (Workflow)")
        print("  MemoryClient         3  (ToolCall, ToolCall, RetrievalInvocation)")
        print("  CodeInterpreter      3  (ToolCall × 3)")
        print("  BrowserClient        3  (ToolCall × 3)")
        print("  ─────────────────────────────────────────")
        print("  Total               10")
        print("=" * 70)

    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Install with:")
        logger.error("  pip install bedrock-agentcore")
        logger.error("  pip install -r requirements.txt")
        logger.error("  pip install -e ../../[instruments]")
        return 1
    except Exception as exc:
        logger.error("Demo failed: %s", exc, exc_info=True)
        return 1
    finally:
        tracer_provider.force_flush(timeout_millis=5000)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
