"""MCP server exposing Investigation Agent as a tool."""

# Suppress MCP server startup logs - must be done FIRST, before any imports
import logging
import os
import sys

# Set environment variable to suppress FastMCP logs
os.environ.setdefault("FASTMCP_QUIET", "1")

# Suppress all MCP-related loggers before any MCP imports
# Set root logger to WARNING to suppress INFO messages
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="")
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

# Suppress specific loggers
for logger_name in [
    "mcp.server",
    "mcp.server.server",
    "mcp",
    "fastmcp",
    "fastmcp.server",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    # Remove all handlers to prevent output
    logger.handlers = []

import asyncio  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from fastmcp import FastMCP  # noqa: E402

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import investigation_agent  # noqa: E402
from config import Config  # noqa: E402

# Setup OTEL instrumentation for this subprocess
# (zero-code instrumentation doesn't apply to child processes)
_tracer_provider = None

# def _setup_subprocess_instrumentation():
#     """Setup OTEL providers and LangChain instrumentation for subprocess."""
#     global _tracer_provider
    
#     endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
#     # Only setup if OTEL endpoint is configured
#     if not endpoint:
#         print("[investigation-agent] No OTEL_EXPORTER_OTLP_ENDPOINT, skipping instrumentation", file=sys.stderr, flush=True)
#         return
    
#     print(f"[investigation-agent] Setting up instrumentation -> {endpoint}", file=sys.stderr, flush=True)
    
#     try:
#         from opentelemetry import trace
#         from opentelemetry.sdk.trace import TracerProvider
#         from opentelemetry.sdk.trace.export import BatchSpanProcessor
#         from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        
#         protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
#         if protocol == "http/protobuf":
#             from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
#         else:
#             from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        
#         resource = Resource.create({
#             SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "investigation-agent-mcp"),
#         })
        
#         _tracer_provider = TracerProvider(resource=resource)
#         _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
#         trace.set_tracer_provider(_tracer_provider)
        
#         # Now instrument LangChain
#         from opentelemetry.instrumentation.langchain import LangchainInstrumentor
#         instrumentor = LangchainInstrumentor()
#         if not instrumentor.is_instrumented_by_opentelemetry:
#             instrumentor.instrument()
#     except Exception:
#         pass  # Instrumentation optional


# def _flush_telemetry():
#     """Flush telemetry before subprocess exits."""
#     global _tracer_provider
#     if _tracer_provider:
#         try:
#             _tracer_provider.force_flush(timeout_millis=5000)
#         except Exception:
#             pass

# _setup_subprocess_instrumentation()

mcp = FastMCP("investigation-agent")


@mcp.tool()
async def investigate_incident(
    service_id: str,
    investigation_checklist: str,
    scenario_id: str = None,
) -> dict:
    """Investigate an incident by querying metrics, logs, and traces.

    This tool exposes the Investigation Agent as an MCP tool that can be called
    by other agents or external systems.

    Args:
        service_id: The service identifier to investigate
        investigation_checklist: JSON string with investigation steps
        scenario_id: Optional scenario ID for seeded data

    Returns:
        Dict containing investigation results with hypotheses and evidence
    """
    print(f"[investigation-agent] Called: service_id={service_id}, scenario_id={scenario_id}", file=sys.stderr, flush=True)
    
    # Create a minimal state for the agent
    config = Config.from_env()
    if scenario_id:
        config.scenario_id = scenario_id

    state = {
        "service_id": service_id,
        "scenario_id": scenario_id,
        "session_id": f"mcp-{asyncio.get_event_loop().time()}",
        "triage_result": {
            "investigation_checklist": json.loads(investigation_checklist)
            if isinstance(investigation_checklist, str)
            else investigation_checklist,
        },
        "current_agent": "investigation",
        "hypotheses": [],
        "confidence_score": 0.0,
        "eval_metrics": {},
    }

    try:
        # Run investigation agent
        updated_state = investigation_agent(state, config)

        # Flush telemetry before returning
        _flush_telemetry()

        # Extract results
        investigation_result = updated_state.get("investigation_result", {})
        hypotheses = updated_state.get("hypotheses", [])
        confidence_score = updated_state.get("confidence_score", 0.0)

        return {
            "status": "success",
            "service_id": service_id,
            "hypotheses": hypotheses,
            "investigation_result": investigation_result,
            "confidence_score": confidence_score,
            "evidence_count": sum(len(h.get("evidence", [])) for h in hypotheses),
        }
    except Exception as e:
        _flush_telemetry()
        return {
            "status": "error",
            "error": str(e),
            "service_id": service_id,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
