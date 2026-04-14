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


def _configure_manual_instrumentation():
    """Configure manual OpenTelemetry instrumentation."""

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry import _events, _logs, metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk._events import EventLoggerProvider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    _logs.set_logger_provider(LoggerProvider())
    _logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )
    _events.set_event_logger_provider(EventLoggerProvider())

    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    instrumentor = LangchainInstrumentor()
    instrumentor.instrument()

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor
    instrumentor2 = FastMCPInstrumentor()
    instrumentor2.instrument()

if __name__ == "__main__":
    _configure_manual_instrumentation()
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
            return {
                "status": "error",
                "error": str(e),
                "service_id": service_id,
            }

    mcp.run(transport="stdio", show_banner=False)
