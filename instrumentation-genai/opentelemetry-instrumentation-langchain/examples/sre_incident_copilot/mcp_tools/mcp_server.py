"""MCP Server - HTTP/SSE mode for Kubernetes deployment.

This server exposes observability tools via the MCP protocol over HTTP (SSE transport).
Designed to run in a separate pod and be called by the SRE Copilot using MCP client.

Usage:
    # Run as MCP HTTP server (SSE transport)
    python -m mcp_tools.mcp_server

    OTEL_LOGS_EXPORTER=none opentelemetry-instrument python -m mcp_tools.mcp_server
"""

import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# Suppress logging before imports
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="")
for logger_name in ["mcp", "mcp.server", "fastmcp", "fastmcp.server"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

from fastmcp import FastMCP

# Import seeded data
from mcp_tools.observability_tools import (
    SEEDED_LOGS_DATA,
    SEEDED_METRICS_DATA,
    SEEDED_TRACES_DATA,
)

# Create MCP server
mcp = FastMCP("observability-tools-http")


@mcp.tool()
async def metrics_query(
    service_id: str,
    metric_name: str,
    time_window: str = "15m",
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Query metrics for a service.

    Args:
        service_id: The service identifier (e.g., "payment-service")
        metric_name: Name of the metric (e.g., "error_rate", "latency_p95")
        time_window: Time window for the query (e.g., "15m", "1h")
        scenario_id: Optional scenario ID for seeded data

    Returns:
        Dict containing metric data with timestamps and values
    """
    if scenario_id and scenario_id in SEEDED_METRICS_DATA:
        data = SEEDED_METRICS_DATA[scenario_id].get(metric_name, [])
        if data:
            return {
                "service_id": service_id,
                "metric_name": metric_name,
                "time_window": time_window,
                "data_points": data,
                "status": "success",
            }

    # Generate synthetic data if not seeded
    now = datetime.now(timezone.utc)
    data_points = []
    for i in range(4):
        timestamp = now - timedelta(minutes=15 - i * 5)
        value = (
            random.uniform(0.1, 5.0)
            if "rate" in metric_name
            else random.uniform(100, 1000)
        )
        data_points.append({
            "timestamp": timestamp.isoformat() + "Z",
            "value": round(value, 2),
        })

    return {
        "service_id": service_id,
        "metric_name": metric_name,
        "time_window": time_window,
        "data_points": data_points,
        "status": "success",
    }


@mcp.tool()
async def logs_search(
    service_id: str,
    query: str,
    time_window: str = "15m",
    limit: int = 10,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Search logs for a service.

    Args:
        service_id: The service identifier
        query: Search query (e.g., "error", "timeout", "connection")
        time_window: Time window for the search
        limit: Maximum number of results
        scenario_id: Optional scenario ID for seeded data

    Returns:
        Dict containing matching log entries
    """
    if scenario_id and scenario_id in SEEDED_LOGS_DATA:
        logs = SEEDED_LOGS_DATA[scenario_id]
        query_lower = query.lower()
        query_words = query_lower.split()
        filtered_logs = [
            log
            for log in logs
            if query_lower in log.get("message", "").lower()
            or query_lower in log.get("level", "").lower()
            or any(
                word in log.get("message", "").lower()
                for word in query_words
                if len(word) > 3
            )
        ]
        return {
            "service_id": service_id,
            "query": query,
            "time_window": time_window,
            "results": filtered_logs[:limit],
            "count": len(filtered_logs),
            "status": "success",
        }

    # Generate synthetic logs if not seeded
    synthetic_logs = []
    now = datetime.now(timezone.utc)
    for i in range(min(limit, 5)):
        synthetic_logs.append({
            "timestamp": (now - timedelta(minutes=10 - i * 2)).isoformat() + "Z",
            "level": random.choice(["ERROR", "WARN", "INFO"]),
            "message": f"Synthetic log entry for {service_id}: {query}",
            "service": service_id,
        })

    return {
        "service_id": service_id,
        "query": query,
        "time_window": time_window,
        "results": synthetic_logs,
        "count": len(synthetic_logs),
        "status": "success",
    }


@mcp.tool()
async def trace_query(
    service_id: str,
    operation: Optional[str] = None,
    time_window: str = "15m",
    limit: int = 5,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Query traces for a service.

    Args:
        service_id: The service identifier
        operation: Optional operation name to filter by
        time_window: Time window for the query
        limit: Maximum number of traces
        scenario_id: Optional scenario ID for seeded data

    Returns:
        Dict containing trace data
    """
    if scenario_id and scenario_id in SEEDED_TRACES_DATA:
        traces = SEEDED_TRACES_DATA[scenario_id]
        if operation:
            traces = [t for t in traces if t.get("operation") == operation]
        return {
            "service_id": service_id,
            "operation": operation,
            "time_window": time_window,
            "traces": traces[:limit],
            "count": len(traces),
            "status": "success",
        }

    # Generate synthetic traces if not seeded
    synthetic_traces = []
    for i in range(min(limit, 3)):
        synthetic_traces.append({
            "trace_id": f"trace-{i:03d}",
            "span_id": f"span-{i:03d}",
            "service": service_id,
            "operation": operation or "unknown",
            "duration_ms": random.randint(100, 5000),
            "status": random.choice(["ok", "error"]),
        })

    return {
        "service_id": service_id,
        "operation": operation,
        "time_window": time_window,
        "traces": synthetic_traces,
        "count": len(synthetic_traces),
        "status": "success",
    }


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", 8081))
    print(f"[MCP Server] Starting on port {port}", flush=True)
    # Run with SSE transport for HTTP-based MCP
    mcp.run(transport="sse", port=port)
