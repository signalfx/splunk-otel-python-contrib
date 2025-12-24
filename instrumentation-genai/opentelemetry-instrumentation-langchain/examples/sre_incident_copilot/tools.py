"""Tools for agents to use."""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Suppress MCP client logging
logging.getLogger("mcp").setLevel(logging.ERROR)

# Use stderr for debug output to avoid interfering with MCP stdio communication
def debug_print(msg: str):
    """Print debug messages to stderr to avoid breaking MCP stdio protocol."""
    print(msg, file=sys.stderr)

from data_loader import DataLoader
from runbook_search import RunbookSearch


# Initialize shared resources
_data_loader = DataLoader(data_dir=os.getenv("DATA_DIR", "data"))
_runbook_search = RunbookSearch(data_dir=os.getenv("DATA_DIR", "data"))


@tool
def service_catalog_lookup(service_id: str) -> str:
    """Look up service information from the service catalog.

    Args:
        service_id: The service identifier (e.g., "payment-service")

    Returns:
        JSON string with service details including team, owner, dependencies, SLOs
    """
    service = _data_loader.get_service(service_id)
    if service:
        return json.dumps(service, indent=2)
    return json.dumps({"error": f"Service {service_id} not found"})


@tool
def runbook_search(query: str, k: int = 3) -> str:
    """Search runbooks for relevant incident response procedures.

    Args:
        query: Search query describing the incident or issue
        k: Number of results to return (default: 3)

    Returns:
        JSON string with runbook sections and citations
    """
    debug_print(f"🔧 [TOOL] runbook_search called: query={query[:100]}..., k={k}")
    results = _runbook_search.search(query, k=k)
    debug_print(f"🔧 [TOOL] runbook_search result: Found {len(results)} runbook sections")
    if results:
        for i, r in enumerate(results[:2], 1):  # Show first 2
            debug_print(
                f"   [{i}] {r.get('source', 'unknown')} (score: {r.get('similarity_score', 0):.3f})"
            )
    return json.dumps(results, indent=2)


async def _call_mcp_tool(
    tool_name: str, params: Dict[str, Any], scenario_id: Optional[str] = None
) -> Dict[str, Any]:
    """Call an MCP tool from the observability tools server."""
    mcp_script_path = os.path.join(
        os.path.dirname(__file__), "mcp_tools", "observability_tools.py"
    )

    # Ensure absolute path
    if not os.path.isabs(mcp_script_path):
        mcp_script_path = os.path.abspath(mcp_script_path)

    # Pass environment variables to suppress MCP server logs
    env = os.environ.copy()
    env.setdefault("FASTMCP_QUIET", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    
    server_params = StdioServerParameters(
        command="python",
        args=[mcp_script_path],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Add scenario_id to params if provided
                if scenario_id and "scenario_id" not in params:
                    params["scenario_id"] = scenario_id

                result = await session.call_tool(tool_name, params)

                if result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        return json.loads(content.text)
                    return {"status": "success", "data": str(content)}
                return {"status": "error", "message": "No content returned"}
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {"status": "error", "message": error_msg}


@tool
def metrics_query(
    service_id: str,
    metric_name: str,
    time_window: str = "15m",
    scenario_id: Optional[str] = None,
) -> str:
    """Query metrics for a service using MCP tool.

    Args:
        service_id: The service identifier
        metric_name: Name of the metric (e.g., "error_rate", "latency_p95")
        time_window: Time window for the query (e.g., "15m", "1h")
        scenario_id: Optional scenario ID for seeded data (if not provided, reads from CURRENT_SCENARIO_ID env var)

    Returns:
        JSON string with metric data

    Metric Selection Guidance by Alert Type:
    - latency_spike alerts: Query latency_p95, cache_hit_rate, cache_miss_rate,
      database query performance metrics (query_performance)
    - error_rate_critical alerts: Query error_rate, downstream_service_error_rate,
      authentication_failure_rate (for auth services), token_validation_errors
    - cache_miss_storm alerts: Query cache_hit_rate, cache_miss_rate, cache_eviction_rate,
      cache_memory_usage
    - queue_depth_runaway alerts: Query queue_depth, queue_processing_rate,
      downstream_service_latency_p95, downstream_service_error_rate
      (queue buildup is often caused by slow downstream services)
    - database_connection_pool alerts: Query connection_pool_usage, connection_pool_exhaustion_rate,
      database query performance metrics

    Common metrics: error_rate, latency_p95, cache_hit_rate, downstream_service_error_rate,
    downstream_service_latency_p95, queue_depth, queue_processing_rate
    """
    debug_print(
        f"🔧 [TOOL] metrics_query called: service_id={service_id}, metric_name={metric_name}, scenario_id={scenario_id}"
    )

    # Get scenario_id from env if not provided
    if not scenario_id:
        scenario_id = os.environ.get("CURRENT_SCENARIO_ID")
        debug_print(f"🔧 [TOOL] Got scenario_id from env: {scenario_id}")

    params = {
        "service_id": service_id,
        "metric_name": metric_name,
        "time_window": time_window,
    }
    if scenario_id:
        params["scenario_id"] = scenario_id

    try:
        # Use nest_asyncio if available, otherwise use asyncio.run
        try:
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                _call_mcp_tool("metrics_query", params, scenario_id)
            )
        except (ImportError, RuntimeError):
            # Fallback to asyncio.run (creates new event loop)
            result = asyncio.run(_call_mcp_tool("metrics_query", params, scenario_id))
        debug_print(f"🔧 [TOOL] metrics_query result: {json.dumps(result)[:200]}...")
        return json.dumps(result, indent=2)
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        debug_print(f"❌ Error calling metrics_query MCP tool: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg}, indent=2)


@tool
def logs_search(
    service_id: str,
    query: str,
    time_window: str = "15m",
    limit: int = 10,
    scenario_id: Optional[str] = None,
) -> str:
    """Search logs for a service using MCP tool.

    Args:
        service_id: The service identifier
        query: Search query (e.g., "error", "timeout", "connection")
        time_window: Time window for the search
        limit: Maximum number of results
        scenario_id: Optional scenario ID for seeded data (if not provided, reads from CURRENT_SCENARIO_ID env var)

    Returns:
        JSON string with log entries
    """
    debug_print(
        f"🔧 [TOOL] logs_search called: service_id={service_id}, query={query}, scenario_id={scenario_id}"
    )

    # Get scenario_id from env if not provided
    if not scenario_id:
        scenario_id = os.environ.get("CURRENT_SCENARIO_ID")
        debug_print(f"🔧 [TOOL] Got scenario_id from env: {scenario_id}")

    params = {
        "service_id": service_id,
        "query": query,
        "time_window": time_window,
        "limit": limit,
    }
    if scenario_id:
        params["scenario_id"] = scenario_id

    try:
        try:
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                _call_mcp_tool("logs_search", params, scenario_id)
            )
        except (ImportError, RuntimeError):
            result = asyncio.run(_call_mcp_tool("logs_search", params, scenario_id))
        debug_print(f"🔧 [TOOL] logs_search result: {json.dumps(result)[:200]}...")
        return json.dumps(result, indent=2)
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        debug_print(f"❌ Error calling logs_search MCP tool: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg}, indent=2)


@tool
def trace_query(
    service_id: str,
    operation: Optional[str] = None,
    time_window: str = "15m",
    limit: int = 5,
    scenario_id: Optional[str] = None,
) -> str:
    """Query traces for a service using MCP tool.

    Args:
        service_id: The service identifier
        operation: Optional operation name to filter by
        time_window: Time window for the query
        limit: Maximum number of traces
        scenario_id: Optional scenario ID for seeded data (if not provided, reads from CURRENT_SCENARIO_ID env var)

    Returns:
        JSON string with trace data
    """
    debug_print(
        f"🔧 [TOOL] trace_query called: service_id={service_id}, operation={operation}, scenario_id={scenario_id}"
    )

    # Get scenario_id from env if not provided
    if not scenario_id:
        scenario_id = os.environ.get("CURRENT_SCENARIO_ID")
        debug_print(f"🔧 [TOOL] Got scenario_id from env: {scenario_id}")

    params = {
        "service_id": service_id,
        "time_window": time_window,
        "limit": limit,
    }
    if operation:
        params["operation"] = operation
    if scenario_id:
        params["scenario_id"] = scenario_id

    try:
        try:
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                _call_mcp_tool("trace_query", params, scenario_id)
            )
        except (ImportError, RuntimeError):
            result = asyncio.run(_call_mcp_tool("trace_query", params, scenario_id))
        debug_print(f"🔧 [TOOL] trace_query result: {json.dumps(result)[:200]}...")
        return json.dumps(result, indent=2)
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        debug_print(f"❌ Error calling trace_query MCP tool: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg}, indent=2)


@tool
def task_writer(
    title: str,
    description: str,
    priority: str = "medium",
    assignee: Optional[str] = None,
) -> str:
    """Create a task/ticket (mocked Jira integration).

    Args:
        title: Task title
        description: Task description
        priority: Task priority (low, medium, high, critical)
        assignee: Optional assignee email

    Returns:
        JSON string with task details including task ID
    """
    import uuid

    task_id = str(uuid.uuid4())[:8]
    task = {
        "id": f"TASK-{task_id}",
        "title": title,
        "description": description,
        "priority": priority,
        "assignee": assignee,
        "status": "created",
        "url": f"https://jira.example.com/browse/TASK-{task_id}",
    }
    return json.dumps(task, indent=2)


@tool
def notifier(message: str, channel: str = "incidents") -> str:
    """Send a notification (mocked Slack integration).

    Args:
        message: Notification message
        channel: Channel name (default: "incidents")

    Returns:
        JSON string with notification details
    """
    import uuid

    notification_id = str(uuid.uuid4())[:8]
    notification = {
        "id": f"MSG-{notification_id}",
        "message": message,
        "channel": channel,
        "status": "sent",
        "timestamp": "2024-01-16T12:00:00Z",
    }
    return json.dumps(notification, indent=2)


@tool
def investigation_agent_mcp(
    service_id: str, investigation_checklist: str, scenario_id: Optional[str] = None
) -> str:
    """Call the Investigation Agent as an MCP tool.

    This tool allows other agents to invoke the Investigation Agent as a service.

    Args:
        service_id: The service identifier to investigate
        investigation_checklist: JSON string with investigation steps
        scenario_id: Optional scenario ID for seeded data

    Returns:
        JSON string with investigation results
    """
    import sys
    from pathlib import Path

    mcp_script_path = os.path.join(
        os.path.dirname(__file__), "mcp_tools", "investigation_agent_mcp.py"
    )

    # Pass environment variables to suppress MCP server logs
    env = os.environ.copy()
    env.setdefault("FASTMCP_QUIET", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    
    server_params = StdioServerParameters(
        command="python",
        args=[mcp_script_path],
        env=env,
    )

    params = {
        "service_id": service_id,
        "investigation_checklist": investigation_checklist,
    }
    if scenario_id:
        params["scenario_id"] = scenario_id

    try:

        async def _call():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("investigate_incident", params)
                    if result.content:
                        content = result.content[0]
                        if hasattr(content, "text"):
                            return json.loads(content.text)
                        return {"status": "error", "message": "Invalid response format"}
                    return {"status": "error", "message": "No content returned"}

        result = asyncio.run(_call())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)
