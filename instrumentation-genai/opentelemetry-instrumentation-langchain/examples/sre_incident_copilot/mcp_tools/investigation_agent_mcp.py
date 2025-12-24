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
for logger_name in ["mcp.server", "mcp.server.server", "mcp", "fastmcp", "fastmcp.server"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    # Remove all handlers to prevent output
    logger.handlers = []

import asyncio
import json
from pathlib import Path

from fastmcp import FastMCP

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import investigation_agent  # noqa: E402
from config import Config  # noqa: E402

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


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
