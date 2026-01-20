"""Smoke tests for CrewAI API surface expected by instrumentation."""

from crewai import Agent, Crew, Task
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool


def test_crewai_api_surface():
    """Ensure CrewAI classes expose methods we instrument."""
    assert hasattr(Crew, "kickoff")
    assert hasattr(Agent, "execute_task")
    assert hasattr(Task, "execute_sync")
    assert hasattr(BaseTool, "run")
    assert hasattr(CrewStructuredTool, "invoke")
