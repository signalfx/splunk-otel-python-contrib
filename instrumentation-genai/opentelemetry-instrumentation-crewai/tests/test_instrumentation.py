"""Tests for CrewAI instrumentation mappings.

Tests the mapping of CrewAI operations to OpenTelemetry types:
- Crew.kickoff -> Workflow
- Agent.execute_task -> AgentInvocation
- Task.execute_sync -> Step
- BaseTool.run -> ToolCall
- CrewStructuredTool.invoke -> ToolCall
"""

import pytest
from unittest import mock

import opentelemetry.instrumentation.crewai.instrumentation as crewai_module
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.util.genai.types import (
    Workflow,
    AgentInvocation,
    Step,
    ToolCall,
    Error,
)


class TestCrewKickoffMapping:
    """Test Crew.kickoff -> Workflow mapping."""

    def test_kickoff_creates_workflow(self, stub_handler):
        """Crew.kickoff should create a Workflow span."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Research Crew"

        result = mock.MagicMock()
        result.raw = "Research completed successfully"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {"inputs": {}})

        assert len(stub_handler.started_workflows) == 1
        workflow = stub_handler.started_workflows[0]
        assert isinstance(workflow, Workflow)

    def test_workflow_has_correct_name(self, stub_handler):
        """Workflow should use the Crew's name."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Customer Support Crew"

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        workflow = stub_handler.started_workflows[0]
        assert workflow.name == "Customer Support Crew"

    def test_workflow_default_name_when_crew_has_no_name(self, stub_handler):
        """Workflow should use default name when Crew has no name attribute."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = None

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        workflow = stub_handler.started_workflows[0]
        assert workflow.name == "CrewAI Workflow"

    def test_workflow_has_correct_type(self, stub_handler):
        """Workflow should have workflow_type set to 'crewai.crew'."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        workflow = stub_handler.started_workflows[0]
        assert workflow.workflow_type == "crewai.crew"

    def test_workflow_has_framework_and_system(self, stub_handler):
        """Workflow should have framework and system set to 'crewai'."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        workflow = stub_handler.started_workflows[0]
        assert workflow.framework == "crewai"
        assert workflow.system == "crewai"

    def test_workflow_captures_inputs(self, stub_handler):
        """Workflow should capture input data in input_messages."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        inputs = {"topic": "AI", "depth": "comprehensive"}
        crewai_module._wrap_crew_kickoff(
            mock_wrapped, mock_crew, (), {"inputs": inputs}
        )

        workflow = stub_handler.started_workflows[0]
        assert workflow.input_messages and len(workflow.input_messages) > 0
        content = workflow.input_messages[0].parts[0].content
        assert "topic" in content
        assert "AI" in content

    def test_workflow_captures_inputs_from_args(self, stub_handler):
        """Workflow should capture inputs passed positionally in input_messages."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Args Crew"

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(
            mock_wrapped, mock_crew, ({"topic": "args"},), {}
        )

        workflow = stub_handler.started_workflows[0]
        assert workflow.input_messages and len(workflow.input_messages) > 0
        assert "topic" in workflow.input_messages[0].parts[0].content

    def test_workflow_captures_output(self, stub_handler):
        """Workflow should capture result output in output_messages."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Final research report on artificial intelligence"
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        workflow = stub_handler.stopped_workflows[0]
        assert workflow.output_messages and len(workflow.output_messages) > 0
        assert "artificial intelligence" in workflow.output_messages[0].parts[0].content

    def test_workflow_captures_empty_output(self, stub_handler):
        """Workflow should capture empty output in output_messages."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Empty Output Crew"

        result = mock.MagicMock()
        result.raw = ""
        mock_wrapped = mock.MagicMock(return_value=result)

        crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        workflow = stub_handler.stopped_workflows[0]
        assert workflow.output_messages and len(workflow.output_messages) > 0
        assert workflow.output_messages[0].parts[0].content == '""'

    def test_workflow_error_handling(self, stub_handler):
        """Workflow should capture errors on failure."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Failing Crew"

        def raise_error(*args, **kwargs):
            raise RuntimeError("Crew execution failed")

        mock_wrapped = mock.MagicMock(side_effect=raise_error)

        with pytest.raises(RuntimeError, match="Crew execution failed"):
            crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        assert len(stub_handler.failed_entities) == 1
        entity, error = stub_handler.failed_entities[0]
        assert isinstance(entity, Workflow)
        assert isinstance(error, Error)
        assert "Crew execution failed" in error.message

    def test_kickoff_long_inputs(self, stub_handler):
        """Long inputs should be captured in input_messages."""
        crewai_module._handler = stub_handler

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Done"
        mock_wrapped = mock.MagicMock(return_value=result)

        # Create a very long input
        long_input = "x" * 1000
        crewai_module._wrap_crew_kickoff(
            mock_wrapped, mock_crew, (), {"inputs": {"data": long_input}}
        )

        workflow = stub_handler.started_workflows[0]
        assert workflow.input_messages and len(workflow.input_messages) > 0
        # Long input should be captured in full (no truncation in structured messages)
        assert long_input in workflow.input_messages[0].parts[0].content


class TestAgentExecuteTaskMapping:
    """Test Agent.execute_task -> AgentInvocation mapping."""

    def test_agent_creates_agent_invocation(self, stub_handler):
        """Agent.execute_task should create an AgentInvocation span."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Research Analyst"

        mock_task = mock.MagicMock()
        mock_task.description = "Analyze market trends"

        mock_wrapped = mock.MagicMock(return_value="Analysis complete")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (), {"task": mock_task}
        )

        assert len(stub_handler.started_agents) == 1
        agent = stub_handler.started_agents[0]
        assert isinstance(agent, AgentInvocation)

    def test_agent_invocation_has_correct_name(self, stub_handler):
        """AgentInvocation should use the Agent's role as name."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Senior Data Scientist"

        mock_task = mock.MagicMock()
        mock_task.description = "Build ML model"

        mock_wrapped = mock.MagicMock(return_value="Model built")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (), {"task": mock_task}
        )

        agent = stub_handler.started_agents[0]
        assert agent.name == "Senior Data Scientist"

    def test_agent_default_name_when_no_role(self, stub_handler):
        """AgentInvocation should use default name when agent has no role."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock(spec=[])  # No role attribute

        mock_task = mock.MagicMock()
        mock_task.description = "Task"

        mock_wrapped = mock.MagicMock(return_value="Done")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (), {"task": mock_task}
        )

        agent = stub_handler.started_agents[0]
        assert agent.name == "Unknown Agent"

    def test_agent_captures_task_description(self, stub_handler):
        """AgentInvocation should capture task description in input_messages."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Writer"

        mock_task = mock.MagicMock()
        mock_task.description = "Write a blog post about Python best practices"

        mock_wrapped = mock.MagicMock(return_value="Blog post written")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (), {"task": mock_task}
        )

        agent = stub_handler.started_agents[0]
        assert agent.input_messages and len(agent.input_messages) > 0
        assert "Python best practices" in agent.input_messages[0].parts[0].content

    def test_agent_captures_task_from_args(self, stub_handler):
        """AgentInvocation should capture task passed positionally in input_messages."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Writer"

        mock_task = mock.MagicMock()
        mock_task.description = "Positional task"

        mock_wrapped = mock.MagicMock(return_value="Done")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (mock_task,), {}
        )

        agent = stub_handler.started_agents[0]
        assert agent.input_messages and len(agent.input_messages) > 0
        assert "Positional task" in agent.input_messages[0].parts[0].content

    def test_agent_captures_result(self, stub_handler):
        """AgentInvocation should capture execution result in output_messages."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Analyst"

        mock_task = mock.MagicMock()
        mock_task.description = "Analyze data"

        mock_wrapped = mock.MagicMock(return_value="Data analysis shows 20% growth")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (), {"task": mock_task}
        )

        agent = stub_handler.stopped_agents[0]
        assert agent.output_messages and len(agent.output_messages) > 0
        assert "20% growth" in agent.output_messages[0].parts[0].content

    def test_agent_captures_empty_result(self, stub_handler):
        """AgentInvocation should capture empty output in output_messages."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Analyst"

        mock_task = mock.MagicMock()
        mock_task.description = "Analyze data"

        mock_wrapped = mock.MagicMock(return_value="")

        crewai_module._wrap_agent_execute_task(
            mock_wrapped, mock_agent, (), {"task": mock_task}
        )

        agent = stub_handler.stopped_agents[0]
        assert agent.output_messages and len(agent.output_messages) > 0
        assert agent.output_messages[0].parts[0].content == '""'

    def test_agent_error_handling(self, stub_handler):
        """AgentInvocation should capture errors on failure."""
        crewai_module._handler = stub_handler

        mock_agent = mock.MagicMock()
        mock_agent.role = "Failing Agent"

        mock_task = mock.MagicMock()
        mock_task.description = "Impossible task"

        def raise_error(*args, **kwargs):
            raise ValueError("Agent cannot complete task")

        mock_wrapped = mock.MagicMock(side_effect=raise_error)

        with pytest.raises(ValueError, match="Agent cannot complete task"):
            crewai_module._wrap_agent_execute_task(
                mock_wrapped, mock_agent, (), {"task": mock_task}
            )

        assert len(stub_handler.failed_entities) == 1
        entity, error = stub_handler.failed_entities[0]
        assert isinstance(entity, AgentInvocation)


class TestTaskExecuteMapping:
    """Test Task.execute_sync -> Step mapping."""

    def test_task_creates_step(self, stub_handler):
        """Task.execute_sync should create a Step span."""
        crewai_module._handler = stub_handler

        mock_task = mock.MagicMock()
        mock_task.description = "Research market competitors"
        mock_task.expected_output = "Competitor analysis report"
        mock_task.agent = mock.MagicMock()
        mock_task.agent.role = "Market Researcher"

        mock_wrapped = mock.MagicMock(return_value="Competitor report generated")

        crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

        assert len(stub_handler.started_steps) == 1
        step = stub_handler.started_steps[0]
        assert isinstance(step, Step)

    def test_step_has_correct_name(self, stub_handler):
        """Step should use the task description as name."""
        crewai_module._handler = stub_handler

        mock_task = mock.MagicMock()
        mock_task.description = "Generate sales report"
        mock_task.expected_output = "PDF report"
        mock_task.agent = mock.MagicMock()
        mock_task.agent.role = "Reporter"

        mock_wrapped = mock.MagicMock(return_value="Report generated")

        crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

        step = stub_handler.started_steps[0]
        assert step.name == "Generate sales report"

    def test_step_captures_description(self, stub_handler):
        """Step should capture task description."""
        crewai_module._handler = stub_handler

        mock_task = mock.MagicMock()
        mock_task.description = "Detailed analysis of Q4 performance metrics"
        mock_task.expected_output = "Analysis document"
        mock_task.agent = mock.MagicMock()
        mock_task.agent.role = "Analyst"

        mock_wrapped = mock.MagicMock(return_value="Analysis complete")

        crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

        step = stub_handler.started_steps[0]
        assert "Q4 performance metrics" in step.description

    def test_step_captures_expected_output_as_objective(self, stub_handler):
        """Step should capture expected_output as objective."""
        crewai_module._handler = stub_handler

        mock_task = mock.MagicMock()
        mock_task.description = "Write article"
        mock_task.expected_output = "A 500-word article on AI trends"
        mock_task.agent = mock.MagicMock()
        mock_task.agent.role = "Writer"

        mock_wrapped = mock.MagicMock(return_value="Article written")

        crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

        step = stub_handler.started_steps[0]
        assert "500-word article" in step.objective

    def test_step_captures_assigned_agent(self, stub_handler):
        """Step should capture the assigned agent's role."""
        crewai_module._handler = stub_handler

        mock_task = mock.MagicMock()
        mock_task.description = "Code review"
        mock_task.expected_output = "Review comments"
        mock_task.agent = mock.MagicMock()
        mock_task.agent.role = "Senior Developer"

        mock_wrapped = mock.MagicMock(return_value="Review complete")

        crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

        step = stub_handler.started_steps[0]
        assert step.assigned_agent == "Senior Developer"

    def test_step_error_handling(self, stub_handler):
        """Step should capture errors on failure."""
        crewai_module._handler = stub_handler

        mock_task = mock.MagicMock()
        mock_task.description = "Failing task"
        mock_task.expected_output = "Never reached"
        mock_task.agent = mock.MagicMock()
        mock_task.agent.role = "Agent"

        def raise_error(*args, **kwargs):
            raise RuntimeError("Task execution failed")

        mock_wrapped = mock.MagicMock(side_effect=raise_error)

        with pytest.raises(RuntimeError, match="Task execution failed"):
            crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

        assert len(stub_handler.failed_entities) == 1
        entity, error = stub_handler.failed_entities[0]
        assert isinstance(entity, Step)


class TestBaseToolRunMapping:
    """Test BaseTool.run -> ToolCall mapping."""

    def test_tool_creates_tool_call(self, stub_handler):
        """BaseTool.run should create a ToolCall span."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "web_search"

        mock_wrapped = mock.MagicMock(return_value="Search results")

        crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {"query": "AI news"})

        assert len(stub_handler.started_tool_calls) == 1
        tool_call = stub_handler.started_tool_calls[0]
        assert isinstance(tool_call, ToolCall)

    def test_tool_call_has_correct_name(self, stub_handler):
        """ToolCall should use the tool's name."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "calculator"

        mock_wrapped = mock.MagicMock(return_value="42")

        crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {"expression": "6*7"})

        tool_call = stub_handler.started_tool_calls[0]
        assert tool_call.name == "calculator"

    def test_tool_call_captures_arguments(self, stub_handler):
        """ToolCall should capture arguments."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "file_reader"

        mock_wrapped = mock.MagicMock(return_value="File contents")

        crewai_module._wrap_tool_run(
            mock_wrapped, mock_tool, (), {"path": "/tmp/data.txt", "encoding": "utf-8"}
        )

        tool_call = stub_handler.started_tool_calls[0]
        assert "path" in tool_call.arguments
        assert "/tmp/data.txt" in tool_call.arguments

    def test_tool_call_has_framework_and_system(self, stub_handler):
        """ToolCall should have framework and system set to 'crewai'."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "api_call"

        mock_wrapped = mock.MagicMock(return_value="Response")

        crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {})

        tool_call = stub_handler.started_tool_calls[0]
        assert tool_call.framework == "crewai"
        assert tool_call.system == "crewai"

    def test_tool_call_error_handling(self, stub_handler):
        """ToolCall should capture errors on failure."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "failing_tool"

        def raise_error(*args, **kwargs):
            raise IOError("Tool execution failed")

        mock_wrapped = mock.MagicMock(side_effect=raise_error)

        with pytest.raises(IOError, match="Tool execution failed"):
            crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {})

        assert len(stub_handler.failed_entities) == 1
        entity, error = stub_handler.failed_entities[0]
        assert isinstance(entity, ToolCall)


class TestStructuredToolInvokeMapping:
    """Test CrewStructuredTool.invoke -> ToolCall mapping."""

    def test_structured_tool_creates_tool_call(self, stub_handler):
        """CrewStructuredTool.invoke should create a ToolCall span."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "structured_search"

        mock_wrapped = mock.MagicMock(return_value="Structured search results")

        crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {"query": "test"})

        assert len(stub_handler.started_tool_calls) == 1
        tool_call = stub_handler.started_tool_calls[0]
        assert isinstance(tool_call, ToolCall)

    def test_structured_tool_has_correct_name(self, stub_handler):
        """ToolCall from structured tool should use correct name."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "weather_lookup"

        mock_wrapped = mock.MagicMock(return_value="Sunny, 25°C")

        crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {"city": "Tokyo"})

        tool_call = stub_handler.started_tool_calls[0]
        assert tool_call.name == "weather_lookup"

    def test_structured_tool_captures_arguments(self, stub_handler):
        """ToolCall from structured tool should capture arguments."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "database_query"

        mock_wrapped = mock.MagicMock(return_value=[{"id": 1}])

        crewai_module._wrap_tool_run(
            mock_wrapped, mock_tool, (), {"table": "users", "limit": 10}
        )

        tool_call = stub_handler.started_tool_calls[0]
        assert "table" in tool_call.arguments
        assert "users" in tool_call.arguments

    def test_structured_tool_error_handling(self, stub_handler):
        """ToolCall from structured tool should capture errors."""
        crewai_module._handler = stub_handler

        mock_tool = mock.MagicMock()
        mock_tool.name = "failing_structured_tool"

        def raise_error(*args, **kwargs):
            raise ConnectionError("Database connection failed")

        mock_wrapped = mock.MagicMock(side_effect=raise_error)

        with pytest.raises(ConnectionError, match="Database connection failed"):
            crewai_module._wrap_tool_run(mock_wrapped, mock_tool, (), {})

        assert len(stub_handler.failed_entities) == 1


class TestInstrumentationDependencies:
    """Test instrumentation setup and dependencies."""

    def test_instrumentation_dependencies(self):
        """Instrumentor should declare crewai dependency."""
        instrumentor = CrewAIInstrumentor()
        deps = instrumentor.instrumentation_dependencies()

        assert len(deps) == 1
        assert "crewai" in deps[0]

    def test_instrument_wraps_correct_methods(self, tracer_provider, meter_provider):
        """Instrumentor should wrap the correct CrewAI methods."""
        instrumentor = CrewAIInstrumentor()

        wrapped_calls = []

        def mock_wrap(module, name, wrapper):
            wrapped_calls.append((module, name))

        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper",
            side_effect=mock_wrap,
        ):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

        # Check all expected methods are wrapped
        expected_wraps = [
            ("crewai.crew", "Crew.kickoff"),
            ("crewai.agent", "Agent.execute_task"),
            ("crewai.task", "Task.execute_sync"),
            ("crewai.tools.base_tool", "BaseTool.run"),
            ("crewai.tools.structured_tool", "CrewStructuredTool.invoke"),
        ]

        for expected in expected_wraps:
            assert expected in wrapped_calls, f"Expected {expected} to be wrapped"

        # Cleanup
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            instrumentor.uninstrument()

    def test_uninstrument_unwraps_correct_methods(
        self, tracer_provider, meter_provider
    ):
        """Uninstrumentor should unwrap the correct methods."""
        instrumentor = CrewAIInstrumentor()

        unwrapped_calls = []

        def mock_unwrap(module, name):
            unwrapped_calls.append((module, name))

        # First instrument
        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
        ):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )

        # Then uninstrument
        with mock.patch(
            "opentelemetry.instrumentation.crewai.instrumentation.unwrap",
            side_effect=mock_unwrap,
        ):
            instrumentor.uninstrument()

        # Check all expected methods are unwrapped
        expected_unwraps = [
            ("crewai.crew.Crew", "kickoff"),
            ("crewai.agent.Agent", "execute_task"),
            ("crewai.task.Task", "execute_sync"),
            ("crewai.tools.base_tool.BaseTool", "run"),
            ("crewai.tools.structured_tool.CrewStructuredTool", "invoke"),
        ]

        for expected in expected_unwraps:
            assert expected in unwrapped_calls, f"Expected {expected} to be unwrapped"


class TestWrapperGracefulDegradation:
    """Test that wrappers handle errors gracefully."""

    def test_wrapper_continues_when_instrumentation_setup_fails(self):
        """Wrapper should still call original function if instrumentation setup fails."""
        # Set handler to None to simulate missing handler
        crewai_module._handler = None

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Success"
        mock_wrapped = mock.MagicMock(return_value=result)

        # Should not raise, should call wrapped function
        returned = crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        mock_wrapped.assert_called_once()
        assert returned == result

    def test_wrapper_continues_when_handler_method_fails(self, stub_handler):
        """Wrapper should continue if handler method fails during setup."""
        crewai_module._handler = stub_handler

        # Make start_workflow raise an exception
        stub_handler.start_workflow = mock.MagicMock(
            side_effect=RuntimeError("Handler error")
        )

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Success"
        mock_wrapped = mock.MagicMock(return_value=result)

        # Should not raise, should call wrapped function
        returned = crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        mock_wrapped.assert_called_once()
        assert returned == result

    def test_wrapper_ignores_stop_errors_on_success(self, stub_handler):
        """Wrapper should ignore errors when stopping span on success."""
        crewai_module._handler = stub_handler

        # Make stop_workflow raise an exception
        stub_handler.stop_workflow = mock.MagicMock(
            side_effect=RuntimeError("Stop error")
        )

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        result = mock.MagicMock()
        result.raw = "Success"
        mock_wrapped = mock.MagicMock(return_value=result)

        # Should not raise, should return result
        returned = crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})

        assert returned == result

    def test_wrapper_ignores_fail_errors_on_exception(self, stub_handler):
        """Wrapper should ignore errors when recording failure."""
        crewai_module._handler = stub_handler

        # Make fail raise an exception
        stub_handler.fail = mock.MagicMock(side_effect=RuntimeError("Fail error"))

        mock_crew = mock.MagicMock()
        mock_crew.name = "Test Crew"

        def raise_original_error(*args, **kwargs):
            raise ValueError("Original error")

        mock_wrapped = mock.MagicMock(side_effect=raise_original_error)

        # Should raise the original error, not the fail error
        with pytest.raises(ValueError, match="Original error"):
            crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})
