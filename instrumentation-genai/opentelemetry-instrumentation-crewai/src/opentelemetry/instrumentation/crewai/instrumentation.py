"""
OpenTelemetry CrewAI Instrumentation

Wrapper-based instrumentation for CrewAI using splunk-otel-util-genai.
"""

import json
import logging
from typing import Any, Collection, Optional

from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)
from opentelemetry.util.genai.types import (
    Workflow,
    AgentInvocation,
    Step,
    ToolCall,
    Error,
    InputMessage,
    OutputMessage,
    Text,
)

_instruments = ("crewai >= 0.70.0",)

# Global handler instance (singleton)
_handler: Optional[TelemetryHandler] = None

_logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> Optional[str]:
    """Serialize object to JSON string.

    Uses default=str to handle non-JSON-serializable objects by converting
    them to their string representation while keeping the overall structure
    as valid JSON.
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _make_input_message(messages: dict[str, Any]) -> list[InputMessage]:
    """Create structured input message with full data as JSON."""
    input_messages: list[InputMessage] = []
    if messages is None:
        return []
    for key, value in messages.items():
        if value:
            if key == "description" or key == "expected_output" or key == "inquiry":
                input_message = InputMessage(
                    role="user", parts=[Text(_safe_str(value))]
                )
                input_messages.append(input_message)

    return input_messages


def _make_output_message(result: Any) -> list[OutputMessage]:
    """Create structured output message with full data as JSON."""
    output_messages: list[OutputMessage] = []
    output_message = OutputMessage(role="assistant", parts=[Text(_safe_str(result))])
    output_messages.append(output_message)
    return output_messages


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


class CrewAIInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentation for CrewAI using splunk-otel-util-genai.

    This instrumentor provides standardized telemetry for CrewAI workflows,
    agents, tasks, and tool executions.

    Note: LLM calls are NOT instrumented here. Use opentelemetry-instrumentation-openai
    or other provider-specific instrumentations for LLM observability.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Apply instrumentation to CrewAI components."""
        global _handler

        # Initialize TelemetryHandler with tracer provider
        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            from opentelemetry import trace

            tracer_provider = trace.get_tracer_provider()

        meter_provider = kwargs.get("meter_provider")
        if not meter_provider:
            from opentelemetry import metrics

            meter_provider = metrics.get_meter_provider()

        _handler = get_telemetry_handler(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )

        def _safe_wrap(module: str, name: str, wrapper):
            try:
                wrap_function_wrapper(module, name, wrapper)
            except (ImportError, ModuleNotFoundError):
                _logger.debug(
                    "CrewAI not importable while instrumenting (%s.%s); proceeding without wrapping.",
                    module,
                    name,
                    exc_info=True,
                )
            except Exception:
                _logger.warning(
                    "Failed to instrument CrewAI (%s.%s); proceeding without wrapping.",
                    module,
                    name,
                    exc_info=True,
                )

        # Crew.kickoff -> Workflow
        _safe_wrap("crewai.crew", "Crew.kickoff", _wrap_crew_kickoff)

        # Agent.execute_task -> AgentInvocation
        _safe_wrap("crewai.agent", "Agent.execute_task", _wrap_agent_execute_task)

        # Task.execute_sync -> Step
        _safe_wrap("crewai.task", "Task.execute_sync", _wrap_task_execute)

        # BaseTool.run -> ToolCall
        _safe_wrap("crewai.tools.base_tool", "BaseTool.run", _wrap_tool_run)

        # CrewStructuredTool.invoke -> ToolCall (for @tool decorated functions)
        _safe_wrap(
            "crewai.tools.structured_tool",
            "CrewStructuredTool.invoke",
            _wrap_structured_tool_invoke,
        )

    def _uninstrument(self, **kwargs):
        """Remove instrumentation from CrewAI components."""

        def _safe_unwrap(module: str, name: str):
            try:
                unwrap(module, name)
            except (ImportError, ModuleNotFoundError):
                _logger.debug(
                    "CrewAI not importable while uninstrumenting (%s.%s); continuing cleanup.",
                    module,
                    name,
                    exc_info=True,
                )
            except Exception:
                _logger.warning(
                    "Failed to uninstrument CrewAI (%s.%s); continuing cleanup.",
                    module,
                    name,
                    exc_info=True,
                )

        _safe_unwrap("crewai.crew.Crew", "kickoff")
        _safe_unwrap("crewai.agent.Agent", "execute_task")
        _safe_unwrap("crewai.task.Task", "execute_sync")
        _safe_unwrap("crewai.tools.base_tool.BaseTool", "run")
        _safe_unwrap("crewai.tools.structured_tool.CrewStructuredTool", "invoke")


def _wrap_crew_kickoff(wrapped, instance, args, kwargs):
    """
    Wrap Crew.kickoff to create a Workflow span.

    Maps to: Workflow type from splunk-otel-util-genai
    """
    try:
        handler = _handler

        # Create workflow invocation
        workflow = Workflow(
            name=getattr(instance, "name", None) or "CrewAI Workflow",
            workflow_type="crewai.crew",
            framework="crewai",
            system="crewai",
        )

        inputs = kwargs.get("inputs")
        if inputs is None and args:
            inputs = args[0]
        if inputs is not None:
            workflow.input_messages = _make_input_message(inputs)

        # Start the workflow
        handler.start_workflow(workflow)
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        # Capture result information
        try:
            if result:
                if hasattr(result, "raw"):
                    workflow.output_messages = _make_output_message(result.raw)

            # Stop the workflow successfully
            handler.stop_workflow(workflow)
        except Exception:
            # Ignore instrumentation errors on success path
            pass

        return result
    except Exception as exc:
        # Wrapped function failed - record error and end span
        try:
            handler.fail(workflow, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_agent_execute_task(wrapped, instance, args, kwargs):
    """
    Wrap Agent.execute_task to create an AgentInvocation span.

    Maps to: AgentInvocation type from splunk-otel-util-genai
    """
    try:
        handler = _handler

        # Create agent invocation
        agent_invocation = AgentInvocation(
            name=getattr(instance, "role", "Unknown Agent"),
            framework="crewai",
            system="crewai",
        )

        # Capture task description as input context
        task = kwargs.get("task")
        if task is None and args:
            task = args[0]
        if task is not None:
            messages: dict[str, Any] = {}
            description = getattr(task, "description", None)
            if description:
                messages["description"] = description
            expected_output = getattr(task, "expected_output", None)
            if expected_output:
                messages["expected_output"] = expected_output
            agent_invocation.input_messages = _make_input_message(messages)

        # Start the agent invocation
        handler.start_agent(agent_invocation)
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        # Capture result and metrics
        try:
            if result is not None:
                agent_invocation.output_messages = _make_output_message(result)

            # Extract token usage if available
            if hasattr(instance, "_token_process"):
                try:
                    token_summary = instance._token_process.get_summary()
                    if hasattr(token_summary, "prompt_tokens"):
                        agent_invocation.input_tokens = token_summary.prompt_tokens
                    if hasattr(token_summary, "completion_tokens"):
                        agent_invocation.output_tokens = token_summary.completion_tokens
                except Exception:
                    pass  # Ignore token extraction errors

            # Stop the agent invocation successfully
            handler.stop_agent(agent_invocation)
        except Exception:
            # Ignore instrumentation errors on success path
            pass

        return result
    except Exception as exc:
        # Wrapped function failed - record error and end span
        try:
            handler.fail(agent_invocation, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_task_execute(wrapped, instance, args, kwargs):
    """
    Wrap Task.execute_sync to create a Step span.

    Maps to: Step type from splunk-otel-util-genai
    """
    try:
        handler = _handler

        # Create step
        step = Step(
            name=getattr(instance, "description", None) or "Task Execution",
            framework="crewai",
            system="crewai",
        )

        # Set step fields from task
        if hasattr(instance, "description"):
            step.description = instance.description
        if hasattr(instance, "expected_output"):
            step.objective = instance.expected_output
        if hasattr(instance, "agent") and hasattr(instance.agent, "role"):
            step.assigned_agent = instance.agent.role

        # Start the step
        handler.start_step(step)
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        # Capture result
        try:
            # Stop the step successfully
            handler.stop_step(step)
        except Exception:
            # Ignore instrumentation errors on success path
            pass

        return result
    except Exception as exc:
        # Wrapped function failed - record error and end span
        try:
            handler.fail(step, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_tool_call(wrapped, instance, args, kwargs):
    """Shared wrapper for tool calls."""
    try:
        handler = _handler

        # Create tool call
        tool_call = ToolCall(
            name=getattr(instance, "name", "unknown_tool"),
            arguments=str(kwargs) if kwargs else "{}",
            id=str(id(instance)),
            framework="crewai",
            system="crewai",
        )

        # Start the tool call
        handler.start_tool_call(tool_call)
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)

        # Stop the tool call successfully
        try:
            handler.stop_tool_call(tool_call)
        except Exception:
            # Ignore instrumentation errors on success path
            pass

        return result
    except Exception as exc:
        # Wrapped function failed - record error and end span
        try:
            handler.fail(tool_call, Error(message=str(exc), type=type(exc)))
        except Exception:
            pass
        raise


def _wrap_tool_run(wrapped, instance, args, kwargs):
    """
    Wrap BaseTool.run to create a ToolCall span.

    Maps to: ToolCall type from splunk-otel-util-genai
    """
    return _wrap_tool_call(wrapped, instance, args, kwargs)


def _wrap_structured_tool_invoke(wrapped, instance, args, kwargs):
    """
    Wrap CrewStructuredTool.invoke to create a ToolCall span.

    This handles tools created with the @tool decorator.
    Maps to: ToolCall type from splunk-otel-util-genai
    """
    return _wrap_tool_call(wrapped, instance, args, kwargs)
