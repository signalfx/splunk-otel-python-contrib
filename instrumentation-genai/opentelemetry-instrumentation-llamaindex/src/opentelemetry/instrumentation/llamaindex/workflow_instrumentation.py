"""
Workflow-based agent instrumentation for LlamaIndex.

This module provides instrumentation for Workflow-based agents (ReActAgent, FunctionAgent, etc.)
by intercepting workflow event streams to capture tool calls.
"""

import asyncio

from opentelemetry import trace as _trace_mod
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import AgentInvocation, ToolCall, Workflow


class WorkflowEventInstrumentor:
    """Instrumentor that wraps WorkflowHandler to capture tool events.

    Each agent.run() call creates a new instance of this instrumentor to track
    tool calls for that specific agent. In multi-agent scenarios, multiple
    instrumentor instances coexist, each tracking its own agent.
    """

    def __init__(self, handler: TelemetryHandler):
        self._handler = handler
        self._active_tools = {}  # tool_id -> ToolCall
        self._current_agent = (
            None  # The agent being tracked by this instrumentor instance
        )

    async def instrument_workflow_handler(
        self, workflow_handler, initial_message: str, current_agent: AgentInvocation
    ):
        """
        Instrument a WorkflowHandler by streaming its events and creating telemetry spans.

        Args:
            workflow_handler: The WorkflowHandler returned by agent.run()
            initial_message: The user's initial message to the agent
            current_agent: The AgentInvocation span for the agent being tracked.
                          Tool calls will be associated with this agent as their parent.
        """
        from llama_index.core.agent.workflow.workflow_events import (
            AgentInput,
            AgentOutput,
            ToolCall as WorkflowToolCall,
            ToolCallResult,
        )

        self._current_agent = current_agent
        self._active_agents = {}
        self._current_agent_name = None

        try:
            async for event in workflow_handler.stream_events():
                if isinstance(event, AgentInput):
                    if (
                        self._current_agent
                        and self._current_agent.agent_name == "AgentWorkflow"
                    ):
                        agent_name = event.current_agent_name
                        self._current_agent_name = agent_name
                        agent_invocation = AgentInvocation(
                            name=f"agent.{agent_name}",
                            input_context=str(event.input[-1].content)
                            if event.input
                            else "",
                            attributes={},
                        )
                        agent_invocation.framework = "llamaindex"
                        agent_invocation.agent_name = agent_name
                        if (
                            hasattr(self._current_agent, "span")
                            and self._current_agent.span
                        ):
                            agent_invocation.parent_span = self._current_agent.span
                        self._handler.start_agent(agent_invocation)
                        self._active_agents[agent_name] = agent_invocation

                elif isinstance(event, AgentOutput):
                    if (
                        self._current_agent
                        and self._current_agent.agent_name == "AgentWorkflow"
                    ):
                        agent_name = event.current_agent_name
                        self._current_agent_name = agent_name
                        agent_invocation = self._active_agents.get(agent_name)
                        if agent_invocation:
                            agent_invocation.output_result = str(event.response.content)
                            self._handler.stop_agent(agent_invocation)
                            del self._active_agents[agent_name]

                # Tool call start
                if isinstance(event, WorkflowToolCall):
                    tool_call = ToolCall(
                        arguments=event.tool_kwargs,
                        name=event.tool_name,
                        id=event.tool_id,
                        attributes={},
                    )
                    tool_call.framework = "llamaindex"

                    active_agent = None
                    if self._current_agent_name:
                        active_agent = self._active_agents.get(self._current_agent_name)
                    if active_agent:
                        tool_call.agent_name = active_agent.agent_name
                        tool_call.agent_id = str(active_agent.run_id)
                        if hasattr(active_agent, "span") and active_agent.span:
                            tool_call.parent_span = active_agent.span
                    elif self._current_agent:
                        tool_call.agent_name = self._current_agent.agent_name
                        tool_call.agent_id = str(self._current_agent.run_id)
                        if (
                            hasattr(self._current_agent, "span")
                            and self._current_agent.span
                        ):
                            tool_call.parent_span = self._current_agent.span

                    self._handler.start_tool_call(tool_call)
                    self._active_tools[event.tool_id] = tool_call

                # Tool call end
                elif isinstance(event, ToolCallResult):
                    tool_call = self._active_tools.get(event.tool_id)
                    if tool_call:
                        # Extract result
                        result = event.tool_output
                        if hasattr(result, "content"):
                            tool_call.result = str(result.content)
                        else:
                            tool_call.result = str(result)

                        self._handler.stop_tool_call(tool_call)
                        del self._active_tools[event.tool_id]

        except Exception as e:
            # Try to clean up any active tool spans on error, but don't let
            # instrumentation failures interfere with the application
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error during workflow event instrumentation: {e}")

            try:
                for tool_call in list(self._active_tools.values()):
                    from opentelemetry.util.genai.types import Error

                    error = Error(message=str(e), type=type(e))
                    self._handler.fail_tool_call(tool_call, error)
                self._active_tools.clear()
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up tool spans after error: {cleanup_error}"
                )


def wrap_agent_run(wrapped, instance, args, kwargs):
    """
    Wrap agent.run() to instrument workflow events.

    This creates a Workflow span as the root (since ReActAgent inherits from Workflow),
    then creates an AgentInvocation span nested inside it, establishing a hierarchy
    where workflows orchestrate agents.

    Hierarchy:
        Workflow (agent.run - workflow orchestration layer)
          └─ AgentInvocation (agent execution layer)
              ├─ LLMInvocation (reasoning)
              ├─ ToolCall (tool execution)
              └─ LLMInvocation (final response)
    """
    # Get the initial user message
    user_msg = kwargs.get("user_msg") or (args[0] if args else "")

    # Get TelemetryHandler from callback handler if available
    from llama_index.core import Settings

    telemetry_handler = None
    for callback_handler in Settings.callback_manager.handlers:
        if hasattr(callback_handler, "_handler"):
            telemetry_handler = callback_handler._handler
            break

    # Create workflow and agent spans to establish proper hierarchy
    root_workflow = None
    current_agent = None
    if telemetry_handler:
        # Check if we're already inside a workflow span (multi-agent orchestration)
        # by examining the current span's attributes for gen_ai.operation.name=workflow
        current_span = _trace_mod.get_current_span()
        inside_workflow = False

        if current_span and current_span.is_recording():
            # Check if parent span is a workflow by looking for workflow operation attribute
            # This prevents creating nested workflow spans when agents are orchestrated
            try:
                # Access span attributes (implementation-specific, but standard in SDK)
                if hasattr(current_span, "attributes"):
                    op_name = current_span.attributes.get("gen_ai.operation.name")
                    if op_name == "workflow":
                        inside_workflow = True
            except Exception:
                pass  # If we can't determine, assume not inside workflow

        # Level 1: Create root workflow span only if NOT inside another workflow
        # This supports multi-agent orchestration where a custom Workflow orchestrates
        # multiple ReActAgent instances, creating proper hierarchy:
        #   CustomWorkflow
        #     ├─ Agent 1 (research)
        #     ├─ Agent 2 (analysis)
        #     └─ Agent 3 (synthesis)
        if not inside_workflow:
            root_workflow = Workflow(
                name=f"{type(instance).__name__} Workflow",
                workflow_type="llamaindex.workflow",
                framework="llamaindex",
                initial_input=str(user_msg),
                attributes={},
            )

            # Start the workflow span - this becomes the active span context
            telemetry_handler.start_workflow(root_workflow)

        # Level 2: Create agent invocation (nested inside workflow if one exists)
        # The agent span will automatically become a child of the active span
        # (either our workflow or parent workflow) due to OpenTelemetry's context propagation
        current_agent = AgentInvocation(
            name=f"agent.{type(instance).__name__}",
            input_context=str(user_msg),
            attributes={},
        )
        current_agent.framework = "llamaindex"
        current_agent.agent_name = type(instance).__name__

        # Start the agent span (automatically becomes child of active span)
        # This pushes (agent_name, run_id) onto the _agent_context_stack
        # and stores the span in _span_registry[run_id]
        telemetry_handler.start_agent(current_agent)

    # Call the original run() method to get the workflow handler
    handler = wrapped(*args, **kwargs)

    if telemetry_handler and current_agent:
        # Create workflow instrumentor for tool tracking
        instrumentor = WorkflowEventInstrumentor(telemetry_handler)

        # Wrap the handler to close agent (and workflow if created) spans when complete
        original_handler = handler

        class InstrumentedHandler:
            """Wrapper that closes agent and workflow spans when workflow completes."""

            def __init__(self, original, workflow_span, agent_span):
                self._original = original
                self._root_workflow = workflow_span
                self._current_agent = agent_span
                self._result = None

            def __await__(self):
                # Start background task to instrument workflow events
                async def stream_events():
                    try:
                        await instrumentor.instrument_workflow_handler(
                            self._original, str(user_msg), self._current_agent
                        )
                    except Exception as e:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(f"Error instrumenting workflow events: {e}")

                # Store the task so we can await it if needed
                self._stream_task = asyncio.create_task(stream_events())

                # Wait for the actual workflow to complete and return the result
                return self._await_impl().__await__()

            async def _await_impl(self):
                """Actual async implementation."""
                try:
                    self._result = await self._original
                    self._current_agent.output_result = str(self._result)
                    if self._root_workflow:  # Only set if we created a workflow span
                        self._root_workflow.final_output = str(self._result)

                    # Wait for the event streaming task to complete to ensure all
                    # inner agent invocations are properly closed
                    if hasattr(self, "_stream_task"):
                        try:
                            await asyncio.wait_for(self._stream_task, timeout=5.0)
                        except asyncio.TimeoutError:
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.warning(
                                "Workflow event streaming did not complete within timeout"
                            )
                        except Exception:
                            pass  # Already logged in stream_events()

                    # Stop spans in reverse order: agent first, then workflow (if we created it)
                    telemetry_handler.stop_agent(self._current_agent)
                    if self._root_workflow:  # Only stop if we created a workflow span
                        telemetry_handler.stop_workflow(self._root_workflow)
                except Exception as e:
                    # Try to record the failure in telemetry, but don't let instrumentation
                    # errors interfere with the application's exception handling
                    try:
                        from opentelemetry.util.genai.types import Error

                        error = Error(message=str(e), type=type(e))
                        # Fail spans in reverse order: agent first, then workflow (if we created it)
                        telemetry_handler.fail_agent(self._current_agent, error)
                        if (
                            self._root_workflow
                        ):  # Only fail if we created a workflow span
                            telemetry_handler.fail_workflow(self._root_workflow, error)
                    except Exception as telemetry_error:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Failed to record telemetry error: {telemetry_error}"
                        )
                    # Always re-raise the original application exception
                    raise
                return self._result

            def __getattr__(self, name):
                # Delegate all other attributes to the original handler
                return getattr(self._original, name)

        handler = InstrumentedHandler(original_handler, root_workflow, current_agent)

    return handler
