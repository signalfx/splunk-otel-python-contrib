"""
Workflow-based agent instrumentation for LlamaIndex.

This module provides instrumentation for Workflow-based agents (ReActAgent, FunctionAgent, etc.)
by intercepting workflow event streams to capture tool calls.
"""

import asyncio
from collections.abc import Mapping

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    Workflow,
)


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
        self._agent_scope = str(getattr(current_agent, "run_id", id(current_agent)))
        self._active_agents = {}
        self._current_agent_key = None

        try:
            async for event in workflow_handler.stream_events():
                if isinstance(event, AgentInput):
                    if (
                        self._current_agent
                        and event.current_agent_name
                        and event.current_agent_name != self._current_agent.agent_name
                    ):
                        agent_name = event.current_agent_name
                        agent_key = (self._agent_scope, agent_name)
                        self._current_agent_key = agent_key
                        if agent_key not in self._active_agents:
                            agent_invocation = AgentInvocation(
                                name=f"agent.{agent_name}",
                                input_messages=[
                                    InputMessage(
                                        role="user",
                                        parts=[
                                            Text(content=str(event.input[-1].content))
                                        ],
                                    )
                                ]
                                if event.input
                                else [],
                                attributes={},
                            )
                            agent_invocation.framework = "llamaindex"
                            agent_invocation.agent_name = agent_name
                            if (
                                hasattr(self._current_agent, "span")
                                and self._current_agent.span
                            ):
                                agent_invocation.parent_span = self._current_agent.span
                            elif (
                                hasattr(self._current_agent, "parent_span")
                                and self._current_agent.parent_span
                            ):
                                agent_invocation.parent_span = (
                                    self._current_agent.parent_span
                                )
                            self._handler.start_agent(agent_invocation)
                            self._active_agents[agent_key] = agent_invocation

                elif isinstance(event, AgentOutput):
                    if (
                        self._current_agent
                        and event.current_agent_name
                        and event.current_agent_name != self._current_agent.agent_name
                    ):
                        agent_name = event.current_agent_name
                        agent_key = (self._agent_scope, agent_name)
                        self._current_agent_key = agent_key
                        agent_invocation = self._active_agents.get(agent_key)
                        if agent_invocation:
                            agent_invocation.output_result = str(event.response.content)
                            agent_invocation.output_messages = [
                                OutputMessage(
                                    role="assistant",
                                    parts=[Text(content=str(event.response.content))],
                                )
                            ]

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
                    if self._current_agent_key:
                        active_agent = self._active_agents.get(self._current_agent_key)
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

            for agent_invocation in list(self._active_agents.values()):
                self._handler.stop_agent(agent_invocation)
            self._active_agents.clear()

        except Exception as e:
            # Try to clean up any active tool spans on error, but don't let
            # instrumentation failures interfere with the application
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error during workflow event instrumentation: {e}")

            try:
                from opentelemetry.util.genai.types import Error

                for tool_call in list(self._active_tools.values()):
                    error = Error(message=str(e), type=type(e))
                    self._handler.fail_tool_call(tool_call, error)
                self._active_tools.clear()
                for agent_invocation in list(self._active_agents.values()):
                    error = Error(message=str(e), type=type(e))
                    self._handler.fail_agent(agent_invocation, error)
                self._active_agents.clear()
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up tool spans after error: {cleanup_error}"
                )


def wrap_agent_run(wrapped, instance, args, kwargs):
    """
    Wrap agent.run() to instrument workflow events.

    For standalone agents (not inside a workflow), creates:
        Workflow (workflow ReActAgent)
          └─ AgentInvocation (invoke_agent agent.ReActAgent)
              ├─ LLMInvocation (reasoning)
              ├─ ToolCall (tool execution)
              └─ LLMInvocation (final response)
    """
    from opentelemetry import trace as trace_api

    # Get the initial user message
    user_msg = kwargs.get("user_msg") or (args[0] if args else "")

    # Get TelemetryHandler from callback handler if available
    from llama_index.core import Settings

    telemetry_handler = None
    for callback_handler in Settings.callback_manager.handlers:
        if hasattr(callback_handler, "_handler"):
            telemetry_handler = callback_handler._handler
            break

    # Create workflow and agent spans
    root_workflow = None
    current_agent = None
    if telemetry_handler:
        # Check if we're already inside a workflow span to avoid creating
        # duplicate workflow spans for nested/parallel agent execution.
        current_span = trace_api.get_current_span()
        inside_workflow = False

        if current_span and current_span.is_recording():
            try:
                op_name = None
                span_attrs = getattr(current_span, "attributes", None)
                if isinstance(span_attrs, dict):
                    op_name = span_attrs.get("gen_ai.operation.name")
                if op_name is None and hasattr(current_span, "_attributes"):
                    op_name = current_span._attributes.get("gen_ai.operation.name")
                inside_workflow = op_name == "invoke_workflow"
            except Exception:
                pass  # If we can't determine, assume not inside workflow

        # Create workflow span only for standalone agents (not inside another workflow)
        if not inside_workflow:
            input_messages = (
                [InputMessage(role="user", parts=[Text(content=str(user_msg))])]
                if user_msg
                else []
            )
            root_workflow = Workflow(
                name=f"{type(instance).__name__}",
                workflow_type="llamaindex.workflow",
                framework="llamaindex",
                input_messages=input_messages,
                attributes={},
            )
            telemetry_handler.start_workflow(root_workflow)

        # Create agent invocation
        agent_input_messages = (
            [InputMessage(role="user", parts=[Text(content=str(user_msg))])]
            if user_msg
            else []
        )
        current_agent = AgentInvocation(
            name=f"agent.{type(instance).__name__}",
            input_messages=agent_input_messages,
            attributes={},
        )
        current_agent.framework = "llamaindex"
        current_agent.agent_name = type(instance).__name__

        is_orchestrator_workflow = bool(
            hasattr(instance, "agents")
            and hasattr(instance, "root_agent")
            and isinstance(getattr(instance, "agents", None), (list, tuple, Mapping))
        )
        if is_orchestrator_workflow and root_workflow and root_workflow.span:
            current_agent.parent_span = root_workflow.span

        # Start the agent span for non-orchestrator workflow instances.
        if not is_orchestrator_workflow:
            telemetry_handler.start_agent(current_agent)

    # Call the original run() method to get the workflow handler
    handler = wrapped(*args, **kwargs)

    if telemetry_handler and current_agent:
        # Create workflow instrumentor for tool tracking
        instrumentor = WorkflowEventInstrumentor(telemetry_handler)

        # Wrap the handler to close agent and workflow spans when complete
        original_handler = handler

        class InstrumentedHandler:
            """Wrapper that closes agent and workflow spans when workflow completes."""

            def __init__(
                self,
                original,
                agent_span,
                workflow_span=None,
                is_orchestrator_workflow=False,
            ):
                self._original = original
                self._current_agent = agent_span
                self._root_workflow = (
                    workflow_span  # May be None if inside another workflow
                )
                self._result = None
                self._is_orchestrator_workflow = is_orchestrator_workflow

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
                    if not self._is_orchestrator_workflow:
                        self._current_agent.output_result = str(self._result)
                        self._current_agent.output_messages = [
                            OutputMessage(
                                role="assistant",
                                parts=[Text(content=str(self._result))],
                            )
                        ]

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

                    # Stop agent span for non-orchestrator workflow instances
                    if not self._is_orchestrator_workflow:
                        telemetry_handler.stop_agent(self._current_agent)

                    # Stop workflow span if we created one (standalone agent case)
                    if self._root_workflow:
                        self._root_workflow.output_result = str(self._result)
                        self._root_workflow.output_messages = [
                            OutputMessage(
                                role="assistant",
                                parts=[Text(content=str(self._result))],
                            )
                        ]
                        telemetry_handler.stop_workflow(self._root_workflow)

                except Exception as e:
                    # Try to record the failure in telemetry, but don't let instrumentation
                    # errors interfere with the application's exception handling
                    try:
                        from opentelemetry.util.genai.types import Error

                        error = Error(message=str(e), type=type(e))
                        if not self._is_orchestrator_workflow:
                            telemetry_handler.fail_agent(self._current_agent, error)
                        # Also fail the workflow if we created one
                        if self._root_workflow:
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

        handler = InstrumentedHandler(
            original_handler,
            current_agent,
            root_workflow,
            is_orchestrator_workflow,
        )

    return handler
