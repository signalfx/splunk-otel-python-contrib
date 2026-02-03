"""
Workflow-based agent instrumentation for LlamaIndex.

This module provides instrumentation for Workflow-based agents (ReActAgent, FunctionAgent, etc.)
by intercepting workflow event streams to capture tool calls.
"""

import asyncio

from opentelemetry import trace as _trace_mod
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
                            input_messages=[
                                InputMessage(
                                    role="user",
                                    parts=[Text(content=str(event.input[-1].content))],
                                )
                            ]
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
                    # The span emitter sets "invoke_workflow" for workflow spans
                    if op_name == "invoke_workflow":
                        inside_workflow = True
            except Exception:
                pass  # If we can't determine, assume not inside workflow

        # Create workflow span only for standalone agents (not inside another workflow)
        # Standalone agent: workflow ReActAgent → agent.ReActAgent
        # Inside TravelPlannerWorkflow: Just agent.ReActAgent (nested under parent)
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

        # Create agent invocation (always)
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

        # Start the agent span (automatically becomes child of active span)
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

            def __init__(self, original, agent_span, workflow_span=None):
                self._original = original
                self._current_agent = agent_span
                self._root_workflow = (
                    workflow_span  # May be None if inside another workflow
                )
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
                    self._current_agent.output_messages = [
                        OutputMessage(
                            role="assistant", parts=[Text(content=str(self._result))]
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

                    # Stop agent span
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

        handler = InstrumentedHandler(original_handler, current_agent, root_workflow)

    return handler


def wrap_workflow_run(wrapped, instance, args, kwargs):
    """
    Wrap Workflow.run() to instrument custom workflow subclasses.

    This creates a Workflow span for custom workflow classes like TravelPlannerWorkflow
    that orchestrate multiple agents. The workflow span becomes the parent context,
    so any agents called within will be nested as children.

    Hierarchy for custom workflows:
        Workflow (TravelPlannerWorkflow.run)
          ├─ AgentInvocation (research_agent)
          │   └─ LLMInvocation
          ├─ AgentInvocation (analysis_agent)
          │   └─ LLMInvocation
          └─ AgentInvocation (synthesis_agent)
              └─ LLMInvocation

    Note: This does NOT instrument ReActAgent/FunctionAgent which inherit from Workflow
    but are handled by wrap_agent_run. We detect agent subclasses and skip them here.
    """
    from llama_index.core.agent.workflow import AgentWorkflow

    # Skip if this is an agent (ReActAgent, FunctionAgent, etc.)
    # Those are handled by wrap_agent_run which creates both Workflow + Agent spans
    try:
        # Check if instance is a subclass of AgentWorkflow (which includes ReActAgent, FunctionAgent)
        if isinstance(instance, AgentWorkflow):
            # Let wrap_agent_run handle this
            return wrapped(*args, **kwargs)
    except Exception:
        pass  # If we can't determine, proceed with workflow instrumentation

    # Get user message from args/kwargs
    user_msg = kwargs.get("user_msg") or (args[0] if args else None)

    # Find the callback handler to get the telemetry handler
    telemetry_handler = None
    try:
        from opentelemetry.instrumentation.llamaindex.callback_handler import (
            LlamaindexCallbackHandler,
        )

        # Look for our callback handler in the global Settings
        from llama_index.core import Settings

        if Settings.callback_manager:
            for handler in Settings.callback_manager.handlers:
                if isinstance(handler, LlamaindexCallbackHandler):
                    telemetry_handler = handler._handler
                    break
    except Exception:
        pass

    if not telemetry_handler:
        # No telemetry handler found, just call the original
        return wrapped(*args, **kwargs)

    # Create the workflow span for this custom workflow
    input_messages = (
        [InputMessage(role="user", parts=[Text(content=str(user_msg))])]
        if user_msg
        else []
    )
    workflow = Workflow(
        name=type(instance).__name__,
        workflow_type="llamaindex.workflow",
        framework="llamaindex",
        input_messages=input_messages,
        attributes={},
    )

    # Start the workflow span - this becomes the parent context for nested agents
    telemetry_handler.start_workflow(workflow)

    # Call the original run() method
    original_handler = wrapped(*args, **kwargs)

    # Create an instrumented handler that will close the workflow span when awaited
    class InstrumentedWorkflowHandler:
        """Wrapper that closes the workflow span when the result is awaited."""

        def __init__(self, original, workflow_obj):
            self._original = original
            self._workflow = workflow_obj
            self._result = None

        def __await__(self):
            return self._await_impl().__await__()

        async def _await_impl(self):
            """Actual async implementation."""
            try:
                self._result = await self._original
                self._workflow.final_output = str(self._result)
                # Set output_messages for content events
                self._workflow.output_messages = [
                    OutputMessage(
                        role="assistant", parts=[Text(content=str(self._result))]
                    )
                ]
                telemetry_handler.stop_workflow(self._workflow)
            except Exception as e:
                try:
                    from opentelemetry.util.genai.types import Error

                    error = Error(message=str(e), type=type(e))
                    telemetry_handler.fail_workflow(self._workflow, error)
                except Exception as telemetry_error:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to record telemetry error: {telemetry_error}"
                    )
                raise
            return self._result

        def __getattr__(self, name):
            return getattr(self._original, name)

    return InstrumentedWorkflowHandler(original_handler, workflow)
