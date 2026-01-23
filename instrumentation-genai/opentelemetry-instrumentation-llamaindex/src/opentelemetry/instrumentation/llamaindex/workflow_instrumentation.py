"""
Workflow-based agent instrumentation for LlamaIndex.

This module provides instrumentation for Workflow-based agents (ReActAgent, etc.)
by intercepting workflow event streams to capture tool calls.
"""

import asyncio

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import AgentInvocation, ToolCall


class WorkflowEventInstrumentor:
    """Instrumentor that wraps WorkflowHandler to capture tool events."""

    def __init__(self, handler: TelemetryHandler):
        self._handler = handler
        self._active_tools = {}  # tool_id -> ToolCall
        self._root_agent = None  # Reference to the root agent span

    async def instrument_workflow_handler(
        self, workflow_handler, initial_message: str, root_agent: AgentInvocation
    ):
        """
        Instrument a WorkflowHandler by streaming its events and creating telemetry spans.

        Args:
            workflow_handler: The WorkflowHandler returned by agent.run()
            initial_message: The user's initial message to the agent
            root_agent: The root AgentInvocation span for context
        """
        from llama_index.core.agent.workflow.workflow_events import (
            ToolCall as WorkflowToolCall,
            ToolCallResult,
        )

        self._root_agent = root_agent

        try:
            async for event in workflow_handler.stream_events():
                # Tool call start
                if isinstance(event, WorkflowToolCall):
                    tool_call = ToolCall(
                        arguments=event.tool_kwargs,
                        name=event.tool_name,
                        id=event.tool_id,
                        attributes={},
                    )
                    tool_call.framework = "llamaindex"

                    # Associate with root agent
                    if self._root_agent:
                        tool_call.agent_name = self._root_agent.agent_name
                        tool_call.agent_id = str(self._root_agent.run_id)
                        # Set parent_span explicitly - the agent span is the parent of this tool
                        if hasattr(self._root_agent, "span") and self._root_agent.span:
                            tool_call.parent_span = self._root_agent.span

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
            # Clean up any active tool spans on error
            for tool_call in list(self._active_tools.values()):
                from opentelemetry.util.genai.types import Error

                error = Error(message=str(e), type=type(e))
                self._handler.fail_tool_call(tool_call, error)
            self._active_tools.clear()
            raise


def wrap_agent_run(wrapped, instance, args, kwargs):
    """
    Wrap agent.run() to instrument workflow events.

    This creates a root agent span immediately when agent.run() is called,
    ensuring all LLM and tool calls inherit the same trace context.

    The root span is pushed onto the agent_context_stack, which allows the
    callback handler to retrieve it when LLM/embedding events occur.
    """
    # Get the initial user message
    user_msg = kwargs.get("user_msg") or (args[0] if args else "")

    # Get TelemetryHandler from callback handler if available
    from llama_index.core import Settings
    from opentelemetry.util.genai.types import AgentInvocation
    from opentelemetry import context

    telemetry_handler = None
    for callback_handler in Settings.callback_manager.handlers:
        if hasattr(callback_handler, "_handler"):
            telemetry_handler = callback_handler._handler
            break

    # Create a root agent span immediately to ensure all subsequent calls
    # (LLM, tools) inherit this trace context
    root_agent = None
    parent_context = None
    if telemetry_handler:
        # Create root agent invocation before workflow starts
        root_agent = AgentInvocation(
            name=f"agent.{type(instance).__name__}",
            input_context=str(user_msg),
            attributes={},
        )
        root_agent.framework = "llamaindex"
        root_agent.agent_name = type(instance).__name__

        # Start the root agent span immediately
        # This pushes (agent_name, run_id) onto the _agent_context_stack
        # and stores the span in _span_registry[run_id]
        telemetry_handler.start_agent(root_agent)

        # Capture the current context (which includes the active span)
        # so we can propagate it to async tasks
        parent_context = context.get_current()

    # Call the original run() method to get the workflow handler
    handler = wrapped(*args, **kwargs)

    if telemetry_handler and root_agent and parent_context:
        # Create workflow instrumentor for tool tracking
        instrumentor = WorkflowEventInstrumentor(telemetry_handler)

        # Wrap the handler to close the root span when the workflow completes
        original_handler = handler

        class InstrumentedHandler:
            """Wrapper that closes the root agent span when workflow completes."""

            def __init__(self, original, root_span_agent, ctx):
                self._original = original
                self._root_agent = root_span_agent
                self._result = None
                self._parent_context = ctx

            def __await__(self):
                # Start background task to instrument workflow events
                async def stream_events():
                    try:
                        # Attach the parent context before processing workflow events
                        token = context.attach(self._parent_context)
                        try:
                            await instrumentor.instrument_workflow_handler(
                                self._original, str(user_msg), self._root_agent
                            )
                        finally:
                            context.detach(token)
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
                # Attach the parent context to ensure proper span hierarchy
                token = context.attach(self._parent_context)
                try:
                    self._result = await self._original
                    self._root_agent.output_result = str(self._result)

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

                    telemetry_handler.stop_agent(self._root_agent)
                except Exception as e:
                    from opentelemetry.util.genai.types import Error

                    telemetry_handler.fail_agent(
                        self._root_agent, Error(message=str(e), type=type(e))
                    )
                    raise
                finally:
                    context.detach(token)
                return self._result

            def __getattr__(self, name):
                # Delegate all other attributes to the original handler
                return getattr(self._original, name)

        handler = InstrumentedHandler(original_handler, root_agent, parent_context)

    return handler
