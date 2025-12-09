"""
Workflow-based agent instrumentation for LlamaIndex.

This module provides instrumentation for Workflow-based agents (ReActAgent, etc.)
by intercepting workflow event streams to capture agent steps and tool calls.
"""

import asyncio
from uuid import uuid4

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import AgentInvocation, ToolCall


class WorkflowEventInstrumentor:
    """Instrumentor that wraps WorkflowHandler to capture agent and tool events."""
    
    def __init__(self, handler: TelemetryHandler):
        self._handler = handler
        self._active_agents = {}  # event_id -> AgentInvocation
        self._active_tools = {}   # tool_id -> ToolCall
    
    async def instrument_workflow_handler(self, workflow_handler, initial_message: str):
        """
        Instrument a WorkflowHandler by streaming its events and creating telemetry spans.
        
        Args:
            workflow_handler: The WorkflowHandler returned by agent.run()
            initial_message: The user's initial message to the agent
        """
        from llama_index.core.agent.workflow.workflow_events import (
            AgentInput,
            AgentOutput,
            ToolCall as WorkflowToolCall,
            ToolCallResult,
        )
        
        agent_invocation = None
        agent_run_id = None
        
        try:
            async for event in workflow_handler.stream_events():
                # Agent step start
                if isinstance(event, AgentInput):
                    # Start a new agent invocation
                    agent_run_id = str(uuid4())
                    agent_invocation = AgentInvocation(
                        name=f"agent.{event.current_agent_name}",
                        run_id=agent_run_id,
                        input_context=str(event.input) if hasattr(event, 'input') and event.input else initial_message,
                        attributes={},
                    )
                    agent_invocation.framework = "llamaindex"
                    agent_invocation.agent_name = event.current_agent_name
                    
                    self._handler.start_agent(agent_invocation)
                    self._active_agents[agent_run_id] = agent_invocation
                
                # Tool call start
                elif isinstance(event, WorkflowToolCall):
                    tool_call = ToolCall(
                        arguments=event.tool_kwargs,
                        name=event.tool_name,
                        id=event.tool_id,
                        attributes={},
                    )
                    tool_call.framework = "llamaindex"
                    
                    # Associate with current agent if available
                    if agent_invocation:
                        tool_call.agent_name = agent_invocation.agent_name
                        tool_call.agent_id = str(agent_invocation.run_id)
                        # Set parent_span explicitly - the agent span is the parent of this tool
                        if hasattr(agent_invocation, 'span') and agent_invocation.span:
                            tool_call.parent_span = agent_invocation.span
                    
                    self._handler.start_tool_call(tool_call)
                    self._active_tools[event.tool_id] = tool_call
                
                # Tool call end
                elif isinstance(event, ToolCallResult):
                    tool_call = self._active_tools.get(event.tool_id)
                    if tool_call:
                        # Extract result
                        result = event.tool_output
                        if hasattr(result, 'content'):
                            tool_call.result = str(result.content)
                        else:
                            tool_call.result = str(result)
                        
                        self._handler.stop_tool_call(tool_call)
                        del self._active_tools[event.tool_id]
                
                # Agent step end (when no more tools to call)
                elif isinstance(event, AgentOutput):
                    # Check if this is the final output (no tool calls)
                    if not event.tool_calls and agent_invocation:
                        # Extract response
                        if hasattr(event.response, 'content'):
                            agent_invocation.output_result = str(event.response.content)
                        else:
                            agent_invocation.output_result = str(event.response)
                        
                        self._handler.stop_agent(agent_invocation)
                        if agent_run_id:
                            del self._active_agents[agent_run_id]
                        agent_invocation = None
                        agent_run_id = None
        
        except Exception as e:
            # Clean up any active spans on error
            for tool_call in list(self._active_tools.values()):
                from opentelemetry.util.genai.types import Error
                error = Error(message=str(e), type=type(e))
                self._handler.fail_tool_call(tool_call, error)
            self._active_tools.clear()
            
            if agent_invocation:
                from opentelemetry.util.genai.types import Error
                error = Error(message=str(e), type=type(e))
                self._handler.fail_agent(agent_invocation, error)
                if agent_run_id:
                    del self._active_agents[agent_run_id]
            
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
    user_msg = kwargs.get('user_msg') or (args[0] if args else "")
    
    # Get TelemetryHandler from callback handler if available
    from llama_index.core import Settings
    from opentelemetry.util.genai.types import AgentInvocation
    
    telemetry_handler = None
    for callback_handler in Settings.callback_manager.handlers:
        if hasattr(callback_handler, '_handler'):
            telemetry_handler = callback_handler._handler
            break
    
    # Create a root agent span immediately to ensure all subsequent calls
    # (LLM, tools) inherit this trace context
    root_agent = None
    if telemetry_handler:
        from uuid import uuid4
        
        # Create root agent invocation before workflow starts
        root_agent = AgentInvocation(
            name=f"agent.{type(instance).__name__}",
            run_id=str(uuid4()),
            input_context=str(user_msg),
            attributes={},
        )
        root_agent.framework = "llamaindex"
        root_agent.agent_name = type(instance).__name__
        
        # Start the root agent span immediately
        # This pushes (agent_name, run_id) onto the _agent_context_stack
        # and stores the span in _span_registry[run_id]
        telemetry_handler.start_agent(root_agent)
    
    # Call the original run() method to get the workflow handler
    handler = wrapped(*args, **kwargs)
    
    if telemetry_handler and root_agent:
        # Create workflow instrumentor for detailed step tracking
        instrumentor = WorkflowEventInstrumentor(telemetry_handler)
        
        # Wrap the handler to close the root span when the workflow completes
        original_handler = handler
        
        class InstrumentedHandler:
            """Wrapper that closes the root agent span when workflow completes."""
            def __init__(self, original, root_span_agent):
                self._original = original
                self._root_agent = root_span_agent
                self._result = None
                
            def __await__(self):
                # Start background task to instrument workflow events
                async def stream_events():
                    try:
                        await instrumentor.instrument_workflow_handler(
                            self._original, str(user_msg)
                        )
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Error instrumenting workflow events: {e}")
                
                asyncio.create_task(stream_events())
                
                # Wait for the actual workflow to complete and return the result
                return self._await_impl().__await__()
            
            async def _await_impl(self):
                """Actual async implementation."""
                try:
                    self._result = await self._original
                    self._root_agent.output_result = str(self._result)
                    telemetry_handler.stop_agent(self._root_agent)
                except Exception as e:
                    from opentelemetry.util.genai.types import Error
                    telemetry_handler.fail_agent(
                        self._root_agent,
                        Error(message=str(e), type=type(e))
                    )
                    raise
                return self._result
            
            def __getattr__(self, name):
                # Delegate all other attributes to the original handler
                return getattr(self._original, name)
        
        handler = InstrumentedHandler(original_handler, root_agent)
    
    return handler
