"""
Workflow-based agent instrumentation for LlamaIndex.

This module provides instrumentation for Workflow-based agents (ReActAgent, etc.)
by intercepting workflow event streams to capture agent steps and tool calls.
"""

import asyncio
from typing import Any, Optional
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
    
    This function wraps the run() method of Workflow-based agents to capture
    agent steps and tool calls via workflow event streaming.
    """
    handler = wrapped(*args, **kwargs)
    
    # Get the initial user message
    user_msg = kwargs.get('user_msg') or (args[0] if args else "")
    
    # Get TelemetryHandler from callback handler if available
    from llama_index.core import Settings
    telemetry_handler = None
    for callback_handler in Settings.callback_manager.handlers:
        if hasattr(callback_handler, '_handler'):
            telemetry_handler = callback_handler._handler
            break
    
    if telemetry_handler:
        # Create workflow instrumentor
        instrumentor = WorkflowEventInstrumentor(telemetry_handler)
        
        # Start background task to stream events
        async def stream_events_background():
            try:
                await instrumentor.instrument_workflow_handler(handler, str(user_msg))
            except Exception as e:
                # Log error but don't crash the workflow
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error instrumenting workflow events: {e}")
        
        # Launch background task
        asyncio.create_task(stream_events_background())
    
    return handler
