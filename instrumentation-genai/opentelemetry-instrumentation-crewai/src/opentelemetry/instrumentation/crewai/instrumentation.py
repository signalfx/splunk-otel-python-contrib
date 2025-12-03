"""
OpenTelemetry CrewAI Instrumentation

Wrapper-based instrumentation for CrewAI using splunk-otel-util-genai.
"""

import contextvars
from typing import Collection, Optional

from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Workflow,
    AgentInvocation,
    Step,
    ToolCall,
)

_instruments = ("crewai >= 0.70.0",)

# Global handler instance (singleton)
_handler: Optional[TelemetryHandler] = None

# Context variable to track parent run IDs for nested operations
_current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "crewai_current_run_id", default=None
)


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
            
        _handler = TelemetryHandler(tracer_provider=tracer_provider, meter_provider=meter_provider)
        
        # Crew.kickoff -> Workflow
        wrap_function_wrapper(
            "crewai.crew",
            "Crew.kickoff",
            _wrap_crew_kickoff
        )

        # Agent.execute_task -> AgentInvocation
        wrap_function_wrapper(
            "crewai.agent",
            "Agent.execute_task",
            _wrap_agent_execute_task
        )

        # Task.execute_sync -> Step
        wrap_function_wrapper(
            "crewai.task",
            "Task.execute_sync",
            _wrap_task_execute
        )

        # BaseTool.run -> ToolCall
        wrap_function_wrapper(
            "crewai.tools.base_tool",
            "BaseTool.run",
            _wrap_tool_run
        )

        # CrewStructuredTool.invoke -> ToolCall (for @tool decorated functions)
        wrap_function_wrapper(
            "crewai.tools.structured_tool",
            "CrewStructuredTool.invoke",
            _wrap_structured_tool_invoke
        )

    def _uninstrument(self, **kwargs):
        """Remove instrumentation from CrewAI components."""
        unwrap("crewai.crew.Crew", "kickoff")
        unwrap("crewai.agent.Agent", "execute_task")
        unwrap("crewai.task.Task", "execute_sync")
        unwrap("crewai.tools.base_tool.BaseTool", "run")
        unwrap("crewai.tools.structured_tool.CrewStructuredTool", "invoke")


def _wrap_crew_kickoff(wrapped, instance, args, kwargs):
    """
    Wrap Crew.kickoff to create a Workflow span.
    
    Maps to: Workflow type from splunk-otel-util-genai
    """
    try:
        handler = _handler
        parent_run_id = _current_run_id.get()
        
        # Create workflow invocation
        workflow = Workflow(
            name=getattr(instance, "name", None) or "CrewAI Workflow",
            workflow_type="crewai.crew",
            parent_run_id=parent_run_id,
            framework="crewai",
            system="crewai",
        )
        
        # Add crew-specific attributes
        if hasattr(instance, "process"):
            workflow.attributes["crewai.crew.process"] = str(instance.process)
        if hasattr(instance, "verbose"):
            workflow.attributes["crewai.crew.verbose"] = instance.verbose
        if hasattr(instance, "memory"):
            workflow.attributes["crewai.crew.memory"] = instance.memory
        if hasattr(instance, "agents"):
            workflow.attributes["crewai.crew.agents_count"] = len(instance.agents)
        if hasattr(instance, "tasks"):
            workflow.attributes["crewai.crew.tasks_count"] = len(instance.tasks)
        
        # Start the workflow
        handler.start_workflow(workflow)
        
        # Set as current run ID for child operations
        token = _current_run_id.set(str(workflow.run_id))
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)
    
    try:
        result = wrapped(*args, **kwargs)
        
        # Capture result information
        try:
            if result:
                if hasattr(result, "raw"):
                    workflow.output = str(result.raw)[:1000]  # Truncate large outputs
                if hasattr(result, "token_usage"):
                    workflow.attributes["crewai.crew.token_usage"] = str(result.token_usage)
                if hasattr(result, "usage_metrics"):
                    workflow.attributes["crewai.crew.usage_metrics"] = str(result.usage_metrics)
            
            # Stop the workflow successfully
            handler.stop_workflow(workflow)
        except Exception:
            # Ignore instrumentation errors on success path
            pass
        
        return result
    except Exception as error:
        # Wrapped function failed - try to record error but don't fail if we can't
        try:
            handler.stop_workflow(workflow, error=error)
        except Exception:
            pass
        raise
    finally:
        # Restore previous run ID context
        try:
            _current_run_id.reset(token)
        except Exception:
            pass


def _wrap_agent_execute_task(wrapped, instance, args, kwargs):
    """
    Wrap Agent.execute_task to create an AgentInvocation span.
    
    Maps to: AgentInvocation type from splunk-otel-util-genai
    """
    try:
        handler = _handler
        parent_run_id = _current_run_id.get()
        
        # Create agent invocation
        agent_invocation = AgentInvocation(
            name=getattr(instance, "role", "Unknown Agent"),
            parent_run_id=parent_run_id,
            framework="crewai",
            system="crewai",
        )
        
        # Add agent-specific attributes
        if hasattr(instance, "goal"):
            agent_invocation.attributes["crewai.agent.goal"] = instance.goal
        if hasattr(instance, "backstory"):
            agent_invocation.attributes["crewai.agent.backstory"] = instance.backstory[:500]
        if hasattr(instance, "verbose"):
            agent_invocation.attributes["crewai.agent.verbose"] = instance.verbose
        if hasattr(instance, "allow_delegation"):
            agent_invocation.attributes["crewai.agent.allow_delegation"] = instance.allow_delegation
        if hasattr(instance, "tools") and instance.tools:
            agent_invocation.attributes["crewai.agent.tools_count"] = len(instance.tools)
            agent_invocation.attributes["crewai.agent.tools"] = str([
                getattr(t, "name", str(t)) for t in instance.tools[:10]
            ])
        if hasattr(instance, "llm") and hasattr(instance.llm, "model"):
            agent_invocation.attributes["crewai.agent.llm_model"] = str(instance.llm.model)
        
        # Capture task information from args
        if args and hasattr(args[0], "description"):
            agent_invocation.input = args[0].description[:500]
        
        # Start the agent invocation
        handler.start_agent(agent_invocation)
        
        # Set as current run ID for child operations
        token = _current_run_id.set(str(agent_invocation.run_id))
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)
    
    try:
        result = wrapped(*args, **kwargs)
        
        # Capture result and metrics
        try:
            if result:
                agent_invocation.output = str(result)[:1000]
            
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
    except Exception as error:
        # Wrapped function failed - try to record error but don't fail if we can't
        try:
            handler.stop_agent(agent_invocation, error=error)
        except Exception:
            pass
        raise
    finally:
        # Restore previous run ID context
        try:
            _current_run_id.reset(token)
        except Exception:
            pass


def _wrap_task_execute(wrapped, instance, args, kwargs):
    """
    Wrap Task.execute_sync to create a Step span.
    
    Maps to: Step type from splunk-otel-util-genai
    """
    try:
        handler = _handler
        parent_run_id = _current_run_id.get()
        
        # Create step
        step = Step(
            name=getattr(instance, "description", None) or "Task Execution",
            parent_run_id=parent_run_id,
            framework="crewai",
            system="crewai",
        )
        
        # Add task-specific attributes
        if hasattr(instance, "description"):
            step.input = instance.description[:500]
            step.attributes["crewai.task.description"] = instance.description[:500]
        if hasattr(instance, "expected_output"):
            step.attributes["crewai.task.expected_output"] = instance.expected_output[:500]
        if hasattr(instance, "async_execution"):
            step.attributes["crewai.task.async_execution"] = instance.async_execution
        if hasattr(instance, "context") and instance.context:
            step.attributes["crewai.task.has_context"] = True
        if hasattr(instance, "agent") and hasattr(instance.agent, "role"):
            step.attributes["crewai.task.agent_role"] = instance.agent.role
        
        # Start the step
        handler.start_step(step)
        
        # Set as current run ID for child operations
        token = _current_run_id.set(str(step.run_id))
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)
    
    try:
        result = wrapped(*args, **kwargs)
        
        # Capture result
        try:
            if result:
                step.output = str(result)[:1000]
            
            # Stop the step successfully
            handler.stop_step(step)
        except Exception:
            # Ignore instrumentation errors on success path
            pass
        
        return result
    except Exception as error:
        # Wrapped function failed - try to record error but don't fail if we can't
        try:
            handler.stop_step(step, error=error)
        except Exception:
            pass
        raise
    finally:
        # Restore previous run ID context
        try:
            _current_run_id.reset(token)
        except Exception:
            pass


def _wrap_tool_run(wrapped, instance, args, kwargs):
    """
    Wrap BaseTool.run to create a ToolCall span.
    
    Maps to: ToolCall type from splunk-otel-util-genai
    """
    try:
        handler = _handler
        parent_run_id = _current_run_id.get()
        
        # Create tool call
        tool_call = ToolCall(
            name=getattr(instance, "name", "unknown_tool"),
            arguments=str(kwargs) if kwargs else "{}",
            id=str(id(instance)),
            parent_run_id=parent_run_id,
            framework="crewai",
            system="crewai",
        )
        
        # Add tool-specific attributes
        if hasattr(instance, "description"):
            tool_call.attributes["crewai.tool.description"] = instance.description
        if hasattr(instance, "args_schema"):
            tool_call.attributes["crewai.tool.has_args_schema"] = True
        
        # Capture input arguments
        if args:
            tool_call.input = str(args)[:500]
        if kwargs:
            tool_call.attributes["crewai.tool.kwargs"] = str(kwargs)[:500]
        
        # Start the tool call
        handler.start_tool_call(tool_call)
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)
    
    try:
        result = wrapped(*args, **kwargs)
        
        # Capture result
        try:
            if result:
                tool_call.output = str(result)[:1000]
            
            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)
        except Exception:
            # Ignore instrumentation errors on success path
            pass
        
        return result
    except Exception as error:
        # Wrapped function failed - try to record error but don't fail if we can't
        try:
            handler.stop_tool_call(tool_call, error=error)
        except Exception:
            pass
        raise


def _wrap_structured_tool_invoke(wrapped, instance, args, kwargs):
    """
    Wrap CrewStructuredTool.invoke to create a ToolCall span.
    
    This handles tools created with the @tool decorator.
    Maps to: ToolCall type from splunk-otel-util-genai
    """
    try:
        handler = _handler
        parent_run_id = _current_run_id.get()
        
        # Create tool call
        tool_call = ToolCall(
            name=getattr(instance, "name", "unknown_tool"),
            arguments=str(kwargs) if kwargs else "{}",
            id=str(id(instance)),
            parent_run_id=parent_run_id,
            framework="crewai",
            system="crewai",
        )
        
        # Add tool-specific attributes
        if hasattr(instance, "description"):
            tool_call.attributes["crewai.tool.description"] = instance.description
        if hasattr(instance, "result_as_answer"):
            tool_call.attributes["crewai.tool.result_as_answer"] = instance.result_as_answer
        if hasattr(instance, "max_usage_count"):
            tool_call.attributes["crewai.tool.max_usage_count"] = instance.max_usage_count
        if hasattr(instance, "current_usage_count"):
            tool_call.attributes["crewai.tool.current_usage_count"] = instance.current_usage_count
        
        # Capture input arguments
        if args:
            tool_call.input = str(args[0])[:500] if len(args) > 0 else ""
        if kwargs:
            tool_call.attributes["crewai.tool.kwargs"] = str(kwargs)[:500]
        
        # Start the tool call
        handler.start_tool_call(tool_call)
    except Exception:
        # If instrumentation setup fails, just run the original function
        return wrapped(*args, **kwargs)
    
    try:
        result = wrapped(*args, **kwargs)
        
        # Capture result
        try:
            if result:
                tool_call.output = str(result)[:1000]
            
            # Stop the tool call successfully
            handler.stop_tool_call(tool_call)
        except Exception:
            # Ignore instrumentation errors on success path
            pass
        
        return result
    except Exception as error:
        # Wrapped function failed - try to record error but don't fail if we can't
        try:
            handler.stop_tool_call(tool_call, error=error)
        except Exception:
            pass
        raise

