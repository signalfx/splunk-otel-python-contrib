#!/usr/bin/env python3
"""
Direct Azure OpenAI Application - No Framework
Tests LLMInvocation and AgentInvocation without LangChain/LangGraph

This app demonstrates:
- Direct OpenAI client usage
- LLMInvocation for LLM calls
- AgentInvocation for agent-level operations
- Manual span creation and management
"""

import os
import sys
from openai import AzureOpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Configure OpenTelemetry
resource = Resource.create({
    "service.name": os.getenv("OTEL_SERVICE_NAME", "direct-ai-app"),
    "deployment.environment": "alpha-test",
})

trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

tracer = trace.get_tracer(__name__)


class DirectAIApp:
    """Direct AI application without frameworks"""
    
    def __init__(self):
        # Get OpenAI configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com")
        ) if "AZURE_OPENAI_ENDPOINT" in os.environ else None
        
        # Fallback to ChatGPT OpenAI
        if not self.client:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        
        self.model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    def llm_invocation_test(self, prompt: str) -> str:
        """
        Test LLMInvocation - Direct LLM call with manual instrumentation
        
        This demonstrates:
        - Manual span creation for LLM calls
        - Token usage tracking
        - Response capture
        """
        with tracer.start_as_current_span(
            "llm.invocation",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": self.model,
                "gen_ai.system": "openai",
                "gen_ai.request.temperature": 0.7,
            }
        ) as span:
            try:
                # Make LLM call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                # Extract response
                content = response.choices[0].message.content
                
                # Add telemetry attributes
                span.set_attribute("gen_ai.response.finish_reason", response.choices[0].finish_reason)
                span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
                span.set_attribute("gen_ai.usage.total_tokens", response.usage.total_tokens)
                
                print(f"✓ LLM Response: {content[:100]}...")
                return content
                
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                print(f"✗ LLM Error: {e}")
                raise
    
    def agent_invocation_test(self, task: str) -> dict:
        """
        Test AgentInvocation - Agent-level operation with manual instrumentation
        
        This demonstrates:
        - Agent-level span creation
        - Multi-step agent workflow
        - Agent decision tracking
        """
        with tracer.start_as_current_span(
            "agent.invocation",
            attributes={
                "gen_ai.agent.name": "direct_ai_agent",
                "gen_ai.agent.type": "reasoning",
                "gen_ai.operation.name": "execute_task",
            }
        ) as agent_span:
            try:
                print(f"\n🤖 Agent Task: {task}")
                
                # Step 1: Analyze task
                with tracer.start_as_current_span("agent.step.analyze") as step_span:
                    step_span.set_attribute("gen_ai.step.name", "analyze_task")
                    analysis = self._analyze_task(task)
                    print(f"  ✓ Analysis: {analysis}")
                
                # Step 2: Execute task
                with tracer.start_as_current_span("agent.step.execute") as step_span:
                    step_span.set_attribute("gen_ai.step.name", "execute_task")
                    result = self._execute_task(task)
                    print(f"  ✓ Result: {result[:100]}...")
                
                # Step 3: Validate result
                with tracer.start_as_current_span("agent.step.validate") as step_span:
                    step_span.set_attribute("gen_ai.step.name", "validate_result")
                    validation = self._validate_result(result)
                    print(f"  ✓ Validation: {validation}")
                
                # Set agent outcome
                agent_span.set_attribute("gen_ai.agent.outcome", "success")
                agent_span.set_attribute("gen_ai.agent.steps_completed", 3)
                
                return {
                    "task": task,
                    "analysis": analysis,
                    "result": result,
                    "validation": validation,
                    "status": "success"
                }
                
            except Exception as e:
                agent_span.set_attribute("error", True)
                agent_span.set_attribute("error.message", str(e))
                agent_span.set_attribute("gen_ai.agent.outcome", "failure")
                print(f"✗ Agent Error: {e}")
                raise
    
    def _analyze_task(self, task: str) -> str:
        """Analyze the task (simulated)"""
        return f"Task requires: information retrieval and synthesis"
    
    def _execute_task(self, task: str) -> str:
        """Execute the task using LLM"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a task execution agent."},
                {"role": "user", "content": task}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    
    def _validate_result(self, result: str) -> str:
        """Validate the result (simulated)"""
        if len(result) > 10:
            return "Valid - result meets quality criteria"
        return "Invalid - result too short"


def main():
    """Main execution"""
    print("=" * 80)
    print("🚀 Direct Azure OpenAI Application - No Framework")
    print("=" * 80)
    print("Testing LLMInvocation and AgentInvocation")
    print("=" * 80)
    print()
    
    try:
        # Initialize app
        app = DirectAIApp()
        
        # Test 1: LLMInvocation
        print("\n" + "=" * 80)
        print("Test 1: LLMInvocation (Direct LLM Call)")
        print("=" * 80)
        
        llm_result = app.llm_invocation_test(
            "Explain what OpenTelemetry is in one sentence."
        )
        
        # Test 2: AgentInvocation
        print("\n" + "=" * 80)
        print("Test 2: AgentInvocation (Agent Workflow)")
        print("=" * 80)
        
        agent_result = app.agent_invocation_test(
            "Research and summarize the benefits of observability in AI applications."
        )
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ Test Summary")
        print("=" * 80)
        print("✓ LLMInvocation: PASSED")
        print("✓ AgentInvocation: PASSED")
        print("✓ Telemetry: Exported to OTLP")
        print()
        print("Next Steps:")
        print("1. Check Splunk APM for traces")
        print("2. Verify LLM and Agent spans")
        print("3. Check token usage metrics")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
