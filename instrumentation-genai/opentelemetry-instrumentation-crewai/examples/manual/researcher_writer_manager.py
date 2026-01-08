from crewai import Agent, Crew, Task, Process
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

# Disable CrewAI's built-in telemetry
import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OPENAI_MODEL_NAME"] = "gpt-5-mini"

# Manager agent coordinates the team
manager = Agent(
    role="Project Manager",
    goal="Coordinate team efforts and ensure project success",
    backstory="Experienced project manager skilled at delegation and quality control",
    allow_delegation=True,
    verbose=True,
)

# Specialist agents
researcher = Agent(
    role="Researcher",
    goal="Provide accurate research and analysis",
    backstory="Expert researcher with deep analytical skills",
    allow_delegation=False,  # Specialists focus on their expertise
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Create compelling content",
    backstory="Skilled writer who creates engaging content",
    allow_delegation=False,
    verbose=True,
)

# Manager-led task
project_task = Task(
    description="Create a comprehensive market analysis report with recommendations",
    expected_output="Executive summary, detailed analysis, and strategic recommendations",
    agent=manager,  # Manager will delegate to specialists
)

# Hierarchical crew
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[project_task],
    process=Process.hierarchical,  # Manager coordinates everything
    manager_llm="gpt-4o",  # Specify LLM for manager
    verbose=True,
)


tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
# CRITICAL: Register the tracer provider globally so it can be flushed
trace.set_tracer_provider(tracer_provider)

CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

crew.kickoff()

# ============================================================================
# Trace Wireframe - Hierarchical CrewAI with Manager Delegation
# ============================================================================
# gen_ai.workflow crew 1m2.249844s
# └── gen_ai.step Create a comprehensive market analysis report with recommendations 1m2.238044s
#     └── invoke_agent Crew Manager 1m2.237179s
#         ├── tool Ask question to coworker 7.328951s
#         │   └── invoke_agent Project Manager 7.326713s
#         ├── tool Delegate work to coworker 11.406297s
#         │   └── invoke_agent Project Manager 11.401578s
#         └── tool Delegate work to coworker 6.136559s
#             └── invoke_agent Project Manager 6.130725s
# ============================================================================
