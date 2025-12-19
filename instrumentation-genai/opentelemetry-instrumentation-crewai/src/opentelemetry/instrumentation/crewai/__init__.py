"""
OpenTelemetry CrewAI Instrumentation

Wrapper-based instrumentation for CrewAI using splunk-otel-util-genai.
"""

from opentelemetry.instrumentation.crewai.instrumentation import CrewAIInstrumentor
from opentelemetry.instrumentation.crewai.version import __version__

__all__ = ["CrewAIInstrumentor", "__version__"]
