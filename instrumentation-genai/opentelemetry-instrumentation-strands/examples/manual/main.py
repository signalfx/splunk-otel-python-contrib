#!/usr/bin/env python3
# Copyright Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example demonstrating Strands Agent instrumentation with OpenTelemetry.

This example shows how to instrument a Strands Agent to capture:
- Agent invocation spans
- LLM call spans (via Strands hooks)
- Tool call spans (via Strands hooks)

Requirements:
    - strands-agents >= 1.0.0
    - AWS credentials configured (for Bedrock)
    - OpenTelemetry SDK

Usage:
    python main.py
"""

import logging
import sys

from opentelemetry import trace
from opentelemetry.instrumentation.strands import StrandsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_telemetry():
    """Set up OpenTelemetry with console exporter for demo."""
    # Create tracer provider
    provider = TracerProvider()

    # Add console exporter for demo (exports spans to stdout)
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    return provider


def main():
    """Run the example."""
    logger.info("Setting up OpenTelemetry...")
    tracer_provider = setup_telemetry()

    logger.info("Instrumenting Strands...")
    StrandsInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        # Import Strands components
        from strands.agent import Agent
        from strands.tools import Tool

        logger.info("Creating example tool...")

        # Define a simple tool
        def search_docs(query: str) -> str:
            """Search documentation for information."""
            # Simulate search
            return f"Found documentation for: {query}"

        search_tool = Tool(
            name="search_docs",
            description="Search documentation",
            function=search_docs,
        )

        logger.info("Creating Strands agent...")

        # Create agent with tool
        agent = Agent(
            name="research_agent",
            model="anthropic.claude-v2",
            instructions="You are a helpful research assistant. Use the search_docs tool to find information.",
            tools=[search_tool],
        )

        logger.info("Invoking agent...")

        # Invoke agent (this will create spans)
        result = agent("What are the latest trends in AI?")

        logger.info(f"Agent response: {result}")

        # Async example (if supported)
        logger.info("Testing async invocation...")
        try:
            import asyncio

            async def async_example():
                result = await agent.invoke_async("What is machine learning?")
                return result

            result = asyncio.run(async_example())
            logger.info(f"Async agent response: {result}")
        except AttributeError:
            logger.info("Async invocation not available in this Strands version")

    except ImportError as e:
        logger.error(f"Failed to import Strands: {e}")
        logger.error(
            "Please install strands-agents: pip install strands-agents"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Example completed successfully!")

    # Flush spans
    tracer_provider.force_flush()


if __name__ == "__main__":
    main()
