OpenTelemetry Bedrock AgentCore Instrumentation
===============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-bedrock-agentcore.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-bedrock-agentcore/

This library provides OpenTelemetry instrumentation for `AWS Bedrock AgentCore <https://docs.aws.amazon.com/bedrock/>`_,
the runtime framework for building autonomous AI agents with AWS Bedrock.

Installation
------------

.. code-block:: bash

    pip install splunk-otel-instrumentation-bedrock-agentcore


Usage
-----

Quick Start
~~~~~~~~~~~

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor
    from bedrock_agentcore import BedrockAgentCoreApp

    # Instrument Bedrock AgentCore
    BedrockAgentCoreInstrumentor().instrument()

    # Create your application
    app = BedrockAgentCoreApp()

    @app.entrypoint
    def handler(event):
        # Your agent logic here
        return {"status": "success"}


Complete Example with Telemetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    """AWS Bedrock AgentCore with OpenTelemetry instrumentation."""
    import os
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor

    # Configure OpenTelemetry
    resource = Resource(attributes={
        ResourceAttributes.SERVICE_NAME: "bedrock-agent",
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
    })

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )
    trace.set_tracer_provider(tracer_provider)

    # Instrument Bedrock AgentCore
    BedrockAgentCoreInstrumentor().instrument(tracer_provider=tracer_provider)

    # Now use Bedrock AgentCore components - they're automatically instrumented
    from bedrock_agentcore import BedrockAgentCoreApp
    from bedrock_agentcore.memory.client import MemoryClient
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
    from bedrock_agentcore.tools.browser_client import BrowserClient

    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    memory_client = MemoryClient(region_name=region)
    code_interpreter = CodeInterpreter(region=region)
    browser_client = BrowserClient(region=region)

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def handler(event):
        """Agent entrypoint - creates a Workflow span."""
        memory_id = event.get("memory_id")

        # Memory retrieval - creates RetrievalInvocation span
        memories = memory_client.retrieve_memories(
            memory_id=memory_id,
            namespace="agent",
            query=event.get("query", ""),
        )

        # Code execution - creates ToolCall spans
        code_interpreter.start()
        code_interpreter.execute_code("print('Processing...')")
        code_interpreter.stop()

        # Browser automation - creates ToolCall spans
        browser_client.start()
        browser_client.take_control()
        browser_client.stop()

        return {"status": "success", "memories_found": len(memories) if memories else 0}

    response = handler({"query": "What is AWS Bedrock?", "memory_id": "myMemory-abc123"})
    tracer_provider.force_flush(timeout_millis=5000)


What Gets Instrumented
-----------------------

This instrumentation provides **comprehensive coverage** of all Bedrock AgentCore components:

- **BedrockAgentCoreApp** (1 method) → ``Workflow`` spans
- **MemoryClient** (38 methods) → ``RetrievalInvocation`` + ``ToolCall`` spans
- **CodeInterpreter** (17 methods) → ``ToolCall`` spans
- **BrowserClient** (13 methods) → ``ToolCall`` spans

**Total: 69 instrumented methods** providing complete observability for:

- Memory operations (retrieval, events, conversation, strategy management)
- Code execution (sessions, execution, file operations, packages)
- Browser automation (sessions, control, streaming, resource management)
- Workflow orchestration (entrypoint)

All spans are properly nested with correct parent-child relationships and include
rich attributes about the operation.


Instrumentation Strategy
-------------------------

This instrumentation uses **wrapt monkey-patching** to wrap BedrockAgentCoreApp
entrypoint and tool/memory client methods. Each operation creates the appropriate
GenAI span type (Workflow, ToolCall, or RetrievalInvocation) with full context.


Compositional Instrumentation
------------------------------

This instrumentation focuses on Bedrock AgentCore runtime operations. For complete observability:

**AgentCore + Bedrock**

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

    BedrockAgentCoreInstrumentor().instrument()
    BedrockInstrumentor().instrument()

Adds Bedrock API call spans with additional AWS-specific metrics.

**Full Stack (AgentCore + Bedrock + Strands)**

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    from opentelemetry.instrumentation.strands import StrandsInstrumentor

    BedrockAgentCoreInstrumentor().instrument()
    BedrockInstrumentor().instrument()
    StrandsInstrumentor().instrument()

Complete agent workflow visibility with Strands agent orchestration.


Full Working Example
~~~~~~~~~~~~~~~~~~~~

See ``examples/manual/main.py`` which demonstrates all four components with one
representative operation per span type and OTLP export.

To run the example:

.. code-block:: bash

    cd examples/manual

    # Install dependencies
    pip install -r requirements.txt
    pip install -e ../../[instruments]

    # Run against an OTLP collector (default: http://localhost:4317)
    export AWS_DEFAULT_REGION=us-west-2
    python main.py

    # Override the collector endpoint
    OTEL_EXPORTER_OTLP_ENDPOINT=http://my-collector:4317 python main.py


Configuration
-------------

Instrumentation Options
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor

    # Basic instrumentation
    BedrockAgentCoreInstrumentor().instrument()

    # With custom tracer provider
    BedrockAgentCoreInstrumentor().instrument(tracer_provider=my_tracer_provider)

    # Uninstrumentation
    BedrockAgentCoreInstrumentor().uninstrument()


Requirements
------------

- Python >= 3.10
- bedrock-agentcore >= 1.0.0
- OpenTelemetry API >= 1.38
- ``splunk-otel-util-genai`` >= 0.1.9


Trace Hierarchy Example
------------------------

.. code-block::

    BedrockAgentCoreApp.entrypoint (Workflow)
    ├── Memory: retrieve_memories (RetrievalInvocation)
    ├── CodeInterpreter: execute_code (ToolCall)
    └── Browser: take_control (ToolCall)


Each span includes rich attributes:

- ``gen_ai.system`` = "bedrock-agentcore"
- ``gen_ai.operation.name`` = "workflow" | "tool_call" | "retrieval_invocation"
- Framework-specific attributes (memory IDs, code snippets, browser session IDs, etc.)


Contributing
------------

Contributions are welcome! Please ensure:

- All tests pass
- Code follows project style guidelines
- Instrumentation is defensive (catches exceptions)
- Documentation is updated


Links
-----

- `AWS Bedrock AgentCore <https://docs.aws.amazon.com/bedrock/>`_
- `OpenTelemetry Python <https://opentelemetry.io/docs/languages/python/>`_
- `Splunk GenAI Utilities <https://github.com/signalfx/splunk-otel-python-contrib>`_


License
-------

Apache-2.0
