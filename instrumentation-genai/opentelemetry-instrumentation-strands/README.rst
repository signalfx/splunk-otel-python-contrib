OpenTelemetry Strands Agents Instrumentation
============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-strands.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-strands/

This library provides OpenTelemetry instrumentation for `Strands Agents SDK <https://github.com/aws-samples/strands-agents>`_,
the framework powering AWS Bedrock AgentCore for building autonomous AI agents.

Installation
------------

.. code-block:: bash

    pip install splunk-otel-instrumentation-strands


Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.strands import StrandsInstrumentor
    from strands.agent import Agent
    from strands.tools import Tool

    # Instrument Strands
    StrandsInstrumentor().instrument()

    # Create your agent
    agent = Agent(
        name="research_agent",
        model="anthropic.claude-v2",
        instructions="You are a helpful research assistant",
        tools=[my_tool],
    )

    # Run your agent - telemetry is automatically captured
    result = agent("What are the latest trends in AI?")


What Gets Instrumented
-----------------------

This instrumentation captures:

**Strands Agents SDK:**

- **Agent Invocations** → Mapped to ``AgentInvocation`` spans
- **LLM Calls** → Mapped to ``LLMInvocation`` spans (via Strands hooks)
- **Tool Calls** → Mapped to ``ToolCall`` spans (via Strands hooks)

**Bedrock AgentCore Components:**

- **BedrockAgentCoreApp** → Mapped to ``Workflow`` spans (optional)
- **Memory Operations** (MemoryClient)

  - ``retrieve_memories`` → ``RetrievalInvocation`` span
  - ``create_event``, ``create_blob_event``, ``list_events`` → ``ToolCall`` spans

- **Code Interpreter** (CodeInterpreter) → Mapped to ``ToolCall`` spans

  - start/stop sessions
  - execute_code
  - install_packages
  - upload_file

- **Browser Automation** (BrowserClient) → Mapped to ``ToolCall`` spans

  - start/stop sessions
  - take_control/release_control
  - get_session

All spans are properly nested with correct parent-child relationships and include
rich attributes about the operation.


Instrumentation Strategy
-------------------------

This instrumentation uses a **hybrid hooks + wrapt** approach:

**Strands Hooks** (LLM and Tool Telemetry)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strands SDK provides lifecycle hooks for model and tool calls. This instrumentation
registers hook callbacks via ``HookProvider`` to capture:

- ``BeforeModelCallEvent`` / ``AfterModelCallEvent`` → LLM spans
- ``BeforeToolCallEvent`` / ``AfterToolCallEvent`` → Tool spans

**Wrapt Monkey-Patching** (Agent Lifecycle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent-level spans are created by wrapping:

- ``Agent.__init__`` → Injects hook provider into agent's hook registry
- ``Agent.invoke_async`` → Creates ``AgentInvocation`` spans

``Agent.__call__`` is not wrapped separately because it delegates to ``invoke_async``
internally, so wrapping both would produce duplicate spans.


Built-in Tracer Suppression
----------------------------

Strands SDK has its own OpenTelemetry tracer (``strands.telemetry.tracer.Tracer``)
that creates spans. To avoid double-tracing, this instrumentation can suppress
the built-in tracer:

.. code-block:: bash

    # Suppress Strands' built-in tracer (default)
    export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=true

    # Keep both Strands and instrumentation spans
    export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=false

When suppressed, Strands' tracer methods become no-ops, and only this
instrumentation's spans are created.


Compositional Instrumentation
------------------------------

This instrumentation focuses on Strands agent orchestration. For complete observability:

**Strands Only**

.. code-block:: python

    from opentelemetry.instrumentation.strands import StrandsInstrumentor

    StrandsInstrumentor().instrument()

Provides agent workflow structure and LLM/tool call details via hooks.

**Strands + Bedrock**

.. code-block:: python

    from opentelemetry.instrumentation.strands import StrandsInstrumentor
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

    StrandsInstrumentor().instrument()
    BedrockInstrumentor().instrument()

Adds Bedrock API call spans with additional AWS-specific metrics.

**Full Stack (Strands + Bedrock + Vector Store)**

.. code-block:: python

    from opentelemetry.instrumentation.strands import StrandsInstrumentor
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    from opentelemetry.instrumentation.chromadb import ChromaDBInstrumentor

    StrandsInstrumentor().instrument()
    BedrockInstrumentor().instrument()
    ChromaDBInstrumentor().instrument()

Complete RAG workflow visibility with vector store operations.


Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Suppress Strands' built-in OTel tracer (recommended, default: true)
    export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=true


Instrumentation Options
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from opentelemetry.instrumentation.strands import StrandsInstrumentor

    # Basic instrumentation
    StrandsInstrumentor().instrument()

    # With custom tracer provider
    StrandsInstrumentor().instrument(tracer_provider=my_tracer_provider)

    # Uninstrumentation
    StrandsInstrumentor().uninstrument()


Requirements
------------

- Python >= 3.10
- strands-agents >= 1.0.0
- OpenTelemetry API >= 1.38
- ``splunk-otel-util-genai`` >= 0.1.9


Trace Hierarchy Example
------------------------

.. code-block::

    BedrockAgentCoreApp.entrypoint (Workflow)
    └── Agent: research_agent (AgentInvocation)
        ├── LLM: anthropic.claude-v2 (LLMInvocation)
        └── Tool: search_docs (ToolCall)


Each span includes rich attributes:

- ``gen_ai.system`` = "strands"
- ``gen_ai.operation.name`` = "agent_invocation" | "llm_invocation" | "tool_call"
- Framework-specific attributes (agent name, model ID, tool names, etc.)


Limitations
-----------

- **Async Support**: Both sync (``Agent.__call__``) and async (``Agent.invoke_async``)
  agent invocations are supported.
- **LLM Provider Details**: Captured via Strands hooks. For additional provider-specific
  telemetry, use provider instrumentation (e.g., ``opentelemetry-instrumentation-bedrock``).
- **AgentCore Components**: Memory, Code Interpreter, and Browser Automation are
  instrumented. Gateway and ServerTools are not yet available in the SDK.


Contributing
------------

Contributions are welcome! Please ensure:

- All tests pass
- Code follows project style guidelines
- Instrumentation is defensive (catches exceptions)
- Documentation is updated


Links
-----

- `Strands Agents SDK <https://github.com/aws-samples/strands-agents>`_
- `AWS Bedrock AgentCore <https://docs.aws.amazon.com/bedrock/>`_
- `OpenTelemetry Python <https://opentelemetry.io/docs/languages/python/>`_
- `Splunk GenAI Utilities <https://github.com/signalfx/splunk-otel-python-contrib>`_


License
-------

Apache-2.0
