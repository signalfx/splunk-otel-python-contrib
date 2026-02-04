OpenTelemetry OpenAI Agents Instrumentation (Alpha)
===================================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-openai-agents-v2.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-openai-agents-v2/

This package provides OpenTelemetry instrumentation for 
`OpenAI Agents SDK <https://github.com/openai/openai-agents-python>`_,
a framework for building agentic AI applications. It leverages Splunk distribution 
of ``opentelemetry-util-genai`` for producing telemetry in semantic convention. 
`Core concepts, high-level usage and configuration <https://github.com/signalfx/splunk-otel-python-contrib/>`_

Status: Alpha (APIs and produced telemetry are subject to change).

Installation
------------

.. code-block:: bash

    pip install splunk-otel-instrumentation-openai-agents-v2


Quick Start
-----------

Manual Instrumentation (development/debugging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
    from agents import Agent, Runner, function_tool

    # manual instrumentation, easy to debug in your IDE
    OpenAIAgentsInstrumentor().instrument()

    @function_tool
    def get_weather(city: str) -> str:
        return f"The forecast for {city} is sunny."

    agent = Agent(
        name="Travel Concierge",
        instructions="You are a concise travel concierge.",
        tools=[get_weather],
    )

    result = Runner.run_sync(agent, "How is the weather in Paris?")
    print(result.final_output)

Zero-code instrumentation
^^^^^^^^^^^^^^^^^^^^^^^^^

In zero-code instrumentation mode, ensure you install opentelemetry-distribution and run your 
app with the OpenTelemetry OpenAI Agents Instrumentor enabled:

.. code:: bash

    opentelemetry-instrument python your_agents_app.py

.. code:: python

    from agents import Agent, Runner

    agent = Agent(
        name="Travel Concierge",
        instructions="You are a concise travel concierge.",
    )

    result = Runner.run_sync(agent, "What are top attractions in Tokyo?")
    print(result.final_output)


What Gets Instrumented
-----------------------

This instrumentation captures:

- **Workflows** -> Mapped to ``Workflow`` spans
- **Agents** -> Mapped to ``AgentInvocation`` spans
- **Tool Calls** -> Mapped to ``ToolCall`` spans
- **Generations** -> Mapped to ``LLMInvocation`` spans

All spans are properly nested with correct parent-child relationships and include
rich attributes about the operation.


Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Capture message content (disabled by default)
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

    # Content capture mode
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT

    # Disable metrics
    export OTEL_INSTRUMENTATION_OPENAI_AGENTS_CAPTURE_METRICS=false


Instrumentation Options
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    # Basic instrumentation
    OpenAIAgentsInstrumentor().instrument()

    # With custom tracer provider
    OpenAIAgentsInstrumentor().instrument(tracer_provider=my_tracer_provider)

    # Uninstrumentation
    OpenAIAgentsInstrumentor().uninstrument()


Requirements
------------

- Python >= 3.10
- openai-agents >= 0.3.3
- OpenTelemetry API >= 1.37
- ``splunk-otel-util-genai`` >= 0.1.9


Trace Hierarchy Example
------------------------

.. code-block::

    Workflow: Travel Planning (Workflow)
    └── Agent: Travel Concierge (AgentInvocation)
        ├── LLM: gpt-4o-mini (Generation)
        │   └── gen_ai.choice
        ├── Tool: get_weather (ToolCall)
        └── LLM: gpt-4o-mini (Generation)
            └── gen_ai.choice


Each span includes rich attributes:

- ``gen_ai.system`` = "openai"
- ``gen_ai.framework`` = "openai_agents"
- ``gen_ai.operation.name`` = "invoke_workflow" | "invoke_agent" | "execute_tool" | "chat"
- Framework-specific attributes (agent name, tool names, etc.)


Testing
-------
Run the package tests (from repository root or this directory)::

    pytest instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/tests


Contributing
------------

Issues / PRs welcome in the main splunk-otel-python-contrib repository. This
module is alpha: feedback on attribute coverage, performance, and OpenAI Agents
surface expansion is especially helpful.


Links
-----

- `OpenAI Agents SDK <https://github.com/openai/openai-agents-python>`_
- `OpenTelemetry Python <https://opentelemetry.io/docs/languages/python/>`_
- `Splunk GenAI Utilities <https://github.com/signalfx/splunk-otel-python-contrib>`_


License
-------

Apache-2.0
