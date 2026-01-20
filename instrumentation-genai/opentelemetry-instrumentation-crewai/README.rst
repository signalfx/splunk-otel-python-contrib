OpenTelemetry CrewAI Instrumentation
=====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-crewai.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-crewai/

This library provides OpenTelemetry instrumentation for `CrewAI <https://github.com/joaomdmoura/crewAI>`_,
a framework for orchestrating autonomous AI agents.

Installation
------------

.. code-block:: bash

    pip install splunk-otel-instrumentation-crewai


Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    from crewai import Agent, Task, Crew

    # Instrument CrewAI
    CrewAIInstrumentor().instrument()

    # Create your crew
    agent = Agent(
        role="Research Analyst",
        goal="Provide accurate research",
        backstory="Expert researcher with attention to detail",
    )

    task = Task(
        description="Research the latest AI trends",
        expected_output="A comprehensive report on AI trends",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    # Run your crew - telemetry is automatically captured
    result = crew.kickoff()


What Gets Instrumented
-----------------------

This instrumentation captures:

- **Crews** → Mapped to ``Workflow`` spans
- **Tasks** → Mapped to ``Step`` spans
- **Agents** → Mapped to ``AgentInvocation`` spans
- **Tool Usage** → Mapped to ``ToolCall`` spans

All spans are properly nested with correct parent-child relationships and include
rich attributes about the operation.


Compositional Instrumentation
------------------------------

This instrumentation focuses on CrewAI's workflow orchestration. For complete observability:

**CrewAI Only**

.. code-block:: python

    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    
    CrewAIInstrumentor().instrument()

Provides workflow structure but no LLM call details.

**CrewAI + OpenAI**

.. code-block:: python

    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    
    CrewAIInstrumentor().instrument()
    OpenAIInstrumentor().instrument()

Adds LLM call spans with token usage, model names, and latency metrics.

**Full Stack (CrewAI + OpenAI + Vector Store)**

.. code-block:: python

    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.chromadb import ChromaDBInstrumentor
    
    CrewAIInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    ChromaDBInstrumentor().instrument()

Complete RAG workflow visibility with vector store operations.


Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Disable CrewAI's built-in telemetry (recommended)
    export CREWAI_DISABLE_TELEMETRY=true


Instrumentation Options
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

    # Basic instrumentation
    CrewAIInstrumentor().instrument()

    # With custom tracer provider
    CrewAIInstrumentor().instrument(tracer_provider=my_tracer_provider)

    # Uninstrumentation
    CrewAIInstrumentor().uninstrument()


Requirements
------------

- Python >= 3.9
- CrewAI >= 0.70.0
- OpenTelemetry API >= 1.38
- ``splunk-otel-util-genai`` >= 0.1.4


Trace Hierarchy Example
------------------------

.. code-block::

    Crew: Customer Support (Workflow)
    ├── Task: inquiry_resolution (Step)
    │   └── Agent: Senior Support Representative
    │       ├── LLM: gpt-4o-mini (via openai-instrumentation)
    │       └── Tool: docs_scrape
    └── Task: quality_assurance (Step)
        └── Agent: QA Specialist
            └── LLM: gpt-4o-mini (via openai-instrumentation)


Each span includes rich attributes:

- ``gen_ai.system`` = "crewai"
- ``gen_ai.operation.name`` = "invoke_workflow" | "invoke_agent" | "execute_tool"
- Framework-specific attributes (agent role, task description, tool names, etc.)


Limitations
-----------

- **Async Support**: Currently supports synchronous workflows only. Async support (``kickoff_async()``)
  is planned for a future release.
- **LLM Calls**: Not instrumented here. Use provider-specific instrumentation
  (e.g., ``opentelemetry-instrumentation-openai``).


Contributing
------------

Contributions are welcome! Please ensure:

- All tests pass
- Code follows project style guidelines
- Instrumentation is defensive (catches exceptions)
- Documentation is updated


Links
-----

- `CrewAI Documentation <https://docs.crewai.com/>`_
- `OpenTelemetry Python <https://opentelemetry.io/docs/languages/python/>`_
- `Splunk GenAI Utilities <https://github.com/signalfx/splunk-otel-python-contrib>`_


License
-------

Apache-2.0

