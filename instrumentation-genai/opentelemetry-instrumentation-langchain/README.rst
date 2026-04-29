OpenTelemetry LangChain Instrumentation (Alpha)
===============================================

This package provides OpenTelemetry instrumentation for LangChain LLM/chat
workflows. It leverages Splunk distribution of ``opentelemetry-util-genai`` 
for producing telemetry in semantic convention. `Core concepts, high-level 
usage and configuration  <https://github.com/signalfx/splunk-otel-python-contrib/>`_ 

Status: Alpha (APIs and produced telemetry are subject to change).

Installation
------------
Install from source:

    pip install -e splunk-otel-instrumentation-langchain

This will pull in required OpenTelemetry core + ``opentelemetry-util-genai``.



Quick Start
-----------

Manual Instrumentation (development/debugging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from opentelemetry.instrumentation.langchain import LangChainInstrumentor
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    # manual instrumentation, easy to debug in your IDE
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]
    response = llm.invoke(messages)
    print(response.content)

Zero-code instrumentation
^^^^^^^^^^^^^^^^^^^^^^^^^

in zero-code instrumentation mode, ensure you install opentelemetry-distribution and run you 
app with the OpenTelemetry LangChain Instrumentor enabled::

.. code:: bash

    opentelemetry-instrument python your_langchain_app.py

.. code:: python

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]
    response = llm.invoke(messages)
    print(response.content)

Embedding Support
^^^^^^^^^^^^^^^^^

The instrumentation wraps ``embed_documents()`` and ``embed_query()`` methods for 
embedding classes (e.g., ``OpenAIEmbeddings``, ``AzureOpenAIEmbeddings``).

**Token Counting**: For token usage metrics, the instrumentation uses ``tiktoken`` 
to count input tokens client-side. This is necessary because:

- LangChain's embedding methods only return embedding vectors (``list[list[float]]``)
- The API's ``usage`` response is discarded internally by LangChain
- LangChain has no embedding callbacks (unlike LLM callbacks)

``tiktoken`` is already a required dependency of ``langchain-openai``, so no additional 
installation is needed when using OpenAI embeddings. If ``tiktoken`` is unavailable 
(e.g., for non-OpenAI providers), token metrics are gracefully omitted.

Interrupt/Resume Support
^^^^^^^^^^^^^^^^^^^^^^^^

The instrumentation automatically detects LangGraph interrupt/resume patterns:

- **Error classification** — ``GraphInterrupt``, ``NodeInterrupt``, and ``Interrupt``
  exceptions are classified as interrupts (span status left as ``UNSET`` instead of ``ERROR``).
  ``CancelledError`` and ``TaskCancelledError`` are classified as cancellations.
- **Conversation ID** — ``thread_id`` from LangGraph checkpoint metadata is extracted
  and set as ``gen_ai.conversation.id`` on root workflow/agent spans.

See the ``examples/multi_agent_travel_planner`` demo for interrupt/resume in action.

Known LangGraph Compatibility Issue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``langgraph==1.1.7`` introduced a breaking change that silently drops non-``GraphCallbackHandler``
callback handlers. This prevents the instrumentation from receiving any callbacks.
The issue was fixed in `langgraph 1.1.8 <https://github.com/langchain-ai/langgraph/releases/tag/1.1.8>`_.
This package excludes ``langgraph==1.1.7`` via ``!= 1.1.7`` in its dependency specifier.

Testing
-------
Run the package tests (from repository root or this directory)::

    pytest -k langchain instrumentation-genai/opentelemetry-instrumentation-langchain-alpha/tests

(Recorded cassettes or proper API keys may be required for full integration tests.)

Contributing
------------
Issues / PRs welcome in the main otel-splunk-python-contrib repository. This
module is alpha: feedback on attribute coverage, performance, and LangChain
surface expansion is especially helpful.

License
-------
Apache 2.0

