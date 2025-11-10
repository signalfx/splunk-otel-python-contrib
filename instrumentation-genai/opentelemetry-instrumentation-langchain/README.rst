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

