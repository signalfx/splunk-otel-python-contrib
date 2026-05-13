OpenTelemetry Bedrock AgentCore Instrumentation (Alpha)
=======================================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-bedrock-agentcore.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-bedrock-agentcore/

This package provides OpenTelemetry instrumentation for
`AWS Bedrock AgentCore <https://docs.aws.amazon.com/bedrock/>`_, the runtime
framework for building agentic applications with AWS Bedrock. It leverages
Splunk distribution of ``opentelemetry-util-genai`` for producing telemetry in
semantic convention. `Core concepts, high-level usage and configuration
<https://github.com/signalfx/splunk-otel-python-contrib/>`_

Status: Alpha (APIs and produced telemetry are subject to change).


Installation
------------

.. code-block:: bash

    pip install splunk-otel-instrumentation-bedrock-agentcore


Quick Start
-----------

Manual Instrumentation (development/debugging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from bedrock_agentcore import BedrockAgentCoreApp
    from opentelemetry.instrumentation.bedrock_agentcore import (
        BedrockAgentCoreInstrumentor,
    )

    # Manual instrumentation, easy to debug in your IDE
    BedrockAgentCoreInstrumentor().instrument()

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def handler(event):
        return {"status": "success", "event": event}


Zero-code Instrumentation
^^^^^^^^^^^^^^^^^^^^^^^^^

In zero-code instrumentation mode, ensure you install opentelemetry-distribution
and run your app with the OpenTelemetry Bedrock AgentCore instrumentor enabled::

.. code-block:: bash

    opentelemetry-instrument python your_bedrock_agentcore_app.py

.. code-block:: python

    from bedrock_agentcore import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def handler(event):
        return {"status": "success", "event": event}


What Gets Instrumented
----------------------

This instrumentation captures:

- **BedrockAgentCoreApp** -> Mapped to ``Workflow`` spans
- **MemoryClient.retrieve_memories** -> Mapped to ``RetrievalInvocation`` spans
- **MemoryClient operations** -> Mapped to ``ToolCall`` spans
- **CodeInterpreter operations** -> Mapped to ``ToolCall`` spans
- **BrowserClient operations** -> Mapped to ``ToolCall`` spans

All spans are properly nested with correct parent-child relationships and include
rich attributes about the operation.

Testing Reference
-----------------

For a detailed implementation map, configuration matrix, wrapped SDK method
inventory, span attribute assertions, and suggested test cases, see
`docs/testing-reference.md <docs/testing-reference.md>`_.


Compositional Instrumentation
-----------------------------

This instrumentation focuses on Bedrock AgentCore runtime operations. For
complete observability, combine it with provider-specific or framework-specific
instrumentation.

AgentCore + Bedrock
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import (
        BedrockAgentCoreInstrumentor,
    )
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

    BedrockAgentCoreInstrumentor().instrument()
    BedrockInstrumentor().instrument()

Adds Bedrock API call spans with AWS-specific attributes and metrics.


AgentCore + Bedrock + Strands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import (
        BedrockAgentCoreInstrumentor,
    )
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    from opentelemetry.instrumentation.strands import StrandsInstrumentor

    BedrockAgentCoreInstrumentor().instrument()
    BedrockInstrumentor().instrument()
    StrandsInstrumentor().instrument()

Adds agent orchestration, runtime operations, and Bedrock model call telemetry.


Configuration
-------------

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Capture request/response content (disabled by default)
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

    # Content capture mode
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT

    # Select GenAI emitters
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event


Instrumentation Options
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from opentelemetry.instrumentation.bedrock_agentcore import (
        BedrockAgentCoreInstrumentor,
    )

    # Basic instrumentation
    BedrockAgentCoreInstrumentor().instrument()

    # With custom providers
    BedrockAgentCoreInstrumentor().instrument(
        tracer_provider=my_tracer_provider,
        meter_provider=my_meter_provider,
        logger_provider=my_logger_provider,
    )

    # Uninstrumentation
    BedrockAgentCoreInstrumentor().uninstrument()


Requirements
------------

- Python >= 3.10
- bedrock-agentcore >= 1.0.0
- OpenTelemetry API >= 1.38
- ``splunk-otel-util-genai`` >= 0.1.9


Trace Hierarchy Example
-----------------------

.. code-block::

    BedrockAgentCoreApp.entrypoint (Workflow)
    +-- Memory: retrieve_memories (RetrievalInvocation)
    +-- CodeInterpreter: execute_code (ToolCall)
    +-- Browser: take_control (ToolCall)


Each span includes rich attributes:

- ``gen_ai.system`` = "bedrock-agentcore"
- ``gen_ai.operation.name`` = "invoke_workflow" | "execute_tool" | "retrieval"
- Retrieval span names include the memory provider, for example
  ``retrieval bedrock-agentcore-memory``
- Framework-specific attributes, such as memory IDs, code interpreter session
  IDs, browser session IDs, and operation metadata


Examples
--------

The ``examples/manual`` directory contains a working example that exercises
BedrockAgentCoreApp, MemoryClient, CodeInterpreter, and BrowserClient
instrumentation with OTLP export.

.. code-block:: bash

    cd examples/manual
    pip install -r requirements.txt
    pip install -e ../../[instruments]

    export AWS_DEFAULT_REGION=us-west-2
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    python main.py


Testing
-------

Run the package tests from the repository root or this directory:

.. code-block:: bash

    pytest instrumentation-genai/opentelemetry-instrumentation-bedrock-agentcore/tests


Contributing
------------

Issues / PRs welcome in the main splunk-otel-python-contrib repository. This
module is alpha: feedback on attribute coverage, performance, and Bedrock
AgentCore surface expansion is especially helpful.


Links
-----

- `AWS Bedrock AgentCore <https://docs.aws.amazon.com/bedrock/>`_
- `OpenTelemetry Python <https://opentelemetry.io/docs/languages/python/>`_
- `Splunk GenAI Utilities <https://github.com/signalfx/splunk-otel-python-contrib>`_


License
-------

Apache-2.0
