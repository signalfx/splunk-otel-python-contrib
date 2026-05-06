Bedrock Runtime and AgentCore Composition Example
=================================================

This example shows how to enable Bedrock Runtime instrumentation by itself and
with AgentCore instrumentation in the same process.

The default mode calls ``bedrock-runtime.Converse`` directly and emits an
``LLMInvocation`` span. The AgentCore mode enables
``BedrockAgentCoreInstrumentor`` first, then enables ``BedrockInstrumentor`` and
runs the same Bedrock Runtime call from an AgentCore entrypoint. In that mode,
the Bedrock Runtime LLM span should be a child of the active AgentCore workflow
span.

Setup
-----

From this directory:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e ../../../../util/opentelemetry-util-genai
    pip install -e ../..

For AgentCore mode, also install the Bedrock AgentCore SDK and the AgentCore
instrumentation package that provides:

- ``bedrock_agentcore.runtime.BedrockAgentCoreApp``
- ``opentelemetry.instrumentation.bedrock_agentcore.BedrockAgentCoreInstrumentor``

If you are testing from adjacent local branches or worktrees, install those
packages in editable mode before running with ``--agentcore``.

Run Bedrock Runtime Only
------------------------

.. code-block:: bash

    export AWS_REGION=us-west-2
    export BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
    python main.py

The example uses ``ConsoleSpanExporter`` by default so the exported span JSON is
printed locally. To export through OTLP instead:

.. code-block:: bash

    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    python main.py --exporter otlp

Run With AgentCore
------------------

.. code-block:: bash

    python main.py --agentcore

Equivalent environment-variable form:

.. code-block:: bash

    export BEDROCK_EXAMPLE_ENABLE_AGENTCORE=true
    python main.py

For AgentCore server mode:

.. code-block:: bash

    python main.py --agentcore --serve-agentcore

What To Check
-------------

In console output, compare the trace and parent IDs:

- Bedrock Runtime-only mode should show one Bedrock LLM span.
- AgentCore mode should show an AgentCore workflow span and a Bedrock Runtime
  LLM span in the same trace.
- The Bedrock Runtime LLM span should have the AgentCore workflow span as its
  parent when the Bedrock call runs inside the AgentCore entrypoint.

Useful environment variables:

.. code-block:: bash

    export BEDROCK_PROMPT="Explain span parenting in one sentence."
    export OTEL_SERVICE_NAME=bedrock-runtime-agentcore-example
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
