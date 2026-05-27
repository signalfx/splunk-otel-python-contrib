OpenTelemetry Bedrock Runtime Instrumentation
=============================================

This package instruments AWS Bedrock Runtime model calls made through
``boto3``/``botocore`` and emits GenAI ``LLMInvocation`` telemetry through
``splunk-otel-util-genai``.

Installation
------------

.. code-block:: bash

    pip install splunk-otel-instrumentation-bedrock

Usage
-----

.. code-block:: python

    import boto3
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

    BedrockInstrumentor().instrument()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    response = client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    )

Composition With AgentCore
--------------------------

Use this package with AgentCore instrumentation when your application uses
``BedrockAgentCoreApp`` and calls Bedrock Runtime from inside the entrypoint.
AgentCore owns the agent parent spans, and this package adds the child LLM
spans that evaluation callbacks consume.

.. code-block:: python

    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    from opentelemetry.instrumentation.bedrock_agentcore import (
        BedrockAgentCoreInstrumentor,
    )

    BedrockAgentCoreInstrumentor().instrument()
    BedrockInstrumentor().instrument()

Example
-------

In the repository, see
``examples/manual``
for a runnable Bedrock Runtime example that can also enable AgentCore
instrumentation. It uses console span export by default so you can verify that
Bedrock Runtime LLM spans nest under active AgentCore spans when both
instrumentors are enabled.

What Gets Instrumented
----------------------

- ``bedrock-runtime.Converse`` -> ``LLMInvocation``
- ``bedrock-runtime.ConverseStream`` -> streaming ``LLMInvocation``
- ``bedrock-runtime.InvokeModel`` -> provider-aware ``LLMInvocation``
- ``bedrock-runtime.InvokeModelWithResponseStream`` -> provider-aware streaming
  ``LLMInvocation`` for supported streamed JSON chunk formats

Agent Runtime calls such as ``bedrock-agent-runtime.invoke_agent`` are not
instrumented by this package. Agent orchestration spans belong in AgentCore or
agent-framework instrumentation.

Configuration
-------------

Content capture follows the shared GenAI environment variables:

.. code-block:: bash

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS=true

Useful flags for Bedrock Runtime and AgentCore composition:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Environment variable
     - Purpose
   * - ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
     - Controls whether captured prompt and completion message content is
       emitted by the shared GenAI emitters. Use ``SPAN_AND_EVENT`` when
       evals and exported telemetry both need message bodies.
   * - ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS``
     - Controls whether Bedrock tool definitions are serialized into emitted
       telemetry. Leave disabled when tool schemas are large or sensitive.
   * - ``DISABLE_ADOT_OBSERVABILITY``
     - Set to ``true`` in AgentCore deployments that export to your own OTLP
       collector so AgentCore does not also send telemetry through AWS ADOT
       observability.

The instrumentation always populates message bodies and tool arguments on the
Python invocation objects so evaluations can consume them. The shared GenAI
emitters use the content-capture setting to decide whether those values are
emitted as span attributes or log events.

For zero-code instrumentation, disable this package with
``OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=bedrock``.

InvokeModel Coverage
--------------------

``InvokeModel`` and ``InvokeModelWithResponseStream`` support generic JSON
metadata extraction plus provider-specific shapes aligned with upstream
OpenTelemetry botocore Bedrock behavior:

- Amazon Titan
- Amazon Nova
- Anthropic Claude
- Cohere Command and Command R
- Meta Llama
- Mistral

For non-streaming ``InvokeModel`` responses, the instrumentation reads
``botocore.response.StreamingBody`` only to parse supported JSON response
shapes, then replaces the response body with a fresh stream so application code
can still read it.

Telemetry Details
-----------------

The instrumentation sets:

- ``gen_ai.system`` = ``aws.bedrock``
- ``gen_ai.framework`` = ``boto3``
- ``gen_ai.request.model`` from ``modelId``
- ``gen_ai.provider.name`` = ``aws.bedrock``
- request params such as temperature, top-p, max tokens, and stop sequences
- response ID, response model, finish reasons, and token usage when available
- ``gen_ai.request.stream`` and ``gen_ai.response.time_to_first_chunk`` for
  streaming calls

Requirements
------------

- Python >= 3.10
- boto3 >= 1.34.0
- splunk-otel-util-genai >= 0.1.9
