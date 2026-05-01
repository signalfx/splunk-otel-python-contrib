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
AgentCore provides the workflow/tool/retrieval spans, and this package adds the
child LLM spans that evaluation callbacks consume.

.. code-block:: python

    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    from opentelemetry.instrumentation.bedrock_agentcore import (
        BedrockAgentCoreInstrumentor,
    )

    BedrockAgentCoreInstrumentor().instrument()
    BedrockInstrumentor().instrument()

What Gets Instrumented
----------------------

- ``bedrock-runtime.Converse`` -> ``LLMInvocation``
- ``bedrock-runtime.ConverseStream`` -> streaming ``LLMInvocation``
- ``bedrock-runtime.InvokeModel`` -> conservative ``LLMInvocation``
- ``bedrock-runtime.InvokeModelWithResponseStream`` -> conservative streaming
  ``LLMInvocation``

Agent Runtime calls such as ``bedrock-agent-runtime.invoke_agent`` are not
instrumented by this package. Agent orchestration spans belong in AgentCore or
agent-framework instrumentation.

Configuration
-------------

Content capture follows the shared GenAI environment variables:

.. code-block:: bash

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS=true

When content capture is disabled, the instrumentation still emits model,
operation, token, finish reason, and request metadata, but message bodies and
tool arguments/results are omitted.

Telemetry Details
-----------------

The instrumentation sets:

- ``gen_ai.system`` = ``aws.bedrock``
- ``gen_ai.framework`` = ``boto3``
- ``gen_ai.request.model`` from ``modelId``
- ``gen_ai.provider.name`` inferred from the model ID
- request params such as temperature, top-p, max tokens, and stop sequences
- response ID, response model, finish reasons, and token usage when available
- ``gen_ai.request.stream`` and ``gen_ai.response.time_to_first_chunk`` for
  streaming calls

Requirements
------------

- Python >= 3.10
- boto3 >= 1.34.0
- splunk-otel-util-genai >= 0.1.9
