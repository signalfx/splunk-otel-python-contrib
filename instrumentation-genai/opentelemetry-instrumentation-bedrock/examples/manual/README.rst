Bedrock Runtime and AgentCore Composition Example
=================================================

This example shows how to enable Bedrock Runtime instrumentation by itself and
with AgentCore instrumentation in the same process.

The default mode calls ``bedrock-runtime.Converse`` directly and emits an
``LLMInvocation`` span. The AgentCore mode enables
``BedrockAgentCoreInstrumentor`` first, then enables ``BedrockInstrumentor`` and
runs the same Bedrock Runtime call from an AgentCore entrypoint. In that mode,
the Bedrock Runtime LLM span should be a child of the active AgentCore parent
span.

Setup
-----

From this directory:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

The base requirements support the default Bedrock Runtime-only mode. The
AgentCore SDK and AgentCore instrumentation are optional because they are only
needed for ``python main.py --with-agentcore``. To run AgentCore mode, install
the optional AgentCore dependencies listed as comments in ``requirements.txt``.

The example sets local defaults for all environment variables it reads. The
same defaults are listed in ``.env.example`` for shell-based workflows.

To load them explicitly in your shell:

.. code-block:: bash

    set -a
    source .env.example
    set +a

AgentCore mode needs the Bedrock AgentCore SDK and the AgentCore
instrumentation package that provides:

- ``bedrock_agentcore.runtime.BedrockAgentCoreApp``
- ``opentelemetry.instrumentation.bedrock_agentcore.BedrockAgentCoreInstrumentor``
- ``bedrock_agentcore.memory.client.MemoryClient``
- ``bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter``
- ``bedrock_agentcore.tools.browser_client.BrowserClient``

Use published package versions when they are available. If you are testing from
adjacent local branches or worktrees, install the commented editable packages in
``requirements.txt`` before running with ``--with-agentcore``.

Run Bedrock Runtime Only
------------------------

.. code-block:: bash

    python main.py

The example exports to OTLP by default. It uses
``OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317`` unless you override the
endpoint. It configures OTLP span, metric, and log exporters so evaluation
metrics and ``gen_ai.evaluation.result`` log events can be exported alongside
the LLM spans:

.. code-block:: bash

    OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.example:4317 python main.py

To print span JSON locally instead of sending telemetry to a collector:

.. code-block:: bash

    BEDROCK_EXAMPLE_EXPORTER=console python main.py

Run With Evals
--------------

Enable an evaluator before the first instrumentor is created:

.. code-block:: bash

    export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(toxicity,bias))"
    export BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS=60
    python main.py

For a local smoke test that does not need DeepEval or judge-model credentials,
use the built-in length evaluator:

.. code-block:: bash

    export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="length(LLMInvocation(length))"
    BEDROCK_EXAMPLE_EXPORTER=console python main.py

The example waits up to ``BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS`` for async eval
work to finish before shutting down exporters. Set it to ``0`` to skip waiting.

Run With AgentCore
------------------

.. code-block:: bash

    python main.py --with-agentcore

Equivalent environment-variable form:

.. code-block:: bash

    export BEDROCK_EXAMPLE_ENABLE_AGENTCORE=true
    python main.py

AgentCore mode also performs best-effort AgentCore capability calls around the
Bedrock Runtime call, following the AgentCore manual example:

- ``MemoryClient.list_memories`` and, if needed,
  ``MemoryClient.create_or_get_memory``
- ``MemoryClient.retrieve_memories`` after the Bedrock Runtime call
- ``MemoryClient.create_event`` after the Bedrock Runtime call
- ``CodeInterpreter.start``, ``execute_code``, and ``stop``
- ``BrowserClient.start``, ``take_control``, and ``stop``

These calls are non-fatal. If your AWS account lacks permission or a service is
not available in the selected region, the example prints a skip message and
continues. The Bedrock Runtime LLM call runs first so a slow optional AgentCore
capability does not block the primary LLM span. If
``BEDROCK_AGENTCORE_MEMORY_ID`` is not set, the example finds or creates a
memory by ``BEDROCK_AGENTCORE_MEMORY_NAME``.

To use an existing memory instead of creating or finding one by name:

.. code-block:: bash

    export BEDROCK_AGENTCORE_MEMORY_ID=your-memory-id
    export BEDROCK_AGENTCORE_MEMORY_NAMESPACE=bedrock-runtime-agentcore-example
    python main.py --with-agentcore

These AgentCore capability calls are the default behavior for
``--with-agentcore``. To run only the Bedrock Runtime LLM call, omit
``--with-agentcore``.

For AgentCore server mode:

.. code-block:: bash

    export BEDROCK_EXAMPLE_SERVE_AGENTCORE=true
    python main.py --with-agentcore

What To Check
-------------

In your collector, compare the trace and parent IDs. If you run with
``BEDROCK_EXAMPLE_EXPORTER=console``, compare those IDs in console output
instead:

- The example prints ``Trace ID: <trace-id>`` when the first span starts, so
  you can find the trace even when exporting with OTLP.
- Bedrock Runtime-only mode should show one Bedrock LLM span.
- AgentCore mode should show an AgentCore parent span and a Bedrock Runtime
  LLM span in the same trace.
- AgentCore mode should also show AgentCore memory, code interpreter, and
  browser child spans when those capabilities are available.
- The Bedrock Runtime LLM span should have the active AgentCore span as its
  parent when the Bedrock call runs inside the AgentCore entrypoint.

Useful environment variables:

.. code-block:: bash

    export BEDROCK_PROMPT="Explain span parenting in one sentence."
    export BEDROCK_EXAMPLE_EXPORTER=otlp
    export BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS=60
    export BEDROCK_AGENTCORE_MEMORY_NAME=bedrockRuntimeAgentCoreExampleMemory
    export BEDROCK_AGENTCORE_MEMORY_NAMESPACE=bedrock-runtime-agentcore-example
    export BEDROCK_AGENTCORE_MEMORY_ACTOR_ID=bedrock-runtime-agentcore-example-user
    export BEDROCK_AGENTCORE_MEMORY_SESSION_ID=bedrock-runtime-agentcore-example-session
    export OTEL_SERVICE_NAME=bedrock-runtime-agentcore-example
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT
    export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(toxicity,bias))"
    export DISABLE_ADOT_OBSERVABILITY=true

For AgentCore deployments that export to your own OTLP endpoint, keep
``DISABLE_ADOT_OBSERVABILITY=true`` so AgentCore does not also send telemetry
through AWS ADOT observability.
