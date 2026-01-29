OpenTelemetry GenAI Utilities
==============================

Minimal Overview
----------------
Utility function to provide APIs and data types to ease instrumentation of Generative AI workloads using OpenTelemetry semantic conventions.

Example usage for LLM Invocation.

.. code-block:: python

  from opentelemetry.util.genai.handler import get_telemetry_handler
  handler = get_telemetry_handler()
  user_input = "Hello"
  inv = LLMInvocation(request_model="gpt-5-nano", provider="openai",
    input_messages=[InputMessage(role="user", parts=[Text(user_input)])])
  handler.start_llm(inv)
  # your code which actuall invokes llm here
  # response = client.chat.completions.create(...)
  # ....
  inv.output_messages = [OutputMessage(role="assistant", parts=[Text("Hi!")], finish_reason="stop")]
  handler.stop_llm(inv)


See the example in ``examples/agentic_example.py`` for a full agent + LLM invocation flow.

Concurrent Evaluation Mode
--------------------------

For high-throughput LLM-as-a-Judge evaluations, enable concurrent processing:

.. code-block:: bash

    # Enable concurrent evaluation with 4 workers
    export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
    export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
    
    # Optional: Set bounded queue for backpressure
    export OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=100

Configuration options:

* ``OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT`` - Enable concurrent mode (default: ``false``)
* ``OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS`` - Number of worker threads (default: ``4``)
* ``OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE`` - Bounded queue size, ``0`` for unbounded (default: ``0``)

Concurrent mode processes multiple evaluations in parallel, significantly improving throughput
when using LLM-as-a-Judge metrics like DeepEval's Bias, Toxicity, and Answer Relevancy.

Metrics
-------

This library emits histogram metrics with explicit bucket boundaries per `OpenTelemetry GenAI semantic conventions <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md>`_:

**Duration Metrics** (unit: seconds)

- ``gen_ai.client.operation.duration`` - Duration of GenAI client operations
- ``gen_ai.workflow.duration`` - Duration of GenAI workflows
- ``gen_ai.agent.duration`` - Duration of agent operations

Bucket boundaries: ``[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]``

**Token Usage Metrics** (unit: tokens)

- ``gen_ai.client.token.usage`` - Number of input and output tokens used

Bucket boundaries: ``[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]``

**Evaluation Metrics** (unit: score 0-1)

- ``gen_ai.evaluation.score`` - GenAI evaluation score (default, when ``OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC`` is unset or ``true``)
- ``gen_ai.evaluation.<name>`` - Individual evaluation metrics (when ``OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC=false``)

Bucket boundaries: ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]``

Further Documentation
---------------------
For architecture, design rationale, and broader usage patterns please consult:
* `Core concepts, high-level usage and setup <https://github.com/signalfx/splunk-otel-python-contrib/>`_
* ``README.packages.architecture.md`` – extensibility architecture & emitter pipeline design.

Those documents cover configuration (environment variables, content capture modes, evaluation emission, extensibility via entry points) and release/stability notes.
  
Support & Stability
-------------------
GenAI semantic conventions are incubating.

License
-------
Apache 2.0 (see ``LICENSE``).
