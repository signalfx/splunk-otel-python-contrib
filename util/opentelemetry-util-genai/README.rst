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
