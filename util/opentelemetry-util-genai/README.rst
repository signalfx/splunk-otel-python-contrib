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
  inv = LLMInvocation(request_model="gpt-4", provider="openai",
    input_messages=[InputMessage(role="user", parts=[Text(user_input)])])
  handler.start_llm(inv)
  # your code which actuall invokes llm here
  # response = client.chat.completions.create(...)
  # ....
  inv.output_messages = [OutputMessage(role="assistant", parts=[Text("Hi!")], finish_reason="stop")]
  handler.stop_llm(inv)


See the example in ``examples/agentic_example.py`` for a full agent + LLM invocation flow.

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
