OpenTelemetry OpenAI Instrumentation
====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-openai.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-openai/

This library allows tracing LLM requests and logging of messages made by the
`OpenAI Python API library <https://pypi.org/project/openai/>`_. It also captures
the duration of the operations and the number of tokens used as metrics.

Many LLM platforms support the OpenAI SDK. This means systems such as the following are observable with this instrumentation when accessed using it:

.. list-table:: OpenAI Compatible Platforms
   :widths: 40 25
   :header-rows: 1

   * - Name
     - gen_ai.system
   * - `Azure OpenAI <https://github.com/openai/openai-python?tab=readme-ov-file#microsoft-azure-openai>`_
     - ``az.ai.openai``
   * - `Gemini <https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/>`_
     - ``gemini``
   * - `Perplexity <https://docs.perplexity.ai/api-reference/chat-completions>`_
     - ``perplexity``
   * - `xAI <https://x.ai/api>`_ (Compatible with Anthropic)
     - ``xai``
   * - `DeepSeek <https://api-docs.deepseek.com/>`_
     - ``deepseek``
   * - `Groq <https://console.groq.com/docs/openai>`_
     - ``groq``
   * - `MistralAI <https://docs.mistral.ai/api/>`_
     - ``mistral_ai``

Installation
------------

If your application is already instrumented with OpenTelemetry, add this
package to your requirements.
::

    pip install splunk-otel-instrumentation-openai

If you don't have an OpenAI application, yet, try our `examples <examples>`_
which only need a valid OpenAI API key.

Check out `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up OpenAI instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

Instrumenting all clients
*************************

When using the instrumentor, all clients will automatically trace OpenAI operations including chat completions and embeddings.
You can also optionally capture prompts and completions as log events.

Make sure to configure OpenTelemetry tracing, logging, and events to capture all telemetry emitted by the instrumentation.

.. code-block:: python

    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    client = OpenAI()
    # Chat completion example
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a short poem on open telemetry."},
        ],
    )
    
    # Embeddings example
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Generate vector embeddings for this text"
    )

Enabling message content
*************************

Message content such as the contents of the prompt, completion, function arguments and return values
are not captured by default. To capture message content as log events, set the environment variable
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` to `true`.

See the `upstream OpenTelemetry documentation <https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation-genai/opentelemetry-instrumentation-openai-v2#enabling-message-content>`_ for more details.

Suppressing nested instrumentation
***********************************

When using multiple instrumentations together (e.g., LangChain + OpenAI), the higher-level
instrumentation automatically sets ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`` in the
OpenTelemetry context to prevent duplicate spans for the same underlying LLM call.

This is handled transparently — **no user configuration is needed**. For example, when
LangChain instrumentation is active alongside OpenAI instrumentation, you will see
LangChain spans without redundant nested OpenAI spans.

.. note::

   This is not an environment variable and cannot be configured for zero-code instrumentation.
   It is a context key managed programmatically by instrumentation libraries.

Uninstrument
************

To uninstrument clients, call the uninstrument method:

.. code-block:: python

    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()
    # ...

    # Uninstrument all clients
    OpenAIInstrumentor().uninstrument()

References
----------
* `Splunk OpenTelemetry OpenAI Instrumentation <https://github.com/signalfx/splunk-otel-python-contrib/tree/main/instrumentation-genai/opentelemetry-instrumentation-openai-v2>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_

