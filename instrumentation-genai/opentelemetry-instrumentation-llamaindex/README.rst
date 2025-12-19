OpenTelemetry LlamaIndex Instrumentation
=========================================

This library provides automatic instrumentation for LlamaIndex applications using OpenTelemetry.

Installation
------------

Development installation::

    # Install the package in editable mode
    cd instrumentation-genai/opentelemetry-instrumentation-llamaindex
    pip install -e .
    
    # Install test dependencies
    pip install -e ".[test]"
    
    # Install util-genai (required for telemetry)
    cd ../../util/opentelemetry-util-genai
    pip install -e .


Quick Start
-----------

.. code-block:: python

    import os
    from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    
    # Enable metrics (default is spans only)
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"
    
    # Setup tracing
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    
    # Setup metrics
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    # Enable instrumentation with providers
    LlamaindexInstrumentor().instrument(
        tracer_provider=trace.get_tracer_provider(),
        meter_provider=meter_provider
    )
    
    # Use LlamaIndex as normal
    from llama_index.llms.openai import OpenAI
    from llama_index.core.llms import ChatMessage, MessageRole
    
    llm = OpenAI(model="gpt-3.5-turbo")
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    response = llm.chat(messages)


Running Tests
-------------

**LLM Tests**:

.. code-block:: bash

    # Set environment variables
    export OPENAI_API_KEY=your-api-key
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric
    
    # Run the test
    cd tests
    python test_llm_instrumentation.py

**Embedding Tests**:

.. code-block:: bash

    # Set environment variables
    export OPENAI_API_KEY=your-api-key
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric
    
    # Run the test
    cd tests
    python test_embedding_instrumentation.py


Expected Output
---------------

**LLM Span Attributes**::

    {
        "gen_ai.framework": "llamaindex",
        "gen_ai.request.model": "gpt-3.5-turbo",
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": 24,
        "gen_ai.usage.output_tokens": 7
    }

**Embedding Span Attributes**::

    {
        "gen_ai.operation.name": "embeddings",
        "gen_ai.request.model": "text-embedding-3-small",
        "gen_ai.provider.name": "openai",
        "gen_ai.embeddings.dimension.count": 1536
    }

**Metrics**::

    Metric: gen_ai.client.operation.duration
      Duration: 0.6900 seconds
      Count: 1
    
    Metric: gen_ai.client.token.usage
      Token type: input, Sum: 24, Count: 1
      Token type: output, Sum: 7, Count: 1


Key Implementation Differences from LangChain
----------------------------------------------

**1. Event-Based Callbacks**

LlamaIndex uses ``on_event_start(event_type, ...)`` and ``on_event_end(event_type, ...)`` 
instead of LangChain's method-based callbacks (``on_llm_start``, ``on_llm_end``).

Event types are dispatched via ``CBEventType`` enum::

    CBEventType.LLM       # LLM invocations (chat, complete)
    CBEventType.AGENT     # Agent steps (not yet instrumented)
    CBEventType.EMBEDDING # Embedding operations (get_text_embedding, get_text_embedding_batch)

**2. Handler Registration**

LlamaIndex uses ``handlers`` list::

    callback_manager.handlers.append(handler)

LangChain uses ``inheritable_handlers``::

    callback_manager.inheritable_handlers.append(handler)

**3. Response Structure**

LlamaIndex ``ChatMessage`` uses ``blocks`` (list of TextBlock objects)::

    message.content  # Computed property from blocks[0].text
    
LangChain uses simple strings::

    message.content  # Direct string property

**4. Token Usage**

LlamaIndex returns objects (not dicts)::

    response.raw.usage.prompt_tokens      # Object attribute
    response.raw.usage.completion_tokens  # Object attribute
    
LangChain returns dicts::

    response["usage"]["prompt_tokens"]      # Dict key
    response["usage"]["completion_tokens"]  # Dict key


Supported Features
------------------

**LLM Operations**

* ✅ Chat completion (``llm.chat()``, ``llm.stream_chat()``)
* ✅ Text completion (``llm.complete()``, ``llm.stream_complete()``)
* ✅ Token usage tracking
* ✅ Model name detection
* ✅ Framework attribution

**Embedding Operations**

* ✅ Single text embedding (``embed_model.get_text_embedding()``)
* ✅ Batch embedding (``embed_model.get_text_embedding_batch()``)
* ✅ Query embedding (``embed_model.get_query_embedding()``)
* ✅ Provider detection (OpenAI, Azure, AWS Bedrock, Google, Cohere, HuggingFace, Ollama, and more)
* ✅ Dimension count tracking
* ✅ Input text capture

**Provider Detection**

Embedding instrumentation automatically detects the provider from class names:

* **OpenAI**: ``OpenAIEmbedding``
* **Azure**: ``AzureOpenAIEmbedding``
* **AWS**: ``BedrockEmbedding``
* **Google**: ``GeminiEmbedding``, ``VertexTextEmbedding``, ``GooglePaLMEmbedding``
* **Cohere**: ``CohereEmbedding``
* **HuggingFace**: ``HuggingFaceEmbedding``, ``HuggingFaceInferenceAPIEmbedding``
* **Ollama**: ``OllamaEmbedding``
* **Anthropic**: ``AnthropicEmbedding``
* **MistralAI**: ``MistralAIEmbedding``
* **Together**: ``TogetherEmbedding``
* **Fireworks**: ``FireworksEmbedding``
* **Voyage**: ``VoyageEmbedding``
* **Jina**: ``JinaEmbedding``


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `LlamaIndex <https://www.llamaindex.ai/>`_
* `LlamaIndex Callbacks <https://docs.llamaindex.ai/en/stable/module_guides/observability/callbacks/>`_
