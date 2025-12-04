OpenTelemetry Weaviate Instrumentation
=======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-weaviate.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-weaviate/

This library allows tracing requests made by the Weaviate Python client to a Weaviate vector database.

Installation
------------

::

    pip install splunk-otel-instrumentation-weaviate


Usage
-----

Instrumenting all Weaviate clients
***********************************

When using the instrumentor, all clients will automatically be instrumented.

.. code-block:: python

    from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor
    import weaviate

    # Instrument Weaviate
    WeaviateInstrumentor().instrument()

    # This client will be automatically instrumented
    client = weaviate.connect_to_local()

    # Use the client as normal
    collection = client.collections.get("MyCollection")
    result = collection.query.fetch_objects(limit=10)


Retrieval Spans for RAG Pipelines
**********************************

For RAG (Retrieval-Augmented Generation) pipelines, use the manual span helpers to create structured traces where embedding and database operations are children of a parent retrieval span.

.. code-block:: python

    from opentelemetry.instrumentation.weaviate import (
        WeaviateInstrumentor,
        retrieval_span,
        embedding_span,
    )

    WeaviateInstrumentor().instrument()

    # Create retrieval span with embedding as child
    with retrieval_span(
        query_text="What are vector databases?",
        top_k=10,
        collection_name="Articles",
        embedding_model="nomic-embed-text"
    ) as span:
        # Child 1: Generate embedding
        with embedding_span(query, model="nomic-embed-text"):
            embedding = lm_studio_client.embeddings.create(input=query)
        
        # Child 2: Query Weaviate (auto-instrumented)
        results = collection.query.near_vector(
            near_vector=embedding.data[0].embedding,
            limit=10
        )
        
        # Add custom attributes
        span.set_attribute("db.retrieval.documents_retrieved", len(results.objects))

**Resulting span hierarchy:**

::

    db.retrieval.client
    ├─ generate_embedding
    └─ db.weaviate.collections.query.near_vector
        └─ db.weaviate.collections.query.get

**Builder pattern for complex scenarios:**

.. code-block:: python

    from opentelemetry.instrumentation.weaviate import RetrievalSpanBuilder

    builder = RetrievalSpanBuilder(
        query_text="search text",
        top_k=10,
        collection_name="Articles",
        embedding_model="nomic-embed-text",
        # Custom attributes via kwargs
        user_id="user123",
        session_id="session456"
    )

    with builder.span() as span:
        # Your retrieval logic here
        pass

See ``RETRIEVAL_SPANS.md`` for detailed documentation.


Examples
--------

The ``examples/manual/`` directory contains working examples:

* ``example_v4.py`` - Comprehensive example showing various Weaviate v4 operations
* ``example_rag_pipeline.py`` - Complete RAG pipeline with LM Studio embeddings and manual retrieval spans
* ``example_manual_retrieval_spans.py`` - Different patterns for manual retrieval spans

Running the examples
********************

1. Install dependencies::

    pip install weaviate-client>=4.0.0 opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

2. Start a local Weaviate instance::

    docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest

3. Run an example::

    cd examples/manual
    python3 example_v4.py

4. (Optional) Configure OTLP endpoint::

    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"


Supported Versions
------------------

This instrumentation supports Weaviate client versions 3.x and 4.x.

References
----------

* `OpenTelemetry Weaviate Instrumentation <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/weaviate/weaviate.html>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `Weaviate Documentation <https://weaviate.io/developers/weaviate>`_
