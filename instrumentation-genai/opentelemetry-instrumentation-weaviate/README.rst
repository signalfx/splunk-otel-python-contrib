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

When using the instrumentor, all Weaviate clients will automatically be instrumented.

.. code-block:: python

    from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor
    import weaviate

    # Instrument Weaviate
    WeaviateInstrumentor().instrument()

    # This client will be automatically instrumented
    client = weaviate.connect_to_local()

    # Use the client as normal - all operations will be traced
    collection = client.collections.get("MyCollection")
    result = collection.query.fetch_objects(limit=10)


Configuration
-------------

Content Capture
***************

By default, document content from similarity search operations is **not** captured in span events.
To enable content capture, set the following environment variable:

.. code-block:: bash

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

When enabled, similarity search operations will emit ``weaviate.document`` events containing:

* ``db.weaviate.document.content`` - The document content/properties
* ``db.weaviate.document.distance`` - Distance metric (if available)
* ``db.weaviate.document.certainty`` - Certainty score (if available)
* ``db.weaviate.document.score`` - Relevance score (if available)
* ``db.weaviate.document.query`` - The search query (if available)

The document count is always captured in the ``db.weaviate.documents.count`` span attribute regardless of this setting.


Examples
--------

The ``examples/manual/`` directory contains a working example:

* ``example_v4.py`` - Comprehensive example showing various Weaviate v4 operations with automatic tracing

Running the example
*******************

1. Install dependencies::

    pip install weaviate-client>=4.0.0 opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

2. Start a local Weaviate instance::

    docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest

3. Run the example::

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
