Langsmith to GenAI Semantic Convention Translator
=================================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-util-genai-translator-langsmith.svg
   :target: https://pypi.org/project/splunk-otel-util-genai-translator-langsmith/

This package provides automatic translation of Langsmith-specific span attributes
to OpenTelemetry GenAI semantic convention compliant format.

Overview
--------

Langsmith is the observability platform for LangChain applications, providing
tracing, evaluation, and monitoring capabilities. This translator bridges
Langsmith's attribute format to the standardized GenAI semantic conventions.

Installation
------------

::

    pip install splunk-otel-util-genai-translator-langsmith

Usage
-----

The translator automatically enables when the package is installed. No additional
configuration is required.

To explicitly enable or configure:

.. code-block:: python

    from opentelemetry.util.genai.langsmith import enable_langsmith_translator

    # Enable with default configuration
    enable_langsmith_translator()

    # Enable with custom attribute transformations
    enable_langsmith_translator(
        attribute_transformations={
            "rename": {
                "my.custom.attr": "gen_ai.custom.attr"
            }
        }
    )

Disabling Auto-Enable
---------------------

Set the environment variable ``OTEL_INSTRUMENTATION_GENAI_LANGSMITH_DISABLE=true``
to disable automatic registration.

Attribute Mappings
------------------

The translator performs the following attribute mappings:

Content & Messages
~~~~~~~~~~~~~~~~~~

* ``langsmith.entity.input`` → ``gen_ai.input.messages``
* ``langsmith.entity.output`` → ``gen_ai.output.messages``
* ``gen_ai.prompt`` → ``gen_ai.input.messages``
* ``gen_ai.completion`` → ``gen_ai.output.messages``

Model & System
~~~~~~~~~~~~~~

* ``langsmith.metadata.ls_provider`` → ``gen_ai.system``
* ``langsmith.metadata.ls_model_name`` → ``gen_ai.request.model``
* ``langsmith.metadata.ls_model_type`` → ``gen_ai.operation.name``

Request Parameters
~~~~~~~~~~~~~~~~~~

* ``langsmith.metadata.ls_temperature`` → ``gen_ai.request.temperature``
* ``langsmith.metadata.ls_max_tokens`` → ``gen_ai.request.max_tokens``
* ``langsmith.metadata.ls_top_p`` → ``gen_ai.request.top_p``
* ``langsmith.metadata.ls_top_k`` → ``gen_ai.request.top_k``

Token Usage
~~~~~~~~~~~

* ``langsmith.token_usage.prompt_tokens`` → ``gen_ai.usage.input_tokens``
* ``langsmith.token_usage.completion_tokens`` → ``gen_ai.usage.output_tokens``
* ``langsmith.token_usage.total_tokens`` → ``gen_ai.usage.total_tokens``

Tool Calling
~~~~~~~~~~~~

* ``langsmith.tool.name`` → ``gen_ai.tool.call.name``
* ``langsmith.tool.id`` → ``gen_ai.tool.call.id``
* ``langsmith.tool.arguments`` → ``gen_ai.tool.call.arguments``
* ``langsmith.tool.output`` → ``gen_ai.tool.call.result``

Session & Run Tracking
~~~~~~~~~~~~~~~~~~~~~~

* ``langsmith.session_id`` → ``gen_ai.conversation.id``
* ``langsmith.thread_id`` → ``gen_ai.conversation.id``
* ``langsmith.run_id`` → ``gen_ai.run.id``

References
----------

* `Langsmith Documentation <https://docs.smith.langchain.com/>`_
* `GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/>`_
