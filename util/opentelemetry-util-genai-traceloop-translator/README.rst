OpenTelemetry GenAI Traceloop Translator
=========================================

This package automatically translates Traceloop-instrumented spans into OpenTelemetry GenAI semantic conventions.
It intercepts spans with ``traceloop.*`` attributes and creates corresponding spans with ``gen_ai.*`` attributes,
enabling seamless integration between Traceloop instrumentation and GenAI observability tools.

Mapping Table
-------------

============================== ================================ 
Traceloop Key                  Added Key                        
============================== ================================
``traceloop.workflow.name``    ``gen_ai.workflow.name``
``traceloop.entity.name``      ``gen_ai.agent.name``
``traceloop.entity.path``      ``gen_ai.workflow.path``
``traceloop.correlation.id``   ``gen_ai.conversation.id``
``traceloop.entity.input``     ``gen_ai.input.messages``
``traceloop.entity.output``    ``gen_ai.output.messages``
============================== ================================


Installation
------------
.. code-block:: bash

   pip install opentelemetry-util-genai-traceloop-translator

Quick Start (Automatic Registration)
-------------------------------------
The easiest way to use the translator is to simply import it - no manual setup required!

.. code-block:: python

   import os
   from openai import OpenAI


   from traceloop.sdk import Traceloop
   from traceloop.sdk.decorators import workflow

   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

   Traceloop.init(app_name="story_service")


   @workflow(name="streaming_story")
   def joke_workflow():
      stream = client.chat.completions.create(
         model="gpt-4o-2024-05-13",
         messages=[{"role": "user", "content": "Tell me a story about opentelemetry"}],
         stream=True,
      )

      for part in stream:
         print(part.choices[0].delta.content or "", end="")
      print()


   joke_workflow()
   # The translator automatically creates new gen_ai.* attributes based on the mapping.

Tests
-----
.. code-block:: bash

   pytest util/opentelemetry-util-genai-traceloop-translator/tests

