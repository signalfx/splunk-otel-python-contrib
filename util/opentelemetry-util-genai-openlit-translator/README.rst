OpenTelemetry GenAI OpenLit Translator
=========================================

This package automatically translates openlit sdk instrumented spans into OpenTelemetry GenAI semantic conventions.
It intercepts spans with ```gen_ai.*``` openlit specific attributes and creates corresponding spans with ``gen_ai.*`` semantic convention compliant attributes,
enabling seamless integration between openlit instrumentation and GenAI observability tools.

Mapping Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old Key (OpenLit)
     - New Key (OTel SemConv)
   * - ``gen_ai.completion.0.content``
     - ``gen_ai.output.messages``
   * - ``gen_ai.prompt.0.content``
     - ``gen_ai.input.messages``
   * - ``gen_ai.prompt``
     - ``gen_ai.input.messages``
   * - ``gen_ai.completion``
     - ``gen_ai.output.messages``
   * - ``gen_ai.content.prompt``
     - ``gen_ai.input.messages``
   * - ``gen_ai.content.completion``
     - ``gen_ai.output.messages``
   * - ``gen_ai.request.embedding_dimension``
     - ``gen_ai.embeddings.dimension.count``
   * - ``gen_ai.token.usage.input``
     - ``gen_ai.usage.input_tokens``
   * - ``gen_ai.token.usage.output``
     - ``gen_ai.usage.output_tokens``
   * - ``gen_ai.llm.provider``
     - ``gen_ai.provider.name``
   * - ``gen_ai.llm.model``
     - ``gen_ai.request.model``
   * - ``gen_ai.llm.temperature``
     - ``gen_ai.request.temperature``
   * - ``gen_ai.llm.max_tokens``
     - ``gen_ai.request.max_tokens``
   * - ``gen_ai.llm.top_p``
     - ``gen_ai.request.top_p``
   * - ``gen_ai.operation.type``
     - ``gen_ai.operation.name``
   * - ``gen_ai.output_messages``
     - ``gen_ai.output.messages``
   * - ``gen_ai.session.id``
     - ``gen_ai.conversation.id``
   * - ``gen_ai.openai.thread.id``
     - ``gen_ai.conversation.id``
   * - ``gen_ai.tool.args``
     - ``gen_ai.tool.call.arguments``
   * - ``gen_ai.tool.result``
     - ``gen_ai.tool.call.result``
   * - ``gen_ai.vectordb.name``
     - ``db.system.name``
   * - ``gen_ai.vectordb.search.query``
     - ``db.query.text``
   * - ``gen_ai.vectordb.search.results_count``
     - ``db.response.returned_rows``


Installation
------------
.. code-block:: bash

   pip install opentelemetry-util-genai-openlit-translator

Quick Start (Automatic Registration)
-------------------------------------
The easiest way to use the translator is to simply import it - no manual setup required!

.. code-block:: python

   from openai import OpenAI
   import openlit
   from dotenv import load_dotenv
   import os
   import traceback

   load_dotenv()

   try:
      openlit.init(otlp_endpoint="http://0.0.0.0:4318")

      client = OpenAI(
         api_key=os.getenv("OPENAI_API_KEY")
      )

      chat_completion = client.chat.completions.create(
         messages=[
               {
                  "role": "user",
                  "content": "What is LLM Observability?",
               }
         ],
         model="gpt-3.5-turbo",
      )
      print("response:", chat_completion.choices[0].message.content)
   except Exception as e:
      print(f"An error occurred: {e}")
      traceback.print_exc()


Tests
-----
.. code-block:: bash

   pytest util/opentelemetry-util-genai-openlit-translator/tests

