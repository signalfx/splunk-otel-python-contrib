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

Quick Start
-----------
The translator automatically registers when Traceloop is initialized. See the examples for complete usage:

- **Basic Example**: `examples/traceloop_processor_example.py <examples/traceloop_processor_example.py>`_ - Demonstrates workflow, agent, task, and tool decorators with automatic span translation.

- **Evaluation Example**: `examples/traceloop_eval_example.py <examples/traceloop_eval_example.py>`_ - Shows how to run GenAI evaluations on translated spans.

.. code-block:: bash

   # Run basic example
   python examples/traceloop_processor_example.py

   # Run evaluation example
   python examples/traceloop_eval_example.py

Evaluations
-----------

In order to enable GenAI evaluations for Traceloop spans, the corresponding packages must be installed:

.. code-block:: bash

   # install evaluation packages
   pip install opentelemetry-util-genai-evals opentelemetry-util-genai-evals-deepeval

Also, see `.env.example` for environment variable setup, which includes setting up configuration for the evaluation provider.

The translator enables GenAI evaluations by converting Traceloop spans to the standardized ``gen_ai.*`` format.
Once translated, LLM / Agent spans can be evaluated using metrics like:

- **Answer Relevancy** - How relevant is the response to the input
- **Faithfulness** - Is the response grounded in context
- **Bias/Toxicity** - Content safety checks

Evaluations require a compatible evaluation provider (e.g., DeepEval). See `examples/traceloop_eval_example.py <examples/traceloop_eval_example.py>`_ for a complete example.

Tests
-----

To run the test suite, use the following command:
.. code-block:: bash

   pytest util/opentelemetry-util-genai-traceloop-translator/tests

