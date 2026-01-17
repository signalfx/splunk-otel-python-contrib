OpenTelemetry GenAI Utilities Evals for Deepeval (opentelemetry-util-genai-evals-deepeval)
==========================================================================================

This package plugs the `deepeval <https://github.com/confident-ai/deepeval>`_ metrics
prompt template suite into the OpenTelemetry GenAI evaluation pipeline. When it is
installed a ``deepeval`` evaluator is registered automatically and, unless
explicitly disabled, is executed for every LLM/agent invocation.

Unlike Deepeval’s standard workflow, this evaluator does **not** invoke Deepeval’s
evaluation runner / metric classes. Instead, it:

* Reuses Deepeval metric templates as rubric text only
* Performs a single **batched** LLM-as-a-judge call per invocation (multiple
  metrics in one prompt)
* Extracts token usage directly from the LLM client response telemetry (e.g.
  OpenAI ``usage.prompt_tokens`` / ``usage.completion_tokens``)

Installation
------------

Install the evaluator (and its runtime dependencies) from PyPI:

.. code-block:: bash

    pip install opentelemetry-util-genai-evals-deepeval

The command pulls in ``opentelemetry-util-genai``, ``deepeval`` and ``openai``
automatically so the evaluator is ready to use right after installation.

Requirements
------------

* ``opentelemetry-util-genai`` together with ``deepeval`` and ``openai`` –
  these are installed automatically when you install this package.
* An OpenAI Chat Completions compatible model reachable via the OpenAI SDK.
  By default the evaluator uses OpenAI's ``gpt-4o-mini`` model, so ensure
  ``OPENAI_API_KEY`` is available. To override the model, set
  ``DEEPEVAL_EVALUATION_MODEL`` (or ``DEEPEVAL_MODEL`` / ``OPENAI_MODEL``).

Configuration
-------------

Use ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS`` to select the metrics that
should run. Leaving the variable unset enables every registered evaluator with its
default metric set. Examples:

* ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=Deepeval`` – run the default
  Deepeval bundle (Bias, Toxicity, Answer Relevancy, Hallucination, Sentiment).
* ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=Deepeval(LLMInvocation(bias(threshold=0.75)))`` –
  override the Bias threshold for LLM invocations and skip the remaining metrics.
* ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=none`` – disable the evaluator entirely.

Results are emitted through the standard GenAI evaluation emitters (events,
metrics, spans). Each metric includes helper attributes such as
``deepeval.success``, ``deepeval.threshold`` and any evaluation model metadata
returned by Deepeval. Metrics that cannot run because required inputs are missing
(for example Faithfulness without a ``retrieval_context``) are marked as
``label="skipped"`` and carry a ``deepeval.error`` attribute so you can wire the
necessary data or disable that metric explicitly.

Default OpenAI Usage
--------------------

If you're using OpenAI directly, no additional configuration is needed. Simply set
``OPENAI_API_KEY`` and the evaluator will work out of the box:

.. code-block:: bash

    export OPENAI_API_KEY="sk-your-openai-api-key"

Note: this evaluator currently uses the OpenAI SDK directly and does not implement
the legacy ``DEEPEVAL_LLM_*`` custom-provider configuration described by Deepeval.
