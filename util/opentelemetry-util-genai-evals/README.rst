OpenTelemetry Util GenAI Evaluations
====================================

This package provides the base evaluation manager and builtin evaluators for
``opentelemetry-util-genai``. It is loaded dynamically by the core GenAI
telemetry utilities via the completion callback plugin mechanism.

Features
--------

* **Evaluation Manager** - Background processing of LLM and agent evaluations
* **Concurrent Processing** - Multi-worker parallel evaluation for high throughput
* **Bounded Queue** - Backpressure support to prevent memory exhaustion
* **Async Evaluation** - Native async support for LLM-as-a-Judge evaluators

Concurrent Evaluation Mode
--------------------------

Enable concurrent processing for improved throughput with LLM-based evaluations:

.. code-block:: bash

    # Enable concurrent mode with 4 workers
    export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
    export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
    
    # Optional: Bounded queue for backpressure
    export OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=100

**Sequential Mode (Default):**

* Single worker thread processes evaluations one at a time
* Guaranteed ordering of evaluation results
* Lower resource consumption

**Concurrent Mode:**

* Multiple worker threads with asyncio event loops
* Parallel LLM API calls for faster evaluation throughput
* Recommended for LLM-as-a-Judge evaluators (e.g., DeepEval)

Creating Custom Evaluators
--------------------------

Implement the ``Evaluator`` base class and optionally override async methods
for native async support:

.. code-block:: python

    from opentelemetry.util.genai.evals.base import Evaluator, EvaluationResult
    
    class MyEvaluator(Evaluator):
        @property
        def supports_async(self) -> bool:
            return True  # Enable native async evaluation
        
        async def evaluate_llm_async(self, invocation):
            # Your async evaluation logic
            return [EvaluationResult(name="my_metric", score=0.95)]
