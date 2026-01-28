OpenTelemetry LangChain Rate Limiter Test
=========================================

This example focuses on testing evaluation rate limiting for LangChain
instrumentation. It runs multiple LLM calls and lets the evaluation queue
process them under different rate limiter settings.

When :code:`main.py <main.py>` is run, it exports traces (and optionally logs)
to an OTLP-compatible endpoint. Traces include details such as the chain name,
LLM usage, token usage, and durations for each operation.

Environment variables:

- ``OTEL_INSTRUMENTATION_LANGCHAIN_CAPTURE_MESSAGE_CONTENT=true`` can be used
  to capture full prompt/response content.
- ``OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS`` controls which evaluators run.
- ``OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS`` and
  ``OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST`` control rate limiting.

Setup
-----

1. **Update** the :code:`.env <.env>` file with any environment variables you
   need (e.g., your OpenAI key, or :code:`OTEL_EXPORTER_OTLP_ENDPOINT` if not
   using the default http://localhost:4317).
2. Set up a virtual environment:

   .. code-block:: console

       python3 -m venv .venv
       source .venv/bin/activate
       pip install "python-dotenv[cli]"
       pip install -r requirements.txt

3. **(Optional)** Install a development version of the new instrumentation:

   .. code-block:: console

       # E.g., from a local path or a git repo
       pip install -e /path/to/opentelemetry-python-contrib/instrumentation-genai/opentelemetry-instrumentation-langchain

Run
---

Run the example like this:

.. code-block:: console

    dotenv run -- opentelemetry-instrument python main.py

You should see an example chain output while traces are exported to your
configured observability tool.

Rate limiter scenarios
----------------------

Edit :code:`.env <.env>` and choose one scenario by uncommenting it.

- **Scenario A (baseline)**: :code:`RPS=1, BURST=3` (evaluations should appear quickly)
- **Scenario B (moderate)**: :code:`RPS=0.2, BURST=2`
- **Scenario C (strict)**: :code:`RPS=0.05, BURST=1` (slowest)

To disable the rate limiter entirely:

.. code-block:: console

    OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS=0
