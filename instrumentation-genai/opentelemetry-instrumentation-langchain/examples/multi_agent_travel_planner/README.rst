Multi-Agent Travel Planner Sample
=================================

This example shows how to orchestrate a small team of LangChain agents with
`LangGraph <https://python.langchain.com/docs/langgraph/>`_ while the
OpenTelemetry LangChain instrumentation captures GenAI spans and forwards them
to an OTLP collector.

The graph contains four specialists (coordinator, flights, hotels, activities)
and a final synthesiser node that produces an itinerary.  Each specialist relies
on a simple, deterministic tool so you can run the example without any external
travel APIs while still observing tool spans wired up to the agent calls.

Prerequisites
-------------

* Python 3.10+
* One of the following LLM credential sets:

  - An OpenAI API key with access to ``gpt-4o-mini`` (or adjust ``OPENAI_MODEL``)
  - Cisco CircuIT access with ``CISCO_APP_KEY`` plus either ``CISCO_CIRCUIT_TOKEN``
    or OAuth credentials (``CISCO_CLIENT_ID`` and ``CISCO_CLIENT_SECRET``). Set
    ``TRAVEL_LLM_PROVIDER=circuit`` to activate the CircuIT integration.  Optional
    overrides: ``CIRCUIT_DEFAULT_DEPLOYMENT``, ``CIRCUIT_UPSTREAM_BASE`` and
    ``CIRCUIT_TOKEN_CACHE``
* A running OTLP collector (gRPC on ``localhost:4317`` by default)

Setup
-----

.. code-block:: bash

   cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   # Copy the sample environment file and update values as needed.
   cp .env.example .env
   source .env

Run the sample
--------------

.. code-block:: bash

   # From this directory, after activating the virtual environment and sourcing
   # your environment variables:
   python main.py

The script prints each agent's contribution followed by the final itinerary.
At the same time it streams OTLP traces.  You should see:

* A root span named ``invoke_agent travel_multi_agent_planner`` that captures
  the overall orchestration, including ``gen_ai.input.messages`` and a preview
  of the final plan.
* LangChain instrumentation spans for each agent's LLM invocation with
  ``gen_ai.provider.name=openai`` and ``service.name`` derived from
  ``OTEL_SERVICE_NAME``.

Using Cisco CircuIT
-------------------

To route the demo through Cisco CircuIT's OpenAI-compatible endpoint:

.. code-block:: bash

   export TRAVEL_LLM_PROVIDER=circuit
   export CISCO_APP_KEY=your-app-key

   # Option 1: provide a static access token
   export CISCO_CIRCUIT_TOKEN=token-from-cisco

   # Option 2: let the demo mint tokens via OAuth client credentials
    export CISCO_CLIENT_ID=your-client-id
    export CISCO_CLIENT_SECRET=your-client-secret
    # Optional cache file for minted tokens (defaults to a temp path)
    export CIRCUIT_TOKEN_CACHE=/tmp/circuit_travel_demo.json

    # Optional overrides if you are proxying via LiteLLM or a CircuIT shim
    export CIRCUIT_API_BASE=http://localhost:4000/v1
    export CIRCUIT_DEFAULT_DEPLOYMENT=gpt-5-nano

    # Optional: force OAuth minting even if a static token is present
    export TRAVEL_FORCE_CIRCUIT_OAUTH=1

    # Optional: disable connection debug prints (default is enabled)
    export TRAVEL_DEBUG_CONNECTIONS=0


Run ``python main.py`` after exporting these variables.  The example automatically
includes the CircuIT ``appkey`` metadata and refreshes OAuth tokens on demand.
When ``TRAVEL_DEBUG_CONNECTIONS`` is enabled the script prints the token source,
cache location and a redacted preview so you can confirm whether a static value or
an OAuth-minted token is being used.

Tear down
---------

Deactivate the virtual environment when you are done:

.. code-block:: bash

   deactivate
