Multi-Agent Travel Planner Example
==================================

This example demonstrates a multi-agent travel planning workflow using the
OpenAI Agents SDK with OpenTelemetry instrumentation and support for OAuth2
authentication to custom LLM endpoints.

Features
--------

- Multi-agent architecture with specialized agents (Flight, Hotel, Activity, Coordinator)
- Manual OpenTelemetry SDK configuration (traces, metrics, logs, events)
- OAuth2 token management for custom LLM endpoints
- Backward compatible with standard OpenAI API
- Kubernetes CronJob deployment ready
- Docker containerization support

Agents
------

- **Flight Specialist**: Searches for flight options
- **Hotel Specialist**: Recommends accommodations
- **Activity Specialist**: Curates local activities and experiences
- **Travel Coordinator**: Orchestrates and synthesizes the final itinerary

Setup
-----

1. Copy `.env.example <.env.example>`_ to `.env` and configure your LLM provider:

   **Option 1: Standard OpenAI API**

   ::

       OPENAI_API_KEY=your-openai-api-key

   **Option 2: OAuth2 LLM Provider (custom endpoint)**

   ::

       LLM_CLIENT_ID=your-client-id
       LLM_CLIENT_SECRET=your-client-secret
       LLM_TOKEN_URL=https://your-identity-provider/oauth2/token
       LLM_BASE_URL=https://your-llm-gateway/openai/deployments
       OPENAI_MODEL_NAME=gpt-4o-mini

2. Create a virtual environment and install the dependencies:

   ::

       python3 -m venv .venv
       source .venv/bin/activate
       pip install -r requirements.txt
       pip install -e ../../  # Install local instrumentation package

Run
---

Execute the travel planner with manual instrumentation:

::

    python main.py --manual-instrumentation

Or run with zero-code instrumentation:

::

    opentelemetry-instrument python main.py

Expected Output
---------------

::

    [AUTH] Using OAuth2 authentication
    ✓ Manual OpenTelemetry instrumentation configured
    🌍 Multi-Agent Travel Planner
    ============================================================

    Origin: Seattle
    Destination: Paris
    Dates: 2024-02-15 to 2024-02-22

    ✈️  Flight Specialist - Searching for flights...
    Result: Top choice: SkyLine non-stop service Seattle->Paris...

    🏨 Hotel Specialist - Searching for hotels...
    Result: Grand Meridian near the historic centre...

    🎭 Activity Specialist - Curating activities...
    Result: Signature experiences in Paris...

    📝 Coordinator - Creating final itinerary...

    ============================================================
    ✅ Travel Itinerary Complete!
    ============================================================

    [Final itinerary output...]

    ================================================================================
    TELEMETRY OUTPUT BELOW
    ================================================================================

    [FLUSH] Starting telemetry flush
    [FLUSH] Flushing traces (timeout=30s)
    [FLUSH] Flushing metrics (timeout=30s)
    [FLUSH] Flushing logs (timeout=30s)
    [FLUSH] Telemetry flush complete

Docker
------

Build and run with Docker:

::

    # From repository root
    docker build -f instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/travel-planner/Dockerfile -t openai-agents-travel-planner .
    docker run --env-file .env openai-agents-travel-planner

Kubernetes
----------

Deploy as a CronJob:

::

    # Create secrets
    kubectl create secret generic llm-credentials \
      --from-literal=client-id=<your-client-id> \
      --from-literal=client-secret=<your-client-secret> \
      --from-literal=token-url=<your-token-url> \
      --from-literal=base-url=<your-base-url>

    # Apply CronJob
    kubectl apply -f cronjob.yaml

Project Structure
-----------------

::

    travel-planner/
    ├── main.py              # Multi-agent travel planner with OAuth2 support
    ├── util/
    │   ├── __init__.py
    │   └── oauth2_token_manager.py  # OAuth2 token management
    ├── requirements.txt     # Python dependencies
    ├── .env.example         # Environment variable template
    ├── Dockerfile           # Container build
    ├── cronjob.yaml         # Kubernetes CronJob spec
    └── README.rst           # This file
