OpenTelemetry OpenAI Agents Manual Instrumentation Example
==========================================================

This example demonstrates how to manually configure the OpenTelemetry SDK
alongside the OpenAI Agents instrumentation, with support for OAuth2
authentication to custom LLM endpoints.

Features
--------

- Manual OpenTelemetry SDK configuration (traces, metrics, logs)
- OAuth2 token management for custom LLM endpoints
- Backward compatible with standard OpenAI API
- Kubernetes CronJob deployment ready
- Docker containerization support

Running `main.py <main.py>`_ produces spans for the end-to-end agent run,
including tool invocations and model generations. Spans are exported through
OTLP/gRPC to the endpoint configured in the environment.

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
       pip install ../../  # Install local instrumentation package

Run
---

Execute the sample with ``dotenv`` so the environment variables from ``.env``
are applied:

::

    dotenv run -- python main.py

Or run directly if environment variables are already set:

::

    python main.py

Expected Output
---------------

::

    [AUTH] Using OAuth2 authentication
    
    [SUCCESS] Agent execution completed
    Agent response:
    Based on the weather forecast for Barcelona...

    ================================================================================
    TELEMETRY OUTPUT BELOW
    ================================================================================

    [FLUSH] Starting telemetry flush
    [FLUSH] Flushing traces (timeout=30s)
    [FLUSH] Flushing metrics (timeout=30s)
    [FLUSH] Telemetry flush complete

Docker
------

Build and run with Docker:

::

    # From repository root
    docker build -f instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/manual/Dockerfile -t openai-agents-manual .
    docker run --env-file .env openai-agents-manual

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

    manual/
    ├── main.py              # Main example with OAuth2 support
    ├── util/
    │   ├── __init__.py
    │   └── oauth2_token_manager.py  # OAuth2 token management
    ├── requirements.txt     # Python dependencies
    ├── .env.example         # Environment variable template
    ├── Dockerfile           # Container build
    ├── cronjob.yaml         # Kubernetes CronJob spec
    └── README.rst           # This file
