OpenTelemetry OpenAI Agents Zero-Code Instrumentation Example
=============================================================

This example shows how to capture telemetry from OpenAI Agents using
``opentelemetry-instrument`` for zero-code instrumentation, with support
for OAuth2 authentication to custom LLM endpoints.

Features
--------

- Zero-code OpenTelemetry instrumentation via ``opentelemetry-instrument``
- OAuth2 token management for custom LLM endpoints
- Backward compatible with standard OpenAI API
- Kubernetes CronJob deployment ready
- Docker containerization support

When `main.py <main.py>`_ is executed, spans describing the agent workflow are
exported to the configured OTLP endpoint. The spans include details such as the
operation name, tool usage, and token consumption (when available).

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

Execute the sample via ``opentelemetry-instrument`` so the OpenAI Agents
instrumentation is activated automatically:

::

    opentelemetry-instrument python main.py

Or use the run script for proper telemetry flushing:

::

    ./run.sh

Expected Output
---------------

::

    [INIT] Starting zero-code instrumented OpenAI Agents application
    [AUTH] Using OAuth2 authentication
    
    [SUCCESS] Agent execution completed
    Agent response:
    Based on the weather forecast for Barcelona...

    [FLUSH] Force flushing telemetry providers...
    [FLUSH] Flushing traces (timeout=30s)
    [FLUSH] Flushing metrics (timeout=30s)
    [FLUSH] Telemetry flush complete

Docker
------

Build and run with Docker:

::

    # From repository root
    docker build -f instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/zero-code/Dockerfile -t openai-agents-zero-code .
    docker run --env-file .env openai-agents-zero-code

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

    zero-code/
    ├── main.py              # Main example with OAuth2 support
    ├── run.sh               # Wrapper script for telemetry flushing
    ├── util/
    │   ├── __init__.py
    │   └── oauth2_token_manager.py  # OAuth2 token management
    ├── requirements.txt     # Python dependencies
    ├── .env.example         # Environment variable template
    ├── Dockerfile           # Container build
    ├── cronjob.yaml         # Kubernetes CronJob spec
    └── README.rst           # This file
