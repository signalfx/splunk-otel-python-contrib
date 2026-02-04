OpenTelemetry LlamaIndex Instrumentation Example (Zero-Code)
===========================================================

This is an example of how to instrument LlamaIndex calls using zero-code
instrumentation with :code:`opentelemetry-instrument`.

This example provides an HTTP server (:code:`server.py`) that uses
LlamaIndex with either:

- an OAuth2-protected LLM gateway (via :code:`LLM_*` variables), or
- standard OpenAI API key auth (via :code:`OPENAI_API_KEY`).

When run with the :code:`opentelemetry-instrument`
command, it automatically exports traces, metrics, and events to
an OTLP-compatible endpoint.

Dependencies
------------

The following packages are required:

- :code:`llama-index>=0.14.0` - LlamaIndex framework
- :code:`python-dotenv>=1.0.0` - Environment variable management
- :code:`requests>=2.31.0` - HTTP client for OAuth2 gateway API
- :code:`opentelemetry-distro` - OpenTelemetry distribution with auto-instrumentation
- :code:`opentelemetry-exporter-otlp` - OTLP exporter for traces and metrics

Install with:

.. code-block:: console

    pip install -r requirements.txt

Environment Variables
---------------------

Choose one auth mode:

**Mode A - OAuth2 Gateway (required fields):**

- ``LLM_BASE_URL`` - LLM gateway chat completions endpoint
- ``LLM_TOKEN_URL`` - OAuth2 token endpoint
- ``LLM_CLIENT_ID`` - OAuth2 client ID
- ``LLM_CLIENT_SECRET`` - OAuth2 client secret

**Mode A - OAuth2 Gateway (optional fields):**

- ``LLM_APP_KEY`` - app key header or request metadata key
- ``LLM_SCOPE`` - OAuth2 scope
- ``OPENAI_MODEL_NAME`` - model label used for metadata (default: ``gpt-4o-mini``)

**Mode B - OpenAI API key:**

- ``OPENAI_API_KEY`` - OpenAI API key
- ``OPENAI_MODEL_NAME`` - model name (default: ``gpt-4o-mini``)

**OpenTelemetry Configuration:**

- ``OTEL_SERVICE_NAME`` - Service name for telemetry (default: "llamaindex-zero-code-server")
- ``OTEL_EXPORTER_OTLP_ENDPOINT`` - OTLP endpoint (default: http://localhost:4318 for HTTP, http://localhost:4317 for gRPC)
- ``OTEL_EXPORTER_OTLP_PROTOCOL`` - Protocol: "http/protobuf" or "grpc" (default: http/protobuf)
- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` - Set to "true" to capture full prompt/response content
- ``OTEL_INSTRUMENTATION_GENAI_EMITTERS`` - Set to "span_metric_event" to enable metrics and events
- ``OTEL_TRACES_EXPORTER`` - Traces exporter type (default: "otlp")
- ``OTEL_METRICS_EXPORTER`` - Metrics exporter type (default: "otlp")

Telemetry Data
--------------

**Span Attributes:**

- ``gen_ai.system`` - AI system name (e.g., "llamaindex")
- ``gen_ai.request.model`` - Model name (e.g., "gpt-4o-mini")
- ``gen_ai.operation.name`` - Operation type (e.g., "chat", "completion")
- ``gen_ai.usage.input_tokens`` - Number of input tokens
- ``gen_ai.usage.output_tokens`` - Number of output tokens
- ``llm.request.type`` - Request type (e.g., "chat")
- ``server.address`` - CircuIT API server address

**Metrics:**

- ``gen_ai.client.operation.duration`` - Histogram of LLM operation durations
- ``gen_ai.client.token.usage`` - Counter for token usage (input and output)

**Events:**

- ``gen_ai.content.prompt`` - Prompt content (when CAPTURE_MESSAGE_CONTENT=true)
- ``gen_ai.content.completion`` - Completion content (when CAPTURE_MESSAGE_CONTENT=true)

Setup
-----

Set the required environment variables in the :code:`.env` file.

For more information about AI app configuration, see:
`Configure the Python agent for AI applications <https://help.splunk.com/en/splunk-observability-cloud/manage-data/instrument-back-end-services/instrument-back-end-applications-to-send-spans-to-splunk-apm/instrument-a-python-application/configure-the-python-agent-for-ai-applications>`_.

**LLM configuration credentials:**

   .. code-block:: console

       # OpenAI API Key
       OPENAI_API_KEY=sk-YOUR_API_KEY

       # Or OAuth2 LLM Provider (for enterprise deployments)
       LLM_CLIENT_ID=<your-oauth2-client-id>
       LLM_CLIENT_SECRET=<your-oauth2-client-secret>
       LLM_TOKEN_URL=https://<your-identity-provider>/oauth2/token
       LLM_BASE_URL=https://<your-llm-gateway>/openai/deployments
       LLM_APP_KEY=<your-app-key>  # Optional
       LLM_SCOPE=<scope>  # Optional

       # Example: Cisco CircuIT Configuration
       LLM_BASE_URL=https://your-circuit-gateway.cisco.com/v1
       LLM_TOKEN_URL=https://your-circuit-gateway.cisco.com/oauth2/token
       LLM_CLIENT_ID=your_client_id_here
       LLM_CLIENT_SECRET=your_secret_here
       LLM_APP_KEY=llamaindex-zero-code-demo
       LLM_SCOPE=api.read

**OpenTelemetry configuration settings:**

.. code-block:: console

       # Service Identity
       OTEL_SERVICE_NAME=llamaindex-zero-code-example
       OTEL_RESOURCE_ATTRIBUTES=deployment.environment=demo

       # OTLP Exporter
       OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
       OTEL_EXPORTER_OTLP_PROTOCOL=grpc

       # Logs
       OTEL_LOGS_EXPORTER=otlp
       OTEL_PYTHON_LOG_CORRELATION=true
       OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true

       # Metrics
       OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta

**(Optional) GenAI evaluation settings and debug settings:**

.. code-block:: console

       # DeepEval custom LLM-as-a-Judge settings
       DEEPEVAL_LLM_BASE_URL=https://<your-llm-gateway>/openai/deployments/<model>
       DEEPEVAL_LLM_MODEL=gpt-4o-mini
       DEEPEVAL_LLM_PROVIDER=openai
       DEEPEVAL_LLM_CLIENT_ID=<your-oauth2-client-id>
       DEEPEVAL_LLM_CLIENT_SECRET=<your-oauth2-client-secret>
       DEEPEVAL_LLM_TOKEN_URL=https://<your-identity-provider>/oauth2/token
       DEEPEVAL_LLM_CLIENT_APP_NAME=<your-app-key>
       DEEPEVAL_FILE_SYSTEM=READ_ONLY

       # Debug settings
       OTEL_INSTRUMENTATION_GENAI_DEBUG=false
       OTEL_GENAI_EVAL_DEBUG_SKIPS=false
       OTEL_GENAI_EVAL_DEBUG_EACH=false

1. **Create** a :code:`.env` file with the settings above. Example:

   .. code-block:: console

       OPENAI_API_KEY=sk-...
       OPENAI_MODEL_NAME=gpt-4o-mini

2. Set up a virtual environment:

   .. code-block:: console

       python3 -m venv .venv
       source .venv/bin/activate
       pip install -r requirements.txt

3. **(Optional)** Install a development version of the instrumentation:

   .. code-block:: console

       # From the repository root
       pip install -e util/opentelemetry-util-genai
       pip install -e instrumentation-genai/opentelemetry-instrumentation-llamaindex

Run Locally
-----------

1. Start the Splunk Distribution of the OpenTelemetry Collector (example using Docker):

   .. code-block:: console

       docker run -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector

2. Load the environment variables from :code:`.env`:

   .. code-block:: console

       set -a && source .env && set +a

3. Run the server with zero-code instrumentation:

   .. code-block:: console

       opentelemetry-instrument \
           --traces_exporter otlp \
           --metrics_exporter otlp \
           python server.py

4. Send a curl request to generate traces:

   .. code-block:: console

       curl -X POST http://localhost:8080/chat \
         -H "Content-Type: application/json" \
         -d '{"message": "What is 2+2?", "system_prompt": "You are a helpful assistant"}'

5. Check for traces and metrics in your observability backend. If you see warnings about
   connection failures, ensure your OpenTelemetry Collector is running on the configured endpoint.

Docker
------

To build and run with Docker:

.. code-block:: console

    # From the repository root
    docker build --platform linux/amd64 \
      -f instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/zero-code/Dockerfile \
      -t llamaindex-zero-code-server:latest .
    
    # Run with environment variables
    docker run -p 8080:8080 \
      -e LLM_BASE_URL=https://<your-llm-gateway>/openai/deployments/gpt-4o-mini/chat/completions \
      -e LLM_TOKEN_URL=https://<your-identity-provider>/oauth2/token \
      -e LLM_CLIENT_ID=your-client-id \
      -e LLM_CLIENT_SECRET=your-client-secret \
      -e LLM_APP_KEY=your-app-key \
      -e OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
      -e OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
      -e OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event \
      llamaindex-zero-code-server:latest

Kubernetes Deployment
---------------------

1. Create the namespace:

   .. code-block:: console

       kubectl create namespace llamaindex-zero-code

2. Create OAuth2 gateway secret (optional, if using OAuth2 mode):

   .. code-block:: console

       kubectl create secret generic llm-credentials \
         -n llamaindex-zero-code \
         --from-literal=base-url=https://<your-llm-gateway>/openai/deployments/gpt-4o-mini/chat/completions \
         --from-literal=token-url=https://<your-identity-provider>/oauth2/token \
         --from-literal=client-id=<your-client-id> \
         --from-literal=client-secret=<your-client-secret> \
         --from-literal=app-key=<your-app-key> \
         --from-literal=scope=<your-scope>

3. Create OpenAI secret (optional, if using OpenAI mode):

   .. code-block:: console

       kubectl create secret generic openai-credentials \
         -n llamaindex-zero-code \
         --from-literal=api-key=sk-...

4. Deploy the server:

   .. code-block:: console

       kubectl apply -f deployment.yaml

5. Deploy the cronjob client:

   .. code-block:: console

       kubectl apply -f cronjob.yaml

6. Verify the deployment:

   .. code-block:: console

       kubectl get pods -n llamaindex-zero-code
       kubectl logs -n llamaindex-zero-code -l app=llamaindex-zero-code-server

The cronjob will automatically send test requests every 30 minutes during
business hours (8 AM - 5 PM Pacific Time, weekdays).
