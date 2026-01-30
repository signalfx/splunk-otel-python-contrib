OpenTelemetry LlamaIndex Instrumentation Example (Zero-Code)
===========================================================

This is an example of how to instrument LlamaIndex calls using zero-code
instrumentation with :code:`opentelemetry-instrument`.

This example provides an HTTP server (:code:`server.py`) that uses
LlamaIndex with Cisco CircuIT API. When run with the :code:`opentelemetry-instrument`
command, it automatically exports traces, metrics, and events to
an OTLP-compatible endpoint.

Dependencies
------------

The following packages are required:

- :code:`llama-index>=0.14.0` - LlamaIndex framework
- :code:`python-dotenv>=1.0.0` - Environment variable management
- :code:`requests>=2.31.0` - HTTP client for CircuIT API
- :code:`opentelemetry-distro` - OpenTelemetry distribution with auto-instrumentation
- :code:`opentelemetry-exporter-otlp` - OTLP exporter for traces and metrics

Install with:

.. code-block:: console

    pip install -r requirements.txt

Environment Variables
---------------------

**Required:**

- ``CIRCUIT_BASE_URL`` - CircuIT API endpoint (e.g., https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini/chat/completions)
- ``CIRCUIT_TOKEN_URL`` - OAuth2 token endpoint (e.g., https://id.cisco.com/oauth2/default/v1/token)
- ``CIRCUIT_CLIENT_ID`` - CircuIT OAuth2 client ID
- ``CIRCUIT_CLIENT_SECRET`` - CircuIT OAuth2 client secret
- ``CIRCUIT_APP_KEY`` - CircuIT application key

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

1. **Create** a :code:`.env` file with your CircuIT credentials:

   .. code-block:: console

       CIRCUIT_BASE_URL=https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini/chat/completions
       CIRCUIT_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token
       CIRCUIT_CLIENT_ID=your-client-id
       CIRCUIT_CLIENT_SECRET=your-client-secret
       CIRCUIT_APP_KEY=your-app-key

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

1. Export environment variables:

   .. code-block:: console

       export CIRCUIT_BASE_URL="https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini/chat/completions"
       export CIRCUIT_TOKEN_URL="https://id.cisco.com/oauth2/default/v1/token"
       export CIRCUIT_CLIENT_ID="your-client-id"
       export CIRCUIT_CLIENT_SECRET="your-client-secret"
       export CIRCUIT_APP_KEY="your-app-key"
       export OTEL_SERVICE_NAME="llamaindex-zero-code-server"
       export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
       export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event"

2. Run the server with zero-code instrumentation:

   .. code-block:: console

       opentelemetry-instrument \
           --traces_exporter otlp \
           --metrics_exporter otlp \
           python server.py

3. The server will start on port 8080. Test it with curl:

   .. code-block:: console

       curl -X POST http://localhost:8080/chat \
         -H "Content-Type: application/json" \
         -d '{"message": "What is 2+2?", "system_prompt": "You are a helpful assistant"}'

4. Check for traces and metrics in your observability backend. If you see warnings about
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
      -e CIRCUIT_BASE_URL=https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini/chat/completions \
      -e CIRCUIT_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token \
      -e CIRCUIT_CLIENT_ID=your-client-id \
      -e CIRCUIT_CLIENT_SECRET=your-client-secret \
      -e CIRCUIT_APP_KEY=your-app-key \
      -e OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
      -e OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
      -e OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event \
      llamaindex-zero-code-server:latest

Kubernetes Deployment
---------------------

1. Create the namespace:

   .. code-block:: console

       kubectl create namespace llamaindex-zero-code

2. Create the CircuIT credentials secret:

   .. code-block:: console

       kubectl apply -f circuit-secret.yaml

3. Deploy the server:

   .. code-block:: console

       kubectl apply -f deployment.yaml

4. Deploy the cronjob client:

   .. code-block:: console

       kubectl apply -f cronjob.yaml

5. Verify the deployment:

   .. code-block:: console

       kubectl get pods -n llamaindex-zero-code
       kubectl logs -n llamaindex-zero-code -l app=llamaindex-zero-code-server

The cronjob will automatically send test requests every 30 minutes during
business hours (8 AM - 5 PM Pacific Time, weekdays).
