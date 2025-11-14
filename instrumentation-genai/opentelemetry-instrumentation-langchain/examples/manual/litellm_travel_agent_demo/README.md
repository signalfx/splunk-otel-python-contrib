# LiteLLM Travel Agent Demo

This example clones the Traceloop/LangGraph travel-booking supervisor but routes
all chat completions through a LiteLLM proxy (OpenAI-compatible). Point any demo
apps or deepeval runs at LiteLLM and let the proxy manage Cisco CircuIT OAuth.

## Prerequisites

1. Install the shared manual-example dependencies:
   ```bash
   pip install -r instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/requirements.txt
   ```
2. A running LiteLLM proxy that already knows how to mint CircuIT tokens (see
   project-level docs/plan). For local testing we assume it listens on
   `http://localhost:4000/v1` and accepts a static key like
   `Authorization: Bearer litellm-demo-key`.

## Bringing up LiteLLM

1. Switch to the demo directory:
   ```bash
   cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo
   ```
2. Export your Cisco credentials (fill them in `.env` or export manually).
3. Mint a CircuIT token for LiteLLM to forward to the upstream endpoint:
   ```bash
   export CIRCUIT_ACCESS_TOKEN=$(./circuit_token_helper.py)
   ```
   The helper reads `CISCO_CLIENT_ID`, `CISCO_CLIENT_SECRET` and `CISCO_APP_KEY` from the
   environment (or you can pass them with `--client-id`, `--client-secret`, and `--app-key`).
   The helper builds the HTTP Basic auth header exactly as the CircuIT API requires by
   base64-encoding the string "${CISCO_CLIENT_SECRET}:${CISCO_APP_KEY}" and setting
   `Authorization: Basic <base64>` for the token request. It then prints the bearer token.
   Re-run it whenever the token expires.
4. Render the config with your environment values:
   ```bash
   ./scripts/render_config.py litellm-config.example.yaml litellm-config.yaml
   ```
5. Start LiteLLM via Docker (replace the path with your workspace root):
   ```bash
   docker run --rm -it \
     -p 4000:4000 \
     -e CIRCUIT_ACCESS_TOKEN="$CIRCUIT_ACCESS_TOKEN" \
     -e LITELLM_PROXY_KEY=${LITELLM_API_KEY:-litellm-demo-key} \
     -v $(pwd)/litellm-config.yaml:/app/proxy_server_config.yaml\
     ghcr.io/berriai/litellm:main \
     --config /app/litellm-config.yaml
   ```

Verify the proxy is up:
```bash
curl -H "Authorization: Bearer ${LITELLM_API_KEY:-litellm-demo-key}" \
  http://localhost:4000/v1/models
```

> Note: For production you would bake the token minting/refresh logic directly
> into the LiteLLM deployment. These steps are intentionally manual to keep the
> demo lightweight.

## Configuration

1. Copy the example environment file:
   ```bash
   cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo
   cp .env.example .env
   ```
2. Edit `.env` to point to your LiteLLM proxy base URL and API key. When using a
   remote proxy, make sure the networking rules allow connections from your
   workstation/container.

Environment variables:

- `LITELLM_BASE_URL` – OpenAI-compatible base URL (default `http://localhost:4000/v1`).
- `LITELLM_API_KEY` – Key that LiteLLM expects in the `Authorization` header.
- `LITELLM_MODEL` – Model alias that LiteLLM exposes (defaults to `gpt-4o`).
- `LITELLM_TEMPERATURE`, `LITELLM_TIMEOUT`, `LITELLM_MAX_RETRIES` – optional
  tuning knobs passed directly to `ChatOpenAI`.
- `TRAVEL_DEMO_PROMPT` – optional default prompt if you don't pass one on the CLI.
- `CISCO_CLIENT_ID`, `CISCO_CLIENT_SECRET`, `CISCO_APP_KEY` – Cisco CircuIT creds
  consumed by `circuit_token_helper.py` and passed downstream in user metadata.

## Running the demo

```bash
# Activate your env so the .env values are loaded (direnv, dotenv, or manual export)
python instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo/main.py \
  "book a flight from BOS to JFK and a stay at McKittrick Hotel"
```

If you omit the prompt argument, the script falls back to `TRAVEL_DEMO_PROMPT` or
its baked-in default. Streaming output shows supervisor updates and individual
agent steps. All completions flow through LiteLLM, so once the proxy is wired to
CircuIT the same demo can run in local Docker or k8s environments.
