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
   Re-run it whenever the token expires (update your shell env or `.env`).
4. Render the config with your environment values:
   ```bash
   ./scripts/render_config.py litellm-config.example.yaml litellm-config.yaml
   ```
5. Start the local CircuIT shim + LiteLLM stack (docker compose v2):
   ```bash
   docker compose up --build
   ```
   - The compose file builds a tiny container for `circuit_shim.py` and exposes it on
     `${CIRCUIT_SHIM_PORT:-5001}`.
   - LiteLLM runs beside it and mounts `litellm-config.yaml`, which now points to the shim
     service via `${CIRCUIT_API_BASE:-http://circuit-shim:5001/}` instead of
     `https://chat-ai.cisco.com`.

Verify the proxy is up:
```bash
curl -H "Authorization: Bearer ${LITELLM_API_KEY:-litellm-demo-key}" \
  http://localhost:${LITELLM_PORT:-4000}/v1/models
```

> Note: For production you would bake the token minting/refresh logic directly
> into the LiteLLM deployment. These steps are intentionally manual to keep the
> demo lightweight.

## Quick LangChain smoke test

Before running the full supervisor demo you can send a single LangChain request
through LiteLLM to make sure creds/config are correct:

```bash
python instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo/litellm_langchain_smoke.py \
  "Say hi and tell me today's date"
```

The script reads the same `.env` settings as `main.py` (base URL, API key,
model, timeout) and falls back to `LITELLM_SMOKE_PROMPT` if you omit the CLI
argument. Seeing a friendly assistant response confirms LiteLLM and the shim are
reachable from your workstation.

## Configuration

1. Copy the example environment file:
   ```bash
   cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo
   cp .env.example .env
   ```
2. Edit `.env` to point to your LiteLLM proxy base URL and API key (and to store
   the CircuIT token if you prefer file-based env instead of exporting). When
   using a remote proxy, make sure the networking rules allow connections from
   your workstation/container.

Environment variables:

- `CISCO_CLIENT_ID`, `CISCO_CLIENT_SECRET`, `CISCO_APP_KEY` – Cisco CircuIT creds
  consumed by `circuit_token_helper.py` and passed downstream in user metadata.
- `CIRCUIT_ACCESS_TOKEN` – bearer token returned by `circuit_token_helper.py`. The
  token expires; refresh and update this value before re-running `docker compose`.
- `CIRCUIT_UPSTREAM_BASE` – the real Cisco CircuIT base URL (defaults to
  `https://chat-ai.cisco.com`).
- `CIRCUIT_API_BASE` – where LiteLLM reaches the shim (defaults to
  `http://circuit-shim:5001/`). Change this if you host the shim elsewhere.
- `CIRCUIT_SHIM_PORT` / `LITELLM_PORT` – exposed ports for each container.
- `LITELLM_PROXY_KEY`, `LITELLM_API_KEY` – key LiteLLM expects in the
  `Authorization` header.
- `LITELLM_BASE_URL` – OpenAI-compatible base URL (default
  `http://localhost:4000/v1`).
- `LITELLM_MODEL` – Model alias the demo uses when calling LiteLLM (defaults to
  `gpt-4o`).
- `LITELLM_MODEL_NAME`, `LITELLM_UPSTREAM_MODEL` – names used inside the LiteLLM
  config for the exposed alias and actual upstream deployment.
- `LITELLM_TEMPERATURE`, `LITELLM_TIMEOUT`, `LITELLM_MAX_RETRIES` – optional
  tuning knobs passed directly to `ChatOpenAI`.
- `LITELLM_SMOKE_PROMPT` – default prompt for `litellm_langchain_smoke.py` if you
  don't pass one.
- `TRAVEL_DEMO_PROMPT` – optional default prompt if you don't pass one on the CLI.

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
