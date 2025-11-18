# LiteLLM Travel Agent Demo

This example clones the Traceloop/LangGraph travel-booking supervisor but routes
all chat completions through a LiteLLM proxy (OpenAI-compatible). Point any demo
apps or deepeval runs at LiteLLM and let the proxy manage Cisco CircuIT OAuth.

## Prerequisites

1. Install the shared manual-example dependencies:
   ```bash
   pip install -r instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/requirements.txt
   ```
2. A running LiteLLM proxy. The shim now manages CircuIT tokens internally, so
   LiteLLM only needs to reach the shim (default `http://localhost:5001`) and
   expect a static key from demo clients such as
   `Authorization: Bearer litellm-demo-key`.

## Bringing up LiteLLM

1. Switch to the demo directory:
   ```bash
   cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo
   ```
2. Export your Cisco credentials (fill them in `.env` or export manually) so the
   shim can mint and refresh tokens on demand. Optional overrides:
   `CISCO_TOKEN_URL`, `CIRCUIT_TOKEN_CACHE`.
3. Render the config with your environment values:
   ```bash
   ./scripts/render_config.py litellm-config.example.yaml litellm-config.yaml
   ```
4. Start the local CircuIT shim + LiteLLM stack (docker compose v2):
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
  "http://localhost:${LITELLM_PORT:-4000}/v1/models"
```

> Note: The shim caches CircuIT tokens and refreshes on demand. Production
> deployments should embed similar logic directly in the proxy service instead
> of relying on this lightweight reference shim.

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

## Refreshing the CircuIT token

No manual refresh loop is necessary anymore. The shim stores the CircuIT access
token in `CIRCUIT_TOKEN_CACHE`, refreshes it a few minutes before it expires,
and retries once with a forced refresh whenever CircuIT responds with HTTP 401.
Delete the cache file (or change your client secret) if you need to force a
refresh outside of the shim.

## Basic multi-agent planner

Need something richer than the smoke test but lighter than the full LangGraph
supervisor? Run the simplified planner which orchestrates separate flight and
hotel agents via LiteLLM:

```bash
python instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo/multi_agent_basic.py \\
  "book a trip from SFO to CDG and a boutique hotel near the Louvre"
```

The script streams intermediate agent updates and prints the final itinerary. To
avoid the upstream `parallel_tool_calls` incompatibility, this basic planner
keeps the agents purely conversational (no Python tool execution) but still runs
the full multi-agent routing loop. It reads the same `.env` values
(`LITELLM_BASE_URL`, `LITELLM_API_KEY`, etc.) and falls back to
`TRAVEL_DEMO_PROMPT` when no CLI argument is provided.

## Configuration

1. Copy the example environment file:
   ```bash
   cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/litellm_travel_agent_demo
   cp .env.example .env
   ```
2. Edit `.env` to point to your LiteLLM proxy base URL, API key, and Cisco
   credentials. When using a remote proxy, make sure the networking rules allow
   connections from your workstation/container.

Environment variables:

- `CISCO_CLIENT_ID`, `CISCO_CLIENT_SECRET`, `CISCO_APP_KEY` – Cisco CircuIT creds
  consumed by the shim to mint OAuth tokens and passed downstream in user metadata.
- `CISCO_TOKEN_URL`, `CIRCUIT_TOKEN_CACHE` – optional overrides for the token
  endpoint and cache path used by the shim.
- `CIRCUIT_UPSTREAM_BASE` – the real Cisco CircuIT base URL (defaults to
  `https://chat-ai.cisco.com`).
- `CIRCUIT_API_BASE` – where LiteLLM reaches the shim (defaults to
  `http://circuit-shim:5001/`). Change this if you host the shim elsewhere.
- `CIRCUIT_SHIM_API_KEY` – placeholder API key LiteLLM sends to the shim (not
  forwarded upstream).
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
