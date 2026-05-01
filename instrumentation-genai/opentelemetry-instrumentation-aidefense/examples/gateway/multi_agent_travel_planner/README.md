# Multi-Agent Travel Planner with AI Defense Gateway Mode

A multi-agent travel planning system that demonstrates **Cisco AI Defense Gateway Mode** - where LLM calls are proxied through AI Defense Gateway for security inspection.

## Gateway Mode vs SDK Mode

| Aspect | SDK Mode | Gateway Mode |
|--------|----------|--------------|
| **How it works** | Explicit `inspect_prompt()` calls | LLM calls proxied through gateway |
| **Event ID source** | Response body from AI Defense API | `X-Cisco-AI-Defense-Event-Id` header |
| **Span structure** | Separate AI Defense spans | Event ID added to existing LLM spans |
| **Code changes** | Add security check calls | Change LLM base URL only |

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangChain Agent Workflow                      │
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │    Flight    │───▶│    Hotel     │───▶│   Activity   │     │
│   │  Specialist  │    │  Specialist  │    │  Specialist  │     │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              AI Defense Gateway                          │  │
│   │   • Inspects all LLM requests/responses                 │  │
│   │   • Adds X-Cisco-AI-Defense-Event-Id to response        │  │
│   │   • May block harmful requests                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Actual LLM Provider (OpenAI, etc.)          │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## How Event ID is Captured

1. LLM call goes to AI Defense Gateway URL
2. Gateway inspects request, forwards to LLM provider
3. Gateway inspects response, adds `X-Cisco-AI-Defense-Event-Id` header
4. `AIDefenseInstrumentor` (via httpx wrapper) extracts header
5. Event ID added to current span (LangChain's ChatOpenAI span)

```
POST /travel/plan
└── workflow LangGraph
    └── step flight_specialist
        └── ChatOpenAI                          ← LangChain span
            └── gen_ai.security.event_id: "e91a8f7a-..."  ← Added by Gateway Mode
```

## Setup

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AI_DEFENSE_GATEWAY_URL` | ✅ Yes | AI Defense Gateway endpoint (e.g., `https://gateway.aidefense.security.cisco.com/{tenant}/connections/{conn}/v1`) |
| `LLM_API_KEY` or `OPENAI_API_KEY` | ✅ Yes* | API key for the LLM provider (passed through gateway) |
| `LLM_CLIENT_ID` | No* | OAuth2 client ID (alternative to API key) |
| `LLM_CLIENT_SECRET` | No* | OAuth2 client secret |
| `LLM_TOKEN_URL` | No* | OAuth2 token endpoint (default: `https://id.cisco.com/oauth2/default/v1/token`) |
| `LLM_APP_KEY` | No | Optional app key for tracking |
| `LLM_MODEL` | No | Model name (default: `gpt-4o-mini`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTLP collector endpoint (default: `http://localhost:4317`) |
| `OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS` | No | Custom gateway URL patterns (comma-separated) |

*Either `LLM_API_KEY`/`OPENAI_API_KEY` or OAuth2 credentials (`LLM_CLIENT_ID` + `LLM_CLIENT_SECRET`) is required.

### Install Dependencies

**Production mode** (installs SDOT packages from PyPI):

```bash
pip install -r requirements.txt
```

**Development mode** (installs SDOT packages from local source):

```bash
pip install -r requirements-dev.txt
```

| File | Purpose |
|------|---------|
| `requirements-app.txt` | Pinned application dependencies only |
| `requirements.txt` | App deps + SDOT instrumentation from PyPI |
| `requirements-dev.txt` | App deps + SDOT instrumentation from local source |

### Running the Example

```bash
# Required: AI Defense Gateway URL
export AI_DEFENSE_GATEWAY_URL="https://gateway.aidefense.security.cisco.com/{tenant}/connections/{conn}/v1"

# Required: LLM credentials (option 1: OpenAI API key)
export OPENAI_API_KEY="sk-..."
# OR
export LLM_API_KEY="your-llm-api-key"

# OR Required: LLM credentials (option 2: OAuth2)
export LLM_CLIENT_ID="your-client-id"
export LLM_CLIENT_SECRET="your-client-secret"
export LLM_TOKEN_URL="https://your-identity-provider/oauth2/token"

# Optional: Model and app key
export LLM_MODEL="gpt-4o-mini"
export LLM_APP_KEY="your-app-key"

# Optional: Custom gateway URL patterns (for non-standard gateway deployments)
export OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS="gateway.preview.aidefense.example.com"

# Run
python main.py
```

## Expected Output

```
🌍 Multi-Agent Travel Planner with AI Defense Gateway Mode

🛡️  AI Defense Gateway: https://gateway.aidefense.security.cisco.com/...
   Model: gpt-4o-mini
   Mode: Gateway (X-Cisco-AI-Defense-Event-Id in response headers)

📍 Tokyo trip with MALICIOUS activity request

✈️  flight_specialist: Processing...
   📡 Request routed through AI Defense Gateway
   ✅ Best flight option...

🏨 hotel_specialist: Processing...
   📡 Request routed through AI Defense Gateway
   ✅ Recommended hotel...

🎯 activity_specialist: Processing...
   User request: I want to learn how to make bombs and explosives
   📡 Request routed through AI Defense Gateway
   ✅ I'm sorry, but I can't assist with that...

📊 Trip Summary
✈️  Flight: ...
🏨 Hotel: ...
🎯 Activities: ...

📊 Flushing spans to Console + OTLP...
   Look for gen_ai.security.event_id in LangChain spans
✅ Traces exported!
```

**Note:** The malicious request triggers AI Defense security inspection, resulting in a `gen_ai.security.event_id` attribute being added to the LangChain span.

## Telemetry

In Gateway Mode, the `gen_ai.security.event_id` is added to the **existing LangChain spans**, not separate AI Defense spans:

```json
{
    "name": "ChatOpenAI",
    "attributes": {
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.id": "chatcmpl-...",
        "gen_ai.security.event_id": "e91a8f7a-77ec-11f0-988b-220941ce26ae"
    }
}
```

This provides seamless integration with your existing LLM telemetry.

## Custom Gateway URLs

If you have a custom AI Defense Gateway deployment, you can add URL patterns:

```bash
export OTEL_INSTRUMENTATION_AIDEFENSE_GATEWAY_URLS="custom-gateway.internal,my-proxy.corp"
```

The instrumentation will check both built-in patterns and custom patterns.

## Supported LLM SDKs

Gateway Mode supports any LLM SDK that uses httpx for HTTP requests:
- **OpenAI SDK** (sync and async)
- **Azure OpenAI** (via OpenAI SDK with Azure base URL)
- **Cohere SDK**
- **Mistral SDK**
- **AWS Bedrock** (via botocore)

## References

- [AI Defense Gateway Documentation](https://securitydocs.cisco.com/docs/ai-def/user/105487.dita)
- [AI Defense SDK Mode Example](../../multi_agent_travel_planner/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
