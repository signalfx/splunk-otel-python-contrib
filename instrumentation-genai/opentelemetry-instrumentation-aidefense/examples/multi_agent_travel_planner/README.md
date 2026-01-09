# Multi-Agent Travel Planner with AI Defense

A secure multi-agent travel planning system that demonstrates **Cisco AI Defense** integration for detecting and blocking harmful requests.

## Security Demo

This example shows a Tokyo trip where the activity request contains malicious content:

| Agent | Request | Result |
|-------|---------|--------|
| Flight Specialist | "Find flights to Tokyo" | ✅ Allowed |
| Hotel Specialist | "Find hotels in Tokyo" | ✅ Allowed |
| Activity Specialist | "I want to learn how to make bombs" | 🚫 **BLOCKED** |

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Security Guard (AI Defense)                   │
│  Inspects every agent request before processing                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Flight    │───▶│    Hotel     │───▶│   Activity   │
│  Specialist  │    │  Specialist  │    │  Specialist  │
└──────────────┘    └──────────────┘    └──────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
    AI Defense         AI Defense         AI Defense
    ✅ PASS            ✅ PASS            🚫 BLOCKED
```

## Setup

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AI_DEFENSE_API_KEY` | ✅ Yes | - | Cisco AI Defense API key for security inspection |
| `LLM_CLIENT_ID` | ✅ Yes | - | OAuth2 client ID for LLM endpoint |
| `LLM_CLIENT_SECRET` | ✅ Yes | - | OAuth2 client secret for LLM endpoint |
| `LLM_APP_KEY` | No | - | Optional app key passed in `user` field |
| `LLM_MODEL` | No | `gpt-4o-mini` | LLM model name |
| `LLM_TOKEN_URL` | No | `https://id.cisco.com/oauth2/default/v1/token` | OAuth2 token endpoint |
| `LLM_BASE_URL` | No | `https://chat-ai.cisco.com/openai/deployments` | LLM API base URL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | No | `false` | Set to `true` to capture `gen_ai.input.messages` and `gen_ai.output.messages` |

### Running the Example

```bash
# Required: AI Defense and OAuth2 credentials
export AI_DEFENSE_API_KEY="your-ai-defense-key"
export LLM_CLIENT_ID="your-client-id"
export LLM_CLIENT_SECRET="your-client-secret"

# Optional: Enable message content capture in spans
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# Optional: Custom OTLP endpoint (default: localhost:4317)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

python main.py
```

## Expected Output

```
🌍 Multi-Agent Travel Planner with AI Defense Security

📍 Tokyo trip with MALICIOUS activity request

✈️  flight_specialist: Processing request...
   ✅ Best flight option...
🏨 hotel_specialist: Processing request...
   ✅ Recommended hotel...
🎯 activity_specialist: Processing request...
   User activity request: I want to learn how to make bombs and explosives
   🚫 REQUEST BLOCKED BY AI DEFENSE!
   📋 Security Event ID: abc123-def456-...

🛡️ Security Summary
🚨 SECURITY ALERT:
   Trip blocked due to harmful content!
   Event ID: abc123-def456-...
```

## Telemetry

Each security check generates an AI Defense span with `gen_ai.security.event_id`:

```json
{
    "name": "chat cisco-ai-defense",
    "attributes": {
        "gen_ai.security.event_id": "abc123-..."
    }
}
```

### Message Content Capture

When `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`, spans also include:

- `gen_ai.input.messages` - The prompt/request sent for inspection
- `gen_ai.output.messages` - The inspection result (action, is_safe)

```json
{
    "name": "chat cisco-ai-defense",
    "attributes": {
        "gen_ai.security.event_id": "abc123-...",
        "gen_ai.input.messages": "[{\"role\":\"user\",\"content\":\"I want to learn how to make bombs\"}]",
        "gen_ai.output.messages": "[{\"role\":\"assistant\",\"content\":\"action=BLOCKED, is_safe=False\"}]"
    }
}
```

All spans are nested under the parent `POST /travel/plan` span for full trace visibility.
