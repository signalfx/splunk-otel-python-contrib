# Tokenator (Rate Limit Predictor) Example

This example validates the `rate_limit_predictor` emitter end-to-end using the
CrewAI `customer_support.py` app. It records token usage into SQLite and emits
`gen_ai.rate_limit.warning` events.

## Prerequisites

- Python 3.10+
- An LLM provider with working credentials (OAuth2 or API key)
- OpenTelemetry Collector running (for OTLP exporters)

## Install (local dev)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e util/opentelemetry-util-genai
pip install -e util/opentelemetry-util-genai-tokenator
pip install -e util/opentelemetry-util-genai-emitters-splunk
pip install -e instrumentation-genai/opentelemetry-instrumentation-crewai
pip install -e instrumentation-genai/opentelemetry-instrumentation-openai-v2
```

## Environment Variables

Set your provider credentials and GenAI emitter configuration. Use placeholders
for credentials in docs and scripts.

```bash
# LLM credentials (examples; replace with your values)
export LLM_CLIENT_ID=<your-oauth2-client-id>
export LLM_CLIENT_SECRET=<your-oauth2-client-secret>
export LLM_TOKEN_URL=https://<your-idp>/oauth2/token
export LLM_BASE_URL=https://<your-llm-gateway>/openai/deployments
export LLM_APP_KEY=<your-app-key>

# OTel exporters
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_LOGS_EXPORTER=otlp

# Enable content + Tokenator alongside Splunk content events
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk,rate_limit_predictor
export OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS=replace:SplunkConversationEvents,RateLimitPredictor

# Tokenator settings
export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_DB_PATH=~/.opentelemetry_genai_rate_limit.db
export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_WARNING_THRESHOLD=0.0
```

## Run the Example (Zero-Code)

```bash
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples
export OTEL_MANUAL_INSTRUMENTATION=false
opentelemetry-instrument python customer_support.py
```

## Verify SQLite Persistence

```bash
sqlite3 ~/.opentelemetry_genai_rate_limit.db ".tables"
sqlite3 ~/.opentelemetry_genai_rate_limit.db "SELECT COUNT(*) FROM token_usage;"
sqlite3 ~/.opentelemetry_genai_rate_limit.db "SELECT * FROM trace_token_usage ORDER BY last_updated DESC LIMIT 5;"
sqlite3 ~/.opentelemetry_genai_rate_limit.db "SELECT * FROM workflow_patterns ORDER BY last_updated DESC LIMIT 5;"
```

Expected tables:

- `token_usage`
- `trace_token_usage`
- `workflow_patterns`

## Sample Event

```json
{
  "event.name": "gen_ai.rate_limit.warning",
  "gen_ai.provider.name": "openai",
  "gen_ai.request.model": "gpt-4o-mini",
  "rate_limit.type": "tokens_per_minute",
  "rate_limit.current_usage": 551,
  "rate_limit.limit": 200000,
  "rate_limit.utilization_percent": 0.3,
  "rate_limit.will_breach": false,
  "rate_limit.recommendation": "WARNING: Approaching TPM limit. Current: 551 / Limit: 200,000 (0.3%)",
  "trace_id": "d04885941f17d1c2b1d2f72f92ac1df0"
}
```

## Notes

- `splunk` replaces the entire `content_events` category. The override
  `OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS=replace:SplunkConversationEvents,RateLimitPredictor`
  is required to run both Splunk conversation events and Tokenator warnings.
- Set the warning threshold to `0.0` for easy validation.
