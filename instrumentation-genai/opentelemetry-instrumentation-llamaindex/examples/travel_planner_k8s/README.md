# LlamaIndex Travel Planner - Kubernetes Deployment

This directory contains Kubernetes deployment files for the LlamaIndex Travel Planner example application with OpenTelemetry instrumentation.

## Components

- **Server (`main_server.py`)**: HTTP server that exposes a `/plan` endpoint for travel planning requests using LlamaIndex ReActAgent
- **Deployment (`deployment.yaml`)**: Kubernetes Deployment and Service configuration
- **CronJob (`cronjob.yaml`)**: Automated load generator that sends periodic travel planning requests

## Architecture

```
┌─────────────┐
│  CronJob    │  (Load Generator)
│  (curl)     │
└──────┬──────┘
       │ HTTP POST /plan
       ▼
┌─────────────────────────────┐
│  Deployment                 │
│  llamaindex-travel-planner  │
│  ┌─────────────────────┐   │
│  │ LlamaIndex Agent    │   │
│  │ + OTEL Instrumentation│  │
│  └─────────────────────┘   │
└──────┬──────────────────────┘
       │ OTLP (gRPC)
       ▼
┌─────────────┐
│ OTEL        │
│ Collector   │
└─────────────┘
```

## Prerequisites

1. Kubernetes cluster with namespace `travel-planner`
2. OpenAI API key stored as secret:
   ```bash
   kubectl create secret generic openai-api-keys \
     --from-literal=openai-api-key=YOUR_API_KEY \
     -n travel-planner
   ```
3. OpenTelemetry Collector running on cluster nodes (DaemonSet)

## Building the Docker Image

```bash
# From this directory
docker build -t shuniche855/llamaindex-travel-planner:0.0.1 .

# Push to registry
docker push shuniche855/llamaindex-travel-planner:0.0.1
```

## Deployment

```bash
# Deploy the server
kubectl apply -f deployment.yaml

# Deploy the load generator CronJob
kubectl apply -f cronjob.yaml
```

## Testing

### Health Check

```bash
kubectl port-forward -n travel-planner svc/llamaindex-travel-planner-service 8080:80
curl http://localhost:8080/health
```

### Manual Request

```bash
curl -X POST http://localhost:8080/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Paris",
    "origin": "New York",
    "budget": 3000,
    "duration": 5,
    "travelers": 2,
    "interests": ["sightseeing", "food", "culture"],
    "departure_date": "2024-06-15"
  }'
```

## Environment Variables

Key environment variables configured in `deployment.yaml`:

- `OTEL_SERVICE_NAME`: Service name for telemetry
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP collector endpoint
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`: Enable message content capture
- `OPENAI_API_KEY`: OpenAI API key (from secret)

## Monitoring

The application generates:

- **Traces**: LlamaIndex agent execution, LLM calls, tool invocations
- **Metrics**: LLM token usage, latency, error rates
- **Logs**: Application logs with trace correlation

View traces in your observability platform (Splunk O11y, Jaeger, etc.)

## Load Generation Schedule

The CronJob runs:

- **Schedule**: Every 30 minutes during business hours
- **Days**: Monday-Friday
- **Time**: 8am-6pm PST (16:00-02:00 UTC)

Adjust the schedule in `cronjob.yaml` as needed.
