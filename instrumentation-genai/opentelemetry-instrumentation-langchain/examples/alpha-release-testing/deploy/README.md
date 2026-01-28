# Alpha Release Testing - Deployment Configurations

Production-ready deployment configurations for Docker and Kubernetes.

---

## 📁 Files

| File | Purpose | Status |
|------|---------|--------|
| `Dockerfile` | Container image for all test apps | ✅ Ready |
| `cronjob-alpha-tests.yaml` | Kubernetes CronJob manifests | ✅ Ready |
| `otel-collector-config.yaml` | OTEL Collector configuration | ✅ Ready |

---

## 🐳 Docker Deployment

### Build Image

From the **repository root**:
```bash
docker build \
  -f instrumentation-genai/opentelemetry-instrumentation-langchain/examples/alpha-release-testing/deploy/Dockerfile \
  -t alpha-test-apps:latest \
  .
```

### Run Individual Apps

#### LangChain Evaluation
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  alpha-test-apps:latest \
  python tests/apps/langchain_evaluation_app.py
```

#### LangGraph Travel Planner (Zero-Code)
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  -e TRAVEL_POISON_PROB=0.75 \
  alpha-test-apps:latest \
  opentelemetry-instrument python tests/apps/langgraph_travel_planner_app.py
```

#### LangGraph Travel Planner (Manual)
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  alpha-test-apps:latest \
  python tests/apps/langgraph_travel_planner_app.py
```

#### Run All Tests
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  alpha-test-apps:latest \
  ./run_tests.sh all
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

1. **Create Secrets**:
```bash
# OpenAI API Key
kubectl create secret generic openai-credentials \
  --from-literal=api-key=$OPENAI_API_KEY

# Splunk Credentials (rc0)
kubectl create secret generic splunk-credentials-rc0 \
  --from-literal=access-token=$SPLUNK_ACCESS_TOKEN \
  --from-literal=hec-token=$SPLUNK_HEC_TOKEN
```

2. **Deploy OTEL Collector** (optional):
```bash
kubectl apply -f otel-collector-config.yaml
```

### Deploy CronJobs

```bash
# Deploy both LangChain and LangGraph CronJobs
kubectl apply -f cronjob-alpha-tests.yaml
```

This creates two CronJobs:
- `alpha-release-tests-langgraph` - Runs every 30 minutes (on the hour and half-hour)
- `alpha-release-tests-langchain` - Runs every 30 minutes (offset by 15 minutes)

### Check Status

```bash
# View CronJobs
kubectl get cronjobs

# View Jobs
kubectl get jobs

# View Pods
kubectl get pods -l app=alpha-release-tests

# View Logs
kubectl logs -l app=alpha-release-tests --tail=100
```

### Manual Trigger

```bash
# Trigger LangGraph test immediately
kubectl create job --from=cronjob/alpha-release-tests-langgraph manual-langgraph-test

# Trigger LangChain test immediately
kubectl create job --from=cronjob/alpha-release-tests-langchain manual-langchain-test
```

---

## 🔧 Configuration

### Environment Variables

All environment variables from `config/.env.*` templates can be overridden in the Kubernetes manifests.

**Key Variables**:
- `OPENAI_API_KEY` - OpenAI authentication
- `SPLUNK_REALM` - Splunk realm (lab0, rc0, us1)
- `SPLUNK_ACCESS_TOKEN` - Splunk access token
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OTEL Collector endpoint
- `OTEL_SERVICE_NAME` - Service identifier
- `TRAVEL_POISON_PROB` - LangGraph poisoning probability (0.0-1.0)

### Resource Limits

**LangGraph** (more resource-intensive):
- Requests: 512Mi RAM, 500m CPU
- Limits: 1Gi RAM, 1000m CPU

**LangChain** (lighter):
- Requests: 256Mi RAM, 200m CPU
- Limits: 512Mi RAM, 500m CPU

---

## 📊 OTEL Collector Configuration

The `otel-collector-config.yaml` provides:

### Receivers
- OTLP gRPC (port 4317)
- OTLP HTTP (port 4318)

### Exporters
- Splunk OTLP HTTP with authentication
- Console logging (for debugging)

### Processors
- Batch processing (512 batch size, 5s timeout)
- Memory limiter (512 MiB default)

### Usage

```bash
# Deploy as Kubernetes ConfigMap
kubectl create configmap otel-collector-config \
  --from-file=config.yaml=otel-collector-config.yaml

# Set environment variables for Splunk
export SPLUNK_INGEST_URL=https://ingest.rc0.signalfx.com
export SPLUNK_ACCESS_TOKEN=your-token-here
export SPLUNK_MEMORY_TOTAL_MIB=512

# Deploy OTEL Collector with this config
# (requires OTEL Collector Kubernetes deployment manifest)
```

---

## 🧪 Testing Deployment

### Test Docker Build
```bash
# Build
docker build -f deploy/Dockerfile -t alpha-test-apps:latest .

# Test run
docker run --rm alpha-test-apps:latest echo "✅ Build successful"
```

### Test Kubernetes Deployment
```bash
# Dry run
kubectl apply -f deploy/cronjob-alpha-tests.yaml --dry-run=client

# Deploy
kubectl apply -f deploy/cronjob-alpha-tests.yaml

# Verify
kubectl get cronjobs
kubectl describe cronjob alpha-release-tests-langgraph
```

---

## 🔍 Troubleshooting

### Docker Issues

**Build fails**:
```bash
# Check you're in repository root
pwd  # Should end with /splunk-otel-python-contrib

# Verify paths exist
ls instrumentation-genai/
ls util/
```

**Container exits immediately**:
```bash
# Check logs
docker logs <container-id>

# Run interactively
docker run -it --entrypoint /bin/bash alpha-test-apps:latest
```

### Kubernetes Issues

**CronJob not running**:
```bash
# Check CronJob status
kubectl get cronjobs
kubectl describe cronjob alpha-release-tests-langgraph

# Check for recent jobs
kubectl get jobs --sort-by=.metadata.creationTimestamp
```

**Pods failing**:
```bash
# Check pod logs
kubectl logs -l app=alpha-release-tests --tail=100

# Check pod events
kubectl describe pod <pod-name>

# Check secrets exist
kubectl get secrets | grep -E "openai|splunk"
```

**No telemetry in Splunk**:
```bash
# Verify OTEL Collector is running
kubectl get pods -l app=otel-collector

# Check collector logs
kubectl logs -l app=otel-collector

# Verify environment variables
kubectl describe cronjob alpha-release-tests-langgraph | grep -A 20 "Environment:"
```

---

## 📝 Customization

### Change Schedule

Edit `cronjob-alpha-tests.yaml`:
```yaml
spec:
  schedule: "*/15 * * * *"  # Every 15 minutes
  schedule: "0 */2 * * *"   # Every 2 hours
  schedule: "0 9 * * *"     # Daily at 9 AM
```

### Change Realm

Edit environment variables in `cronjob-alpha-tests.yaml`:
```yaml
- name: SPLUNK_REALM
  value: "us1"  # or "lab0"
- name: OTEL_RESOURCE_ATTRIBUTES
  value: "deployment.environment=alpha-us1,realm=us1"
```

### Add More Apps

Add new container in `cronjob-alpha-tests.yaml`:
```yaml
command: ["./run_tests.sh"]
args: ["traceloop"]  # or "direct_azure"
```

---

## 🚀 Production Checklist

Before deploying to production:

- [ ] Secrets created and verified
- [ ] OTEL Collector deployed and configured
- [ ] Resource limits appropriate for cluster
- [ ] Schedule configured correctly
- [ ] Monitoring/alerting set up
- [ ] Logs aggregation configured
- [ ] Image pushed to registry (if using private registry)
- [ ] Network policies configured (if required)
- [ ] RBAC permissions set (if required)

---

**Status**: ✅ Production-Ready  
**Last Updated**: November 12, 2025  
**Migrated From**: qse-evaluation-harness/deploy

