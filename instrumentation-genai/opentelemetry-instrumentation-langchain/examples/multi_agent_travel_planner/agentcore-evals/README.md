# LangChain Travel Planner on Amazon Bedrock AgentCore

This example demonstrates deploying a LangChain multi-agent travel planner to **Amazon Bedrock AgentCore** with OpenTelemetry instrumentation sending traces and metrics to **Splunk Observability Cloud**.

## What is Amazon Bedrock AgentCore?

[Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) is a managed runtime service for hosting and scaling AI agents on AWS. It's **framework and model agnostic** — you can deploy agents built with LangChain, CrewAI, Strands, or custom frameworks.

### AgentCore vs Traditional Docker/K8s Deployment

| Aspect | AgentCore | Docker + Kubernetes |
|--------|-----------|---------------------|
| **Packaging** | Direct code deploy (no Dockerfile needed) | Requires Dockerfile, image build, ECR push |
| **Scaling** | Fully managed auto-scaling | Manual HPA/VPA configuration |
| **Infrastructure** | Zero infrastructure management | Manage EKS cluster, nodes, networking |
| **Cold starts** | Optimized for serverless workloads | Depends on pod scheduling |
| **Deployment** | `agentcore launch` (single command) | kubectl apply, Helm charts, CI/CD pipelines |
| **Cost model** | Pay per invocation | Pay for running pods/nodes |
| **Observability** | Built-in ADOT integration | Manual OTel collector setup |

**When to use AgentCore:**
- Rapid prototyping and deployment
- Variable/bursty workloads
- Teams without K8s expertise

**When to use Docker/K8s:**
- Existing K8s infrastructure
- Fine-grained control over resources
- Multi-tenant deployments
- Complex networking requirements

---

## Prerequisites

```bash
# Install AWS CLI and AgentCore CLI
pip install awscli bedrock-agentcore bedrock-agentcore-starter-toolkit

# Configure AWS credentials
aws configure

# Verify AgentCore access
agentcore --help
```

---

## Code Changes: Flask → AgentCore

This section documents the key code changes required when adapting a Flask application to run on AgentCore. Compare `main.py` (AgentCore) with `../client_server_version/main.py` (Flask).

### 1. Import `BedrockAgentCoreApp` Instead of Flask

```python
# ❌ Flask version
from flask import Flask, request, jsonify
app = Flask(__name__)

# ✅ AgentCore version
from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()
```

### 2. Replace `@app.route` with `@app.entrypoint`

AgentCore uses a single entrypoint decorator instead of HTTP route decorators:

```python
# ❌ Flask version
@app.route("/travel/plan", methods=["POST"])
def plan():
    data = request.get_json()
    # ... process request ...
    return jsonify(result), 200

# ✅ AgentCore version
@app.entrypoint
def invoke(payload: dict) -> dict:
    # payload is already parsed JSON (no request.get_json() needed)
    # ... process request ...
    return {"status": "success", **result}  # Return dict directly (no jsonify)
```

### 3. Payload Handling

| Flask | AgentCore |
|-------|-----------|
| `request.get_json()` | `payload` parameter (already a dict) |
| `jsonify(result)` | Return `dict` directly |
| `return result, 200` | Return `dict` (status code managed by AgentCore) |

### 4. Application Entry Point

```python
# ❌ Flask version
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

# ✅ AgentCore version
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(port=port)  # AgentCore handles host binding
```

### 5. Complete Entrypoint Example

```python
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict) -> dict:
    """
    AgentCore entrypoint - receives JSON payload, returns JSON response.
    
    Expected payload:
    {
        "origin": "Seattle",
        "destination": "Paris",
        "travellers": 2
    }
    """
    origin = payload.get("origin", "Seattle")
    destination = payload.get("destination", "Paris")
    
    try:
        result = process_request(origin, destination)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    app.run(port=8080)
```

---

## Quick Start

> **Note:** All commands should be run from the `agentcore/` directory containing `main.py`:
> ```bash
> cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/agentcore
> ```

### 1. Local Testing

Test the application locally before deploying to AWS:

```bash
# Navigate to the agentcore directory (if not already there)
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/agentcore

# Set environment variables
export CISCO_CLIENT_ID=your-client-id
export CISCO_CLIENT_SECRET=your-client-secret
export CISCO_APP_KEY=your-app-key
export OTEL_CONSOLE_OUTPUT=true  # Enable console output for debugging

# Run locally with AgentCore local server
agentcore run --local

# In another terminal, test the endpoint
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "San Francisco",
    "destination": "Tokyo",
    "user_request": "Plan a week-long trip with boutique hotels",
    "travellers": 2
  }'
```

### 2. Deploy to AWS AgentCore

```bash
# Configure the agent (creates .bedrock_agentcore.yaml)
agentcore configure -e main.py

# Launch to AWS with environment variables
agentcore launch \
  --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://ingest.us1.signalfx.com/v2/trace/otlp \
  --env OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://ingest.us1.signalfx.com/v2/datapoint/otlp \
  --env OTEL_EXPORTER_OTLP_HEADERS="X-SF-Token=YOUR_SPLUNK_TOKEN" \
  --env OTEL_SERVICE_NAME=travel-planner-agentcore \
  --env OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
  --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
  --env DISABLE_ADOT_OBSERVABILITY=true \
  --env CISCO_CLIENT_ID=your-client-id \
  --env CISCO_CLIENT_SECRET=your-client-secret \
  --env CISCO_APP_KEY=your-app-key
```

### 3. Invoke the Deployed Agent

```bash
# Via AgentCore CLI
agentcore invoke '{"origin": "New York", "destination": "London", "travellers": 3}'
```

**Example Response:**

```
╭─────────────────────────────────────────────────────────────────────── travel_planner ───────────────────────────────────────────────────────────────────────╮
│ Session: c0aba755-a7e4-406a-913d-14dc4c6898b8                                                                                                                │
│ Request ID: 89b7a8f8-571e-4320-a6fd-850c8e0b9000                                                                                                             │
│ ARN: arn:aws:bedrock-agentcore:us-east-2:875228160670:runtime/travel_planner-jY98J0ESeL                                                                      │
│ Logs: aws logs tail /aws/bedrock-agentcore/runtimes/travel_planner-jY98J0ESeL-DEFAULT --log-stream-name-prefix "2025/12/11/[runtime-logs" --follow           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Response:
{
  "status": "success",
  "session_id": "8852f37d-55d0-48c3-9bd7-c5ca01a809d2",
  "origin": "New York",
  "destination": "London",
  "departure": "2026-01-10",
  "return_date": "2026-01-17",
  "travellers": 3,
  "flight_summary": "SkyLine non-stop service, $727 return in Premium Economy",
  "hotel_summary": "The Atlas near historic centre, $293/night with breakfast",
  "activities_summary": "Tower of London, London Eye, British Museum, West End show...",
  "final_itinerary": "### Week-Long Itinerary: New York to London...",
  "agent_steps": [
    {"agent": "coordinator", "status": "completed"},
    {"agent": "flight_specialist", "status": "completed"},
    {"agent": "hotel_specialist", "status": "completed"},
    {"agent": "activity_specialist", "status": "completed"},
    {"agent": "plan_synthesizer", "status": "completed"}
  ]
}
```

**View logs during invocation:**

```bash
# Follow logs in real-time
aws logs tail /aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT \
  --log-stream-name-prefix "2025/12/11/[runtime-logs" --follow

# View last hour of logs
aws logs tail /aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT \
  --log-stream-name-prefix "2025/12/11/[runtime-logs" --since 1h
```

**Via AWS CLI:**

```bash
aws bedrock-agentcore-runtime invoke-agent-runtime \
  --agent-runtime-id <your-runtime-id> \
  --payload '{"origin": "Seattle", "destination": "Paris", "travellers": 2}'
```

---

## Local vs Cloud Deployment

| Flag | Description | Use Case |
|------|-------------|----------|
| `agentcore run --local` | Runs a local HTTP server on port 8080 | Development, debugging, testing |
| `agentcore launch` | Deploys to AWS AgentCore Runtime | Production, staging |

### Local Mode Benefits
- Fast iteration cycles
- Console output for debugging
- No AWS costs during development
- Works offline (except for LLM calls)

### Cloud Mode Benefits
- Managed scaling and availability
- AWS IAM integration
- CloudWatch logging
- Production-ready infrastructure

---

## Sending Telemetry to Splunk Observability Cloud

We evaluated three approaches for exporting OpenTelemetry data to Splunk:

### Approach 1: Direct OTLP Export (Recommended for AgentCore) ✅

Export directly from the application to Splunk's OTLP endpoint.

#### gRPC Configuration (Traces Only)

If you only need traces (no metrics), gRPC can work with a single endpoint:

```python
# gRPC - only works for traces with single endpoint
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Environment: OTEL_EXPORTER_OTLP_ENDPOINT=https://ingest.us1.signalfx.com
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
```

#### HTTP Exporter Configuration (Recommended)

The HTTP exporters automatically read from standard OpenTelemetry environment variables:

```python
# Use HTTP exporters for Splunk (supports custom paths)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

# Traces - reads from OTEL_EXPORTER_OTLP_TRACES_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

# Metrics - reads from OTEL_EXPORTER_OTLP_METRICS_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(),
    export_interval_millis=30000  # Export every 30s
)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
```

#### Environment Variables (HTTP)

```bash
# Separate endpoints with Splunk's custom paths
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://ingest.us1.signalfx.com/v2/trace/otlp
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://ingest.us1.signalfx.com/v2/datapoint/otlp

# Auth header (Splunk access token)
OTEL_EXPORTER_OTLP_HEADERS=X-SF-Token=YOUR_SPLUNK_ACCESS_TOKEN

# Service name
OTEL_SERVICE_NAME=your-service-name

# Enable GenAI metrics (required!)
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric

# Capture input/output message content in spans (optional, may contain sensitive data)
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

**Pros:**
- No additional infrastructure
- Works seamlessly with AgentCore
- Simple configuration via env vars

**Cons:**
- No local data processing/filtering
- Direct egress to Splunk from each instance
- Requires HTTP exporters for metrics (gRPC doesn't support custom paths)

### Approach 2: Splunk OTel Collector Gateway on EKS

Deploy the Splunk Distribution of OpenTelemetry Collector on EKS in the same VPC as AgentCore. This provides centralized telemetry processing, filtering, and forwarding to Splunk Observability Cloud.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS VPC                                         │
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────────────────────────────┐  │
│  │   AgentCore      │         │        EKS Cluster                       │  │
│  │   (Fargate)      │         │   (o11y-inframon-ai-otel-collector)      │  │
│  │                  │         │                                          │  │
│  │  ┌────────────┐  │  OTLP   │  ┌─────────────────────────────────────┐ │  │
│  │  │ LangChain/ │  │ gRPC    │  │   Splunk OTel Collector             │ │  │
│  │  │ CrewAI App │──┼────────►│  │   (splunk-monitoring namespace)     │ │  │
│  │  └────────────┘  │ :4317   │  │                                     │ │  │
│  │                  │         │  │   - Receives OTLP traces/metrics    │ │  │
│  └──────────────────┘         │  │   - Processes & enriches data       │ │  │
│                               │  │   - Forwards to Splunk O11y Cloud   │ │  │
│                               │  └──────────────┬──────────────────────┘ │  │
│                               │                 │                        │  │
│                               │    Internal NLB │                        │  │
│                               │    (port 4317)  │                        │  │
│                               └─────────────────┼────────────────────────┘  │
│                                                 │                            │
└─────────────────────────────────────────────────┼────────────────────────────┘
                                                  │ HTTPS
                                                  ▼
                                    ┌──────────────────────────┐
                                    │  Splunk Observability    │
                                    │  Cloud (us1 realm)       │
                                    │                          │
                                    │  - Traces (APM)          │
                                    │  - Metrics (IM)          │
                                    │  - K8s cluster metrics   │
                                    └──────────────────────────┘
```

#### Prerequisites

- EKS cluster in the same VPC as AgentCore
- `kubectl` configured for your cluster
- `eksctl` and `helm` installed
- Splunk Observability Cloud access token

#### Step 1: Create EKS Node Group

```bash
aws eks create-nodegroup \
  --cluster-name o11y-inframon-ai-otel-collector \
  --nodegroup-name primary-nodes \
  --subnets subnet-xxx subnet-yyy subnet-zzz \
  --node-role arn:aws:iam::ACCOUNT_ID:role/NodeInstanceRole \
  --ami-type AL2023_x86_64_STANDARD \
  --capacity-type ON_DEMAND \
  --instance-types t3.medium \
  --scaling-config minSize=1,maxSize=3,desiredSize=2 \
  --region us-west-2
```

#### Step 2: Create Kubernetes Secret

```bash
kubectl create namespace splunk-monitoring

kubectl create secret generic splunk-otel-collector \
  --from-literal=splunk_observability_access_token=YOUR_TOKEN \
  -n splunk-monitoring
```

#### Step 3: Install AWS Load Balancer Controller

```bash
# Associate OIDC provider
eksctl utils associate-iam-oidc-provider \
  --region us-west-2 \
  --cluster o11y-inframon-ai-otel-collector \
  --approve

# Create IAM service account
eksctl create iamserviceaccount \
  --cluster=o11y-inframon-ai-otel-collector \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn=arn:aws:iam::ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve \
  --region us-west-2

# Get VPC ID
VPC_ID=$(aws eks describe-cluster \
  --name o11y-inframon-ai-otel-collector \
  --region us-west-2 \
  --query 'cluster.resourcesVpcConfig.vpcId' \
  --output text)

# Install controller via Helm
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=o11y-inframon-ai-otel-collector \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller \
  --set vpcId=$VPC_ID \
  --set region=us-west-2
```

#### Step 4: Configure Splunk OTel Collector (EKS Add-on)

Apply this YAML configuration in the EKS Add-on console:

```yaml
splunkObservability:
  realm: us1
  metricsEnabled: true
  tracesEnabled: true

clusterName: o11y-inframon-ai-otel-collector
cloudProvider: aws
distribution: eks
environment: production

secret:
  create: false
  name: splunk-otel-collector
  validateSecret: false

gateway:
  enabled: true
  service:
    type: LoadBalancer
    annotations:
      service.beta.kubernetes.io/aws-load-balancer-type: external
      service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: ip
      service.beta.kubernetes.io/aws-load-balancer-scheme: internal
```

#### Step 5: Tag Subnets for NLB Discovery

```bash
# Add cluster tag (required for internal NLB)
aws ec2 create-tags \
  --resources subnet-xxx subnet-yyy subnet-zzz \
  --tags Key=kubernetes.io/cluster/o11y-inframon-ai-otel-collector,Value=shared \
  --region us-west-2
```

#### Step 6: Verify Deployment

```bash
# Check collector pods
kubectl get pods -n splunk-monitoring
```

**Expected Output:**
```
NAME                                                          READY   STATUS    RESTARTS   AGE
splunk-otel-collector-k8s-cluster-receiver-7fb7bcd5c6-4s7sb   1/1     Running   0          47m
```

```bash
# Check LoadBalancer service
kubectl get svc -n splunk-monitoring
```

**Expected Output:**
```
NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP                                                                     PORT(S)
splunk-otel-collector   LoadBalancer   172.20.167.174   k8s-splunkmo-splunkot-xxxxx.elb.us-west-2.amazonaws.com   4317:30913/TCP,4318:31151/TCP...
```

#### Step 7: Configure AgentCore

```bash
# Get NLB endpoint
NLB_DNS=$(kubectl get svc splunk-otel-collector -n splunk-monitoring \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Launch AgentCore with collector endpoint
agentcore launch \
  --env OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
  --env OTEL_EXPORTER_OTLP_ENDPOINT=http://${NLB_DNS}:4317 \
  --env OTEL_SERVICE_NAME=travel-planner-agentcore \
  --env OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
  --env OTEL_LOGS_EXPORTER=none \
  --env CISCO_CLIENT_ID=your-client-id \
  --env CISCO_CLIENT_SECRET=your-client-secret \
  --env CISCO_APP_KEY=your-app-key
```

#### Troubleshooting

**LoadBalancer stuck at `<pending>`:**

1. Check AWS LB Controller is running:
   ```bash
   kubectl get pods -n kube-system | grep aws-load-balancer
   ```

2. Check service events:
   ```bash
   kubectl describe svc splunk-otel-collector -n splunk-monitoring | grep -A 10 Events
   ```

3. **Subnet tag error** - If you see `"3 tagged for other cluster"`:
   ```bash
   aws ec2 create-tags \
     --resources subnet-xxx \
     --tags Key=kubernetes.io/cluster/YOUR-CLUSTER-NAME,Value=shared \
     --region us-west-2
   ```

4. **Fargate IMDS error** - If controller fails with metadata error:
   ```bash
   helm upgrade aws-load-balancer-controller eks/aws-load-balancer-controller \
     -n kube-system \
     --set vpcId=$VPC_ID \
     --set region=us-west-2
   ```

**Pros:**
- Central data processing, filtering, and batching
- Collects Kubernetes cluster metrics and logs
- Multiple export destinations supported
- Better retry logic and buffering

**Cons:**
- Additional infrastructure to manage (EKS cluster)
- Requires AWS Load Balancer Controller setup
- More complex initial configuration

### Approach 3: AWS ADOT (AgentCore Default)

Use AgentCore's built-in AWS Distro for OpenTelemetry.

```bash
# Disable to use custom exporters
DISABLE_ADOT_OBSERVABILITY=true
```

> ⚠️ **Important**: 
> - **Use HTTP exporters** for both traces and metrics to Splunk. gRPC cannot specify Splunk's custom URL paths.
> - Splunk does **NOT** support OTLP logs. You'll see `StatusCode.UNIMPLEMENTED` errors. Remove log exporters when targeting Splunk.
> - Set `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric` to enable GenAI metrics.

---

## Known Issues & Workarounds

### IAM Role Trust Policy Issue

During initial deployment, you may encounter:

```
❌ Launch failed: Role validation failed for 'arn:aws:iam::ACCOUNT:role/ROLE'.
Please verify that the role exists and its trust policy allows assumption by this service
```

**Root Cause:** The IAM role's trust policy doesn't allow `bedrock-agentcore.amazonaws.com` to assume it.

**Workaround:**

1. **Option A: Let AgentCore auto-create the role**
   ```yaml
   # In .bedrock_agentcore.yaml
   aws:
     execution_role: null
     execution_role_auto_create: true
   ```

2. **Option B: Manually update the trust policy via AWS Console**
   
   Go to IAM → Roles → Your Role → Trust relationships → Edit:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "Service": "bedrock-agentcore.amazonaws.com"
         },
         "Action": "sts:AssumeRole"
       }
     ]
   }
   ```

3. **Option C: Attach required policies manually**
   
   If the auto-created role has policy attachment issues, manually attach:
   - `AmazonS3FullAccess` (or scoped S3 permissions)
   - `CloudWatchLogsFullAccess`
   - `AmazonBedrockFullAccess` (if using Bedrock models)

### DeepEval Permission Error

```
[Errno 13] Permission denied: '.deepeval'
```

**Fix:** Disable evaluators or set a writable directory:
```bash
--env OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=""
# or
--env DEEPEVAL_RESULTS_FOLDER=/tmp/.deepeval
```

---

## Managing AgentCore Runtimes

```bash
# Check status of deployed agents
agentcore status

# View logs
aws logs tail /aws/bedrock-agentcore/runtimes/<agent-name> --follow

# Stop/delete the runtime
agentcore stop

# List all runtimes via AWS CLI
aws bedrock-agentcore-control list-agent-runtimes --region us-east-2

# Delete specific runtime
aws bedrock-agentcore-control delete-agent-runtime \
  --agent-runtime-id <runtime-id> \
  --region us-east-2
```

---

## Project Structure

```
agentcore/
├── main.py              # LangChain travel planner with AgentCore entrypoint
├── requirements.txt     # Python dependencies
├── util/
│   ├── __init__.py
│   └── cisco_token_manager.py  # OAuth2 token management for Cisco LLM
└── README.md            # This file
```

---

## References

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html)
- [AgentCore Samples Repository](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
- [Splunk OTLP Ingest - General](https://help.splunk.com/en/splunk-observability-cloud/manage-data/other-data-ingestion-methods/other-data-ingestion-methods)
- [Splunk OTLP Metrics Endpoint API](https://dev.splunk.com/observability/reference/api/ingest_data/latest#endpoint-send-otlp-metrics)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)

