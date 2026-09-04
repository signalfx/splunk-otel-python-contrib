#!/usr/bin/env bash
# Deploy the Fraud-detection agent to Bedrock AgentCore, instrumented with ADOT.
#
#   ./deploy.sh
#
# Unlike ai_underwriting_pipeline/agentcore-splunk/deploy.sh, this does NOT pass
# --disable-otel to `agentcore configure`. The point of this agent is to demonstrate the
# AWS-standardized instrumentation path actually working: `opentelemetry-instrument` wraps
# the entrypoint, aws-opentelemetry-distro auto-instruments botocore etc., and this app adds
# only the community opentelemetry-instrumentation-openai package on top -- see main.py and
# patch_dockerfile.py for why that combination does not hit the semconv-version conflict the
# sibling example's --disable-otel works around.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$HERE/.venv/bin/agentcore" ]] && PATH="$HERE/.venv/bin:$PATH"

AGENT_NAME="fraud_detection"
AWS_REGION="${AWS_REGION:-us-west-2}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

require() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "ERROR: \$$name is required" >&2; exit 1; }
}

require OPENAI_API_KEY
require GALILEO_API_KEY
: "${GALILEO_PROJECT:=ai-underwriting-pipeline}"
: "${GALILEO_LOG_STREAM:=default}"
require GALILEO_OTEL_TRACES_ENDPOINT

# Same AgentCore infra ai_underwriting_pipeline/agentcore-splunk/deploy.sh reuses: the
# execution role, VPC subnets and security group already exist in this account. See that
# script's comments for why subnet-0702145c672de22db is deliberately excluded.
EXEC_ROLE="${AGENTCORE_EXECUTION_ROLE:-arn:aws:iam::875228160670:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-2e5331ee53}"
SUBNETS="${AGENTCORE_SUBNETS:-subnet-0b37c9ca7669536d0,subnet-0e3580ee0738dbf35}"
SEC_GROUPS="${AGENTCORE_SECURITY_GROUPS:-sg-0c4ea20f07e5794cc}"

echo "AO target: $GALILEO_OTEL_TRACES_ENDPOINT (project=$GALILEO_PROJECT logstream=$GALILEO_LOG_STREAM)"

agentcore configure \
  --entrypoint main.py \
  --name "$AGENT_NAME" \
  --requirements-file requirements.txt \
  --region "$AWS_REGION" \
  --execution-role "$EXEC_ROLE" \
  --ecr auto \
  --vpc --subnets "$SUBNETS" --security-groups "$SEC_GROUPS" \
  --non-interactive

# Reorders nothing -- just bumps the distro pin the template hardcodes. Must run between
# configure and launch: configure regenerates the Dockerfile from the template every time.
python3 patch_dockerfile.py "$AGENT_NAME"

runtime_id=$(aws bedrock-agentcore-control list-agent-runtimes \
  --query "agentRuntimes[?agentRuntimeName=='${AGENT_NAME}'].agentRuntimeId | [0]" \
  --output text 2>/dev/null | grep -v '^None$' || true)
echo "  coverage-join runtime id: ${runtime_id:-<none yet, first deploy>}"

agentcore launch \
  --agent "$AGENT_NAME" \
  --auto-update-on-conflict \
  --env AGENTCORE_RUNTIME_ID="$runtime_id" \
  --env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --env OPENAI_MODEL="$MODEL" \
  --env GENAI_LENS_FEATURE_ID="Fraud Detection" \
  --env OTEL_SERVICE_NAME="$AGENT_NAME" \
  --env OTEL_RESOURCE_ATTRIBUTES="deployment.environment=aws-agentcore,service.namespace=genai-lens" \
  --env OTEL_TRACES_EXPORTER="otlp" \
  --env OTEL_METRICS_EXPORTER="none" \
  --env OTEL_LOGS_EXPORTER="none" \
  --env OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
  --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$GALILEO_OTEL_TRACES_ENDPOINT" \
  --env OTEL_EXPORTER_OTLP_TRACES_HEADERS="Galileo-API-Key=${GALILEO_API_KEY},project=${GALILEO_PROJECT},logstream=${GALILEO_LOG_STREAM}" \
  --env OTEL_PYTHON_DISABLED_INSTRUMENTATIONS="fastapi,flask,django"

echo
echo "Done. Invoke with:"
echo "  agentcore invoke --agent $AGENT_NAME '{\"claim_facts\":{\"claim_id\":\"CLM-1\"}}'"
