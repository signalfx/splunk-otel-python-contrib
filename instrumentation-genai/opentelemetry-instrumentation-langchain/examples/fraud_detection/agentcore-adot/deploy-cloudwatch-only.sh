#!/usr/bin/env bash
# Redeploy the Fraud-detection agent with `DISABLE_ADOT_OBSERVABILITY=true`, per
# https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html
# ("Using other observability platforms"): this unsets the AgentCore control plane's default
# ADOT env-var injection, so `opentelemetry-instrument` has no destination to export spans to
# and none reach CloudWatch GenAI Observability / `aws/spans`. The agent's own stdout/stderr
# still lands in its always-on AgentCore-managed runtime log group regardless -- that path is
# independent of ADOT -- which is the "plain CloudWatch, no GenAI dashboard" demonstration.
#
# Deliberately drops deploy.sh's Galileo OTLP override too, so this agent reports to neither
# Galileo AO nor AWS's GenAI Observability -- only bare CloudWatch Logs.
#
#   ./deploy-cloudwatch-only.sh
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

EXEC_ROLE="${AGENTCORE_EXECUTION_ROLE:-arn:aws:iam::875228160670:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-2e5331ee53}"
SUBNETS="${AGENTCORE_SUBNETS:-subnet-0b37c9ca7669536d0,subnet-0e3580ee0738dbf35}"
SEC_GROUPS="${AGENTCORE_SECURITY_GROUPS:-sg-0c4ea20f07e5794cc}"

echo "AO target: none -- plain CloudWatch Logs only (DISABLE_ADOT_OBSERVABILITY=true)"

agentcore configure \
  --entrypoint main.py \
  --name "$AGENT_NAME" \
  --requirements-file requirements.txt \
  --region "$AWS_REGION" \
  --execution-role "$EXEC_ROLE" \
  --ecr auto \
  --vpc --subnets "$SUBNETS" --security-groups "$SEC_GROUPS" \
  --non-interactive

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
  --env DISABLE_ADOT_OBSERVABILITY="true" \
  --env OTEL_PYTHON_DISABLED_INSTRUMENTATIONS="fastapi,flask,django"

echo
echo "Done. Invoke with:"
echo "  agentcore invoke --agent $AGENT_NAME '{\"claim_facts\":{\"claim_id\":\"CLM-1\"}}'"
