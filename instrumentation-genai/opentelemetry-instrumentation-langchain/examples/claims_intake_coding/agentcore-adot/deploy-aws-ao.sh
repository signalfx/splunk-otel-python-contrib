#!/usr/bin/env bash
# Redeploy the Claims intake coding agent WITHOUT the Galileo OTLP override deploy.sh sets,
# so it demonstrates AWS's own built-in observability path (CloudWatch GenAI Observability /
# `aws/spans`, or the agent's own unified log group) instead of Splunk AO.
#
# Per https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html,
# once CloudWatch Transaction Search is enabled account-wide, an AgentCore Runtime hosted agent
# gets AWS-native OTel routing by default -- the AgentCore control plane injects
# AGENT_OBSERVABILITY_ENABLED / OTEL_PYTHON_DISTRO=aws_distro / OTEL_PYTHON_CONFIGURATOR=
# aws_configurator into the running container itself, which is why deploy.sh's Dockerfile has
# no such ENV lines baked in. Setting a custom OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (as deploy.sh
# does, pointed at Galileo) overrides that default entirely. This script is deploy.sh with
# those Galileo overrides removed -- the sole difference between the two files.
#
#   ./deploy-aws-ao.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$HERE/.venv/bin/agentcore" ]] && PATH="$HERE/.venv/bin:$PATH"

AGENT_NAME="claims_intake_coding"
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

echo "AO target: AWS CloudWatch GenAI Observability (no custom OTLP endpoint set)"

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
  --env GENAI_LENS_FEATURE_ID="Claims Intake Coding" \
  --env OTEL_SERVICE_NAME="$AGENT_NAME" \
  --env OTEL_RESOURCE_ATTRIBUTES="deployment.environment=aws-agentcore,service.namespace=genai-lens" \
  --env AGENT_OBSERVABILITY_ENABLED="true" \
  --env OTEL_PYTHON_DISTRO="aws_distro" \
  --env OTEL_PYTHON_CONFIGURATOR="aws_configurator" \
  --env OTEL_PYTHON_DISABLED_INSTRUMENTATIONS="fastapi,flask,django"

echo
echo "Done. Invoke with:"
echo "  agentcore invoke --agent $AGENT_NAME '{\"claim_text\":\"...\"}'"
