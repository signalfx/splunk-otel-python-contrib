#!/usr/bin/env bash
# Deploy the AI underwriting pipeline to Bedrock AgentCore, in either telemetry posture.
#
#   ./deploy.sh ao           OTLP -> Agent Observability (properly instrumented)
#   ./deploy.sh spans-logs   spans -> stdout -> CloudWatch Logs (recover-from-logs input)
#   ./deploy.sh both
#
# Two separate AgentCore runtimes so both postures can be observed side by side, and so the
# reconciliation has one workload that reports to AO and one that visibly does not.
set -euo pipefail

VARIANT="${1:-}"
[[ -n "$VARIANT" ]] || { echo "usage: $0 {ao|spans-logs|both}" >&2; exit 2; }

AWS_REGION="${AWS_REGION:-us-west-2}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
NOISE="${UNDERWRITING_NOISE_RATE:-0.35}"

require() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "ERROR: \$$name is required for this variant" >&2; exit 1; }
}

# Infrastructure is reused from the account's existing AgentCore setup rather than
# auto-created: the execution role, VPC subnets and security group already exist and the
# runtime must sit in the same VPC to reach the model provider.
EXEC_ROLE="${AGENTCORE_EXECUTION_ROLE:-arn:aws:iam::875228160670:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-2e5331ee53}"
# subnet-0702145c672de22db is deliberately omitted: it sits in us-west-2d (usw2-az4), and
# AgentCore only supports usw2-az1/az2/az3 in this region. Including it fails endpoint
# creation AFTER the image has built and the runtime has been created, so the error
# arrives late and looks like a networking fault rather than an unsupported AZ.
# Remaining: subnet-0b37c9ca7669536d0 (us-west-2c/az3), subnet-0e3580ee0738dbf35 (us-west-2a/az1).
SUBNETS="${AGENTCORE_SUBNETS:-subnet-0b37c9ca7669536d0,subnet-0e3580ee0738dbf35}"
SEC_GROUPS="${AGENTCORE_SECURITY_GROUPS:-sg-0c4ea20f07e5794cc}"

deploy_one() {
  local agent_name="$1" export_mode="$2"; shift 2
  echo "=============================================================="
  echo "Deploying $agent_name  (UNDERWRITING_EXPORT_MODE=$export_mode)"
  echo "=============================================================="

  agentcore configure \
    --entrypoint main.py \
    --name "$agent_name" \
    --requirements-file requirements.txt \
    --region "$AWS_REGION" \
    --execution-role "$EXEC_ROLE" \
    --ecr auto \
    --vpc --subnets "$SUBNETS" --security-groups "$SEC_GROUPS" \
    --disable-otel \
    --non-interactive

  # --disable-otel above is the real fix: aws-opentelemetry-distro pins
  # opentelemetry-sdk==1.33.1, which drags opentelemetry-semantic-conventions down to
  # 0.54b1 -- and GEN_AI_CONVERSATION_ID does not exist until a later version, so the
  # Splunk GenAI instrumentation dies at import. This app configures its own
  # TracerProvider and LangchainInstrumentor, so it needs neither the distro nor the
  # `opentelemetry-instrument` wrapper the template adds.
  #
  # patch_dockerfile.py still runs as a safety net: it adds a build-time import guard so a
  # future template change that reintroduces the distro fails the BUILD rather than
  # shipping an image that cannot start. It must run between configure and launch because
  # configure regenerates the Dockerfile and discards manual edits.
  python3 patch_dockerfile.py "$agent_name" || true

  # --auto-update-on-conflict: launch refuses to touch an existing agent otherwise, so a
  # second deploy of the same name fails with ConflictException rather than updating it.
  agentcore launch \
    --agent "$agent_name" \
    --auto-update-on-conflict \
    --env UNDERWRITING_EXPORT_MODE="$export_mode" \
    --env UNDERWRITING_NOISE_RATE="$NOISE" \
    --env OPENAI_API_KEY="$OPENAI_API_KEY" \
    --env OPENAI_MODEL="$MODEL" \
    --env OTEL_SERVICE_NAME="$agent_name" \
    --env OTEL_RESOURCE_ATTRIBUTES="deployment.environment=aws-agentcore,service.namespace=genai-lens" \
    --env OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,splunk" \
    --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true" \
    --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE="SPAN_AND_EVENT" \
    --env OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION="true" \
    --env OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION="replace-category:SplunkEvaluationResults" \
    --env OTEL_INSTRUMENTATION_GENAI_DEBUG="false" \
    "$@"
}

case "$VARIANT" in
  ao|both)
    require OPENAI_API_KEY
    require GALILEO_API_KEY
    : "${GALILEO_PROJECT:=ai-underwriting-pipeline}"
    : "${GALILEO_LOG_STREAM:=default}"
    require GALILEO_OTEL_TRACES_ENDPOINT
    echo "AO target: $GALILEO_OTEL_TRACES_ENDPOINT (project=$GALILEO_PROJECT logstream=$GALILEO_LOG_STREAM)"
    deploy_one "underwriting_ao" "otlp" \
      --env OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
      --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$GALILEO_OTEL_TRACES_ENDPOINT" \
      --env OTEL_EXPORTER_OTLP_TRACES_HEADERS="Galileo-API-Key=${GALILEO_API_KEY},project=${GALILEO_PROJECT},logstream=${GALILEO_LOG_STREAM}" \
      --env OTEL_LOGS_EXPORTER="none" \
      --env OTEL_METRICS_EXPORTER="none"
    ;;
esac

case "$VARIANT" in
  spans-logs|both)
    require OPENAI_API_KEY
    # No OTLP endpoint on purpose. The whole point of this variant is that the spans are
    # real and never leave the process by any route other than stdout.
    deploy_one "underwriting_spans_to_logs" "console" \
      --env OTEL_TRACES_EXPORTER="none" \
      --env OTEL_LOGS_EXPORTER="none" \
      --env OTEL_METRICS_EXPORTER="none"
    ;;
esac

case "$VARIANT" in
  ao|spans-logs|both) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

echo
echo "Done. Invoke with:"
echo "  agentcore invoke --agent underwriting_ao '{\"applicant_id\":\"APP-31001\"}'"
echo "  agentcore invoke --agent underwriting_spans_to_logs '{\"applicant_id\":\"APP-31002\"}'"
