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
    --non-interactive

  # shellcheck disable=SC2086
  agentcore launch \
    --name "$agent_name" \
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
echo "  agentcore invoke --name underwriting_ao '{\"applicant_id\":\"APP-31001\"}'"
echo "  agentcore invoke --name underwriting_spans_to_logs '{\"applicant_id\":\"APP-31002\"}'"
