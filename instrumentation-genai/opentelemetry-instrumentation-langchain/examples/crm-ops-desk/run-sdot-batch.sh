#!/usr/bin/env bash
# Run CRM demo scenarios in batch with SDOT instrumentation.
#
# Each run uses opentelemetry-instrument so traces flow through the OTel
# Collector (gRPC) to Splunk O11y.  Scenarios are mixed
# to exercise different eval metrics:
#
#   --drift     → expired policy → poor context_adherence / tool_selection_quality
#   scenario 1  → angry user     → toxicity flagged
#   scenario 6  → hallucination  → poor context_adherence (injected in crm_tools.py)
#   normal runs → baseline       → high scores across the board
#
# Usage:
#   ./run-sdot-batch.sh                  # 10 runs, mixed scenarios
#   ./run-sdot-batch.sh -n 20            # 20 runs
#   ./run-sdot-batch.sh -n 5 --no-drift  # 5 runs, skip drift scenarios
#   ./run-sdot-batch.sh --http-root      # add HTTP SERVER root span to every run
#
# Prerequisites:
#   - OTel Collector running on localhost:4317
#   - .env with OPENAI_API_KEY
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────
NUM_RUNS=10
INCLUDE_DRIFT=true
HTTP_ROOT=false
DELAY=60

# ── Parse args ──────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [-n NUM_RUNS] [--no-drift] [--http-root] [--delay SECS]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)            NUM_RUNS="$2"; shift 2 ;;
        --no-drift)    INCLUDE_DRIFT=false; shift ;;
        --http-root)   HTTP_ROOT=true; shift ;;
        --delay)       DELAY="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
done

# ── Credentials ─────────────────────────────────────────────────
[[ -f ~/.corporate-certs/env.sh ]] && source ~/.corporate-certs/env.sh

# ── OTel / SDOT configuration ──────────────────────────────────
export OTEL_SERVICE_NAME="crm-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_METRICS_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE="DELTA"
export OTEL_LOGS_EXPORTER="otlp"
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=crm-demo"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE="SPAN_AND_EVENT"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,splunk"

# ── App .env (OpenAI) ───────────────────────────────────────────
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# ── Venv ────────────────────────────────────────────────────────
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# ── Build run plan ──────────────────────────────────────────────
# 14 scenarios (indices 0-13):
#   0-5  normal refund scenarios → baseline (high scores)
#   1    angry user              → input_toxicity, input_tone
#   6    hallucination           → context_adherence failure
#   7    PII leakage             → input_pii, output_pii
#   8    prompt injection        → prompt_injection
#   9    toxic abusive           → input_toxicity, output_toxicity
#   10   incomplete multi-req    → completeness, action_completion
#   11   tool failures           → tool_error_rate, action_advancement
#   12   vague/rambling          → agent_efficiency, tool_selection_quality
#   13   hostile context leakage → output_tone, output_toxicity
#
# Strategy per 10-run cycle:
#   3 baseline, 1 PII, 1 injection, 1 toxic, 1 incomplete/failure,
#   1 efficiency, 1 hallucination, 1 drift

NORMAL_SCENARIOS=(0 2 3 4 5)
METRIC_SCENARIOS=(7 8 9 10 11 12 13)  # new metric-triggering scenarios

declare -a PLAN_INDEX
declare -a PLAN_DRIFT

build_plan() {
    local i=0
    while (( i < NUM_RUNS )); do
        local bucket=$(( i % 10 ))
        case $bucket in
            0|1|2)
                # Normal baseline
                PLAN_INDEX[$i]=${NORMAL_SCENARIOS[$((RANDOM % ${#NORMAL_SCENARIOS[@]}))]}
                PLAN_DRIFT[$i]=false
                ;;
            3)
                # PII leakage → input_pii, output_pii
                PLAN_INDEX[$i]=7
                PLAN_DRIFT[$i]=false
                ;;
            4)
                # Prompt injection → prompt_injection
                PLAN_INDEX[$i]=8
                PLAN_DRIFT[$i]=false
                ;;
            5)
                # Toxic abusive → input_toxicity, output_toxicity
                PLAN_INDEX[$i]=9
                PLAN_DRIFT[$i]=false
                ;;
            6)
                # Incomplete or tool failure → completeness, tool_error_rate
                local choices=(10 11)
                PLAN_INDEX[$i]=${choices[$((RANDOM % 2))]}
                PLAN_DRIFT[$i]=false
                ;;
            7)
                # Efficiency or hostile leakage → agent_efficiency, output_tone
                local choices=(12 13)
                PLAN_INDEX[$i]=${choices[$((RANDOM % 2))]}
                PLAN_DRIFT[$i]=false
                ;;
            8)
                # Hallucination → context_adherence
                PLAN_INDEX[$i]=6
                PLAN_DRIFT[$i]=false
                ;;
            9)
                # Drift — expired policy
                if $INCLUDE_DRIFT; then
                    PLAN_INDEX[$i]=$((RANDOM % 14))
                    PLAN_DRIFT[$i]=true
                else
                    PLAN_INDEX[$i]=${NORMAL_SCENARIOS[$((RANDOM % ${#NORMAL_SCENARIOS[@]}))]}
                    PLAN_DRIFT[$i]=false
                fi
                ;;
        esac
        ((i++))
    done
}

build_plan

# ── Summary ─────────────────────────────────────────────────────
DRIFT_COUNT=0
for d in "${PLAN_DRIFT[@]}"; do $d && ((DRIFT_COUNT++)) || true; done
METRIC_COUNT=0
for idx in "${PLAN_INDEX[@]}"; do [[ $idx -ge 7 ]] && ((METRIC_COUNT++)) || true; done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SDOT Batch Runner — CRM Ops Desk                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Runs:          $NUM_RUNS"
echo "║  Service:       $OTEL_SERVICE_NAME"
echo "║  Collector:     $OTEL_EXPORTER_OTLP_ENDPOINT"
echo "║  HTTP root:     $HTTP_ROOT"
echo "║  Delay:         ${DELAY}s max (~$((DELAY/2))-${DELAY}s random)"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Plan breakdown:"
echo "║    Baseline (0-5):       $((NUM_RUNS - METRIC_COUNT - DRIFT_COUNT))"
echo "║    Metric triggers (7+): $METRIC_COUNT"
echo "║    Drift (expired):      $DRIFT_COUNT"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Trap ────────────────────────────────────────────────────────
RUN_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

cleanup() {
    echo ""
    echo "════════════════════════════════════════"
    echo "  Batch interrupted"
    echo "  Completed: $RUN_COUNT / $NUM_RUNS"
    echo "  Passed:    $PASS_COUNT  Failed: $FAIL_COUNT"
    echo "════════════════════════════════════════"
    exit 0
}
trap cleanup SIGINT

# ── Scenario name lookup ────────────────────────────────────────
SCENARIO_NAMES=(
    "refund_bluetooth_earbuds"
    "refund_dryer"
    "refund_gaming_mouse"
    "refund_air_purifier"
    "refund_coffee_maker"
    "refund_speakers"
    "enquire_status_of_order"
    "pii_leak_refund"
    "prompt_injection_attempt"
    "toxic_abusive_customer"
    "incomplete_multi_request"
    "tool_failure_scenario"
    "vague_rambling_query"
    "hostile_context_leakage"
)

# ── Execute ─────────────────────────────────────────────────────
for (( i=0; i<NUM_RUNS; i++ )); do
    IDX=${PLAN_INDEX[$i]}
    IS_DRIFT=${PLAN_DRIFT[$i]}
    NAME=${SCENARIO_NAMES[$IDX]}

    # Build flags
    FLAGS="--index $IDX"
    TAGS=""

    if $IS_DRIFT; then
        FLAGS="$FLAGS --drift"
        TAGS="${TAGS}[drift] "
    fi
    if $HTTP_ROOT; then
        FLAGS="$FLAGS --http-root"
    fi

    # Tag what we expect to trigger
    case $IDX in
        1)  TAGS="${TAGS}[input_toxicity] " ;;
        6)  TAGS="${TAGS}[hallucination] " ;;
        7)  TAGS="${TAGS}[pii] " ;;
        8)  TAGS="${TAGS}[prompt_injection] " ;;
        9)  TAGS="${TAGS}[toxic_abusive] " ;;
        10) TAGS="${TAGS}[incomplete] " ;;
        11) TAGS="${TAGS}[tool_failure] " ;;
        12) TAGS="${TAGS}[efficiency] " ;;
        13) TAGS="${TAGS}[hostile_output] " ;;
    esac
    if [[ -z "$TAGS" ]]; then TAGS="[baseline] "; fi

    echo "──────────────────────────────────────────────────────────"
    echo "  Run $((i+1))/$NUM_RUNS  ${TAGS} ${NAME}"
    echo "  Flags: $FLAGS"
    echo "──────────────────────────────────────────────────────────"

    if opentelemetry-instrument python main.py $FLAGS; then
        ((PASS_COUNT++))
    else
        echo "  !! scenario failed, continuing..."
        ((FAIL_COUNT++))
    fi
    ((RUN_COUNT++))

    # Random delay between DELAY/2 and DELAY (skip after last)
    if (( i < NUM_RUNS - 1 )) && (( DELAY > 0 )); then
        HALF=$((DELAY / 2))
        RANGE=$((DELAY - HALF))
        ACTUAL_DELAY=$(( HALF + RANDOM % (RANGE + 1) ))
        echo "  Waiting ${ACTUAL_DELAY}s..."
        sleep "$ACTUAL_DELAY"
    fi
done

# ── Final report ────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Batch complete                                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Total:   $RUN_COUNT"
echo "║  Passed:  $PASS_COUNT"
echo "║  Failed:  $FAIL_COUNT"
echo "╚══════════════════════════════════════════════════════════╝"
