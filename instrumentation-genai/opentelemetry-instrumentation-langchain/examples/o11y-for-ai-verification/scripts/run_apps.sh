#!/bin/bash
################################################################################
# Automated Test Application Execution Script
# 
# Purpose: Run all test scenarios from unified_genai_test_app.py
#          and capture trace IDs for subsequent test validation
#
# Usage: ./scripts/run_apps.sh [--scenario SCENARIO_NAME] [--iterations N]
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APPS_DIR="$PROJECT_ROOT/tests/apps"
TRACE_IDS_FILE="$PROJECT_ROOT/.trace_ids.json"
VENV_PATH="$PROJECT_ROOT/.venv"

# Default values
SCENARIO="${1:-all}"
ITERATIONS="${2:-1}"
WAIT_BETWEEN_RUNS=10

echo "=========================================="
echo "🚀 Running Test Applications"
echo "=========================================="
echo "Scenario: $SCENARIO"
echo "Iterations: $ITERATIONS"
echo ""

# Verify virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found at: $VENV_PATH"
    echo "Please create it first: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify environment variables
if [ -z "$SPLUNK_ACCESS_TOKEN" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: Required environment variables not set"
    echo "Please run: source scripts/setup_environment.sh"
    exit 1
fi

# Initialize trace IDs file
echo "{" > "$TRACE_IDS_FILE"
echo '  "generated_at": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",' >> "$TRACE_IDS_FILE"
echo '  "scenarios": {' >> "$TRACE_IDS_FILE"

# Define scenarios to run
if [ "$SCENARIO" = "all" ]; then
    SCENARIOS=(
        "multi_agent_retail"
        "langgraph_workflow"
        "rag_pipeline"
        "streaming_ttft"
        "multi_provider_edge_cases"
        "litellm_proxy"
        "eval_queue_management"
        "eval_error_handling"
        "eval_monitoring_metrics"
        "retail_evaluation_tests"
    )
else
    SCENARIOS=("$SCENARIO")
fi

TOTAL_SCENARIOS=${#SCENARIOS[@]}
CURRENT=0

# Run each scenario
for scenario in "${SCENARIOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo ""
    echo "=========================================="
    echo "📊 Scenario $CURRENT/$TOTAL_SCENARIOS: $scenario"
    echo "=========================================="
    
    for iteration in $(seq 1 $ITERATIONS); do
        echo ""
        echo "🔄 Iteration $iteration/$ITERATIONS"
        echo ""
        
        # Run the app and capture output
        OUTPUT_FILE="/tmp/app_output_${scenario}_${iteration}.log"
        
        cd "$APPS_DIR"
        python unified_genai_test_app.py --scenario "$scenario" 2>&1 | tee "$OUTPUT_FILE"
        
        # Extract trace ID from output (looking for common patterns)
        TRACE_ID=$(grep -oE "trace[_-]?id[: ]*[a-f0-9]{32}" "$OUTPUT_FILE" | head -1 | grep -oE "[a-f0-9]{32}" || echo "")
        
        if [ -z "$TRACE_ID" ]; then
            # Try alternative pattern
            TRACE_ID=$(grep -oE "[a-f0-9]{32}" "$OUTPUT_FILE" | head -1 || echo "not_captured")
        fi
        
        echo ""
        echo "✅ Completed iteration $iteration"
        echo "   Trace ID: $TRACE_ID"
        
        # Store trace ID
        if [ $iteration -eq 1 ]; then
            echo "    \"$scenario\": [" >> "$TRACE_IDS_FILE"
        fi
        
        if [ $iteration -lt $ITERATIONS ]; then
            echo "      \"$TRACE_ID\"," >> "$TRACE_IDS_FILE"
        else
            echo "      \"$TRACE_ID\"" >> "$TRACE_IDS_FILE"
        fi
        
        # Wait between iterations
        if [ $iteration -lt $ITERATIONS ]; then
            echo "   Waiting ${WAIT_BETWEEN_RUNS}s before next iteration..."
            sleep $WAIT_BETWEEN_RUNS
        fi
    done
    
    # Close scenario array
    if [ $CURRENT -lt $TOTAL_SCENARIOS ]; then
        echo "    ]," >> "$TRACE_IDS_FILE"
    else
        echo "    ]" >> "$TRACE_IDS_FILE"
    fi
    
    echo ""
    echo "✅ Scenario $scenario completed ($ITERATIONS iterations)"
    
    # Wait between scenarios
    if [ $CURRENT -lt $TOTAL_SCENARIOS ]; then
        echo "   Waiting ${WAIT_BETWEEN_RUNS}s before next scenario..."
        sleep $WAIT_BETWEEN_RUNS
    fi
done

# Close JSON
echo "  }" >> "$TRACE_IDS_FILE"
echo "}" >> "$TRACE_IDS_FILE"

echo ""
echo "=========================================="
echo "✅ All Applications Executed Successfully"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Scenarios run: $TOTAL_SCENARIOS"
echo "  - Iterations per scenario: $ITERATIONS"
echo "  - Total executions: $((TOTAL_SCENARIOS * ITERATIONS))"
echo "  - Trace IDs saved to: $TRACE_IDS_FILE"
echo ""
echo "Next step: Run tests with captured trace IDs"
echo "  ./scripts/run_tests.sh"
echo ""
