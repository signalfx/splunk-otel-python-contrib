#!/bin/bash
# Run random scenarios every 10 minutes
# Usage: ./run_scenarios.sh [server_url]
#
# Examples:
#   ./run_scenarios.sh                     # Uses http://localhost:8080
#   ./run_scenarios.sh http://sre-copilot:8080

SERVER_URL="${1:-http://localhost:8080}"
INTERVAL_SECONDS=600  # 10 minutes

echo "SRE Copilot Scenario Runner"
echo "Server: $SERVER_URL"
echo "Interval: $INTERVAL_SECONDS seconds"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Generate random scenario ID (001-010)
    SCENARIO_NUM=$(printf "%03d" $((RANDOM % 10 + 1)))
    SCENARIO_ID="scenario-$SCENARIO_NUM"
    
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Running $SCENARIO_ID"
    
    # Run the scenario
    RESPONSE=$(curl -s -X POST "$SERVER_URL/run" \
        -H "Content-Type: application/json" \
        -d "{\"scenario_id\": \"$SCENARIO_ID\"}")
    
    # Extract key info from response
    RUN_ID=$(echo "$RESPONSE" | grep -o '"run_id":"[^"]*"' | cut -d'"' -f4)
    PASSED=$(echo "$RESPONSE" | grep -o '"validation_passed":[^,}]*' | cut -d':' -f2)
    CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence_score":[^,}]*' | cut -d':' -f2)
    
    if [ -n "$RUN_ID" ]; then
        echo "[$TIMESTAMP] Complete: $RUN_ID | passed=$PASSED | confidence=$CONFIDENCE"
    else
        echo "[$TIMESTAMP] Error: $RESPONSE"
    fi
    
    echo "[$TIMESTAMP] Next run in $INTERVAL_SECONDS seconds..."
    echo ""
    
    sleep $INTERVAL_SECONDS
done
