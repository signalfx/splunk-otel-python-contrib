#!/bin/bash
# Alpha Release Testing - Automated Test Runner
# This script runs all test applications with proper environment setup
#
# Usage:
#   ./run_tests.sh                  # Run all tests once
#   ./run_tests.sh langchain        # Run only LangChain test
#   ./run_tests.sh langgraph        # Run only LangGraph test
#   ./run_tests.sh loop_30          # Run all tests every 30 seconds
#   ./run_tests.sh langchain loop_30 # Run LangChain test every 30 seconds

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
LOOP_MODE=false
LOOP_INTERVAL=30
TEST_SELECTION="all"  # all, langchain, langgraph

# Parse first argument
if [ $# -gt 0 ]; then
    case $1 in
        langchain)
            TEST_SELECTION="langchain"
            shift
            ;;
        langgraph)
            TEST_SELECTION="langgraph"
            shift
            ;;
        loop_*)
            # First arg is loop, no test selection
            ;;
        *)
            echo -e "${RED}Invalid argument: $1${NC}"
            echo "Usage:"
            echo "  ./run_tests.sh                  # Run all tests once"
            echo "  ./run_tests.sh langchain        # Run only LangChain test"
            echo "  ./run_tests.sh langgraph        # Run only LangGraph test"
            echo "  ./run_tests.sh loop_30          # Run all tests every 30 seconds"
            echo "  ./run_tests.sh langchain loop_30 # Run LangChain test every 30 seconds"
            echo "  ./run_tests.sh langgraph loop_60 # Run LangGraph test every 60 seconds"
            exit 1
            ;;
    esac
fi

# Parse second argument (loop mode)
if [ $# -gt 0 ]; then
    if [[ $1 =~ ^loop_([0-9]+)$ ]]; then
        LOOP_MODE=true
        LOOP_INTERVAL=${BASH_REMATCH[1]}
        echo -e "${YELLOW}Loop mode enabled: Running tests every ${LOOP_INTERVAL} seconds${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Alpha Release Testing - Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv-langchain" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run setup first:"
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source .venv-langchain/bin/activate

# Check if .env exists
if [ ! -f "config/.env" ]; then
    echo -e "${RED}Error: config/.env not found!${NC}"
    echo "Please create it from template:"
    echo "  cp config/.env.lab0.template config/.env"
    exit 1
fi

# Export all environment variables from .env
echo -e "${GREEN}✓${NC} Loading environment variables..."
set -a
source config/.env
set +a

# Verify OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set in config/.env${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment configured"
echo ""

# Function to run tests
run_tests() {
    local iteration=$1
    
    if [ "$LOOP_MODE" = true ]; then
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}Iteration #${iteration} - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo -e "${YELLOW}========================================${NC}"
        echo ""
    fi
    
    # Navigate to test apps
    cd "$SCRIPT_DIR/tests/apps"
    
    TEST1_STATUS=0
    TEST2_STATUS=0
    
    # Run Test 1: LangChain Evaluation (if selected)
    if [ "$TEST_SELECTION" = "all" ] || [ "$TEST_SELECTION" = "langchain" ]; then
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}Test 1: LangChain Evaluation App${NC}"
        echo -e "${BLUE}========================================${NC}"
        python langchain_evaluation_app.py
        TEST1_STATUS=$?
        
        echo ""
        echo ""
    fi
    
    # Run Test 2: LangGraph Travel Planner (if selected)
    if [ "$TEST_SELECTION" = "all" ] || [ "$TEST_SELECTION" = "langgraph" ]; then
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}Test 2: LangGraph Travel Planner${NC}"
        echo -e "${BLUE}========================================${NC}"
        python langgraph_travel_planner_app.py
        TEST2_STATUS=$?
        
        echo ""
        echo ""
    fi
    
    # Summary
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Test Summary - Iteration #${iteration}${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ "$TEST_SELECTION" = "all" ] || [ "$TEST_SELECTION" = "langchain" ]; then
        if [ $TEST1_STATUS -eq 0 ]; then
            echo -e "${GREEN}✓${NC} LangChain Evaluation App: PASSED"
        else
            echo -e "${RED}✗${NC} LangChain Evaluation App: FAILED"
        fi
    fi
    
    if [ "$TEST_SELECTION" = "all" ] || [ "$TEST_SELECTION" = "langgraph" ]; then
        if [ $TEST2_STATUS -eq 0 ]; then
            echo -e "${GREEN}✓${NC} LangGraph Travel Planner: PASSED"
        else
            echo -e "${RED}✗${NC} LangGraph Travel Planner: FAILED"
        fi
    fi
    
    echo ""
    
    if [ "$LOOP_MODE" = false ]; then
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}Next Steps:${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo "1. Check Splunk APM (lab0): https://app.lab0.signalfx.com"
        echo "2. Navigate to: APM → Agents"
        echo "3. Find service: alpha-release-test"
        echo "4. Verify telemetry, metrics, and traces"
        echo ""
    fi
    
    # Return status
    if [ $TEST1_STATUS -ne 0 ] || [ $TEST2_STATUS -ne 0 ]; then
        return 1
    fi
    return 0
}

# Main execution
if [ "$LOOP_MODE" = true ]; then
    # Loop mode - run continuously
    ITERATION=1
    while true; do
        run_tests $ITERATION
        
        echo -e "${YELLOW}Waiting ${LOOP_INTERVAL} seconds before next iteration...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        sleep $LOOP_INTERVAL
        ITERATION=$((ITERATION + 1))
    done
else
    # Single run mode
    run_tests 1
    
    # Exit with failure if any test failed
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi
