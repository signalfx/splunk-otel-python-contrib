#!/bin/bash
# Alpha Release Testing - One-Time Setup Script
# Run this once to set up the testing environment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Alpha Release Testing - Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Warning: 'uv' not found. Install it with:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo -e "${YELLOW}Falling back to standard Python venv...${NC}"
    USE_UV=false
else
    USE_UV=true
fi

# Create virtual environment
if [ -d ".venv-langchain" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    echo -e "${GREEN}✓${NC} Creating virtual environment..."
    if [ "$USE_UV" = true ]; then
        uv venv .venv-langchain
    else
        python3 -m venv .venv-langchain
    fi
fi

# Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source .venv-langchain/bin/activate

# Install pip if using uv
if [ "$USE_UV" = true ]; then
    echo -e "${GREEN}✓${NC} Installing pip..."
    uv pip install pip
fi

# Install local Splunk packages
echo -e "${GREEN}✓${NC} Installing local Splunk packages..."
pip install -e ../../../../util/opentelemetry-util-genai --no-deps
pip install -e ../../../../util/opentelemetry-util-genai-emitters-splunk --no-deps
pip install -e ../../../../util/opentelemetry-util-genai-evals --no-deps
pip install -e ../../../../util/opentelemetry-util-genai-evals-deepeval
pip install -e ../../../../instrumentation-genai/opentelemetry-instrumentation-langchain/

# Configure environment
if [ ! -f "config/.env" ]; then
    echo -e "${GREEN}✓${NC} Creating config/.env from template..."
    cp config/.env.lab0.template config/.env
    echo -e "${YELLOW}⚠${NC}  Please edit config/.env and verify your credentials"
else
    echo -e "${GREEN}✓${NC} config/.env already exists"
fi

# Verify installation
echo ""
echo -e "${GREEN}✓${NC} Verifying installation..."
python -c "from opentelemetry.instrumentation.langchain import LangchainInstrumentor; print('  ✓ LangChain instrumentation')"
python -c "import deepeval; print('  ✓ DeepEval')"
python -c "import langchain; print('  ✓ LangChain')"
python -c "import langgraph; print('  ✓ LangGraph')"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit config/.env and add your OPENAI_API_KEY (if not already set)"
echo "2. Run tests with: ./run_tests.sh"
echo ""
