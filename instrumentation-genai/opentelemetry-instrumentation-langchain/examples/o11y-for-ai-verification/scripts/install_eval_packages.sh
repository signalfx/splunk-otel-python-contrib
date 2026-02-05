#!/bin/bash
# Install evaluation packages in editable mode from local dev repo
# This mimics the manual installation process that was working

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔧 Installing evaluation packages from local dev repo..."
echo ""

# Use public PyPI for build dependencies
export PIP_INDEX_URL=https://pypi.org/simple

# Install packages in order with --no-deps flag for first 3
echo "📦 Installing opentelemetry-util-genai..."
pip install -e "$PROJECT_ROOT/../../../../util/opentelemetry-util-genai" --no-deps

echo "📦 Installing opentelemetry-util-genai-evals..."
pip install -e "$PROJECT_ROOT/../../../../util/opentelemetry-util-genai-evals" --no-deps

echo "📦 Installing opentelemetry-util-genai-evals-deepeval (with deps)..."
pip install -e "$PROJECT_ROOT/../../../../util/opentelemetry-util-genai-evals-deepeval"

echo ""
echo "✅ Evaluation packages installed successfully"
echo ""
echo "🔍 Verifying installation..."
python -c "import opentelemetry.util.genai; print('  ✓ opentelemetry.util.genai')"
python -c "import opentelemetry.util.genai.evals; print('  ✓ opentelemetry.util.genai.evals')"
python -c "import opentelemetry.util.evaluator.deepeval; print('  ✓ opentelemetry.util.evaluator.deepeval')"

echo ""
echo "✅ All evaluation packages verified"
