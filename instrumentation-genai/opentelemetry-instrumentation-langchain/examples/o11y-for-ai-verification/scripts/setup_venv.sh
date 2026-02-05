#!/bin/bash
################################################################################
# Virtual Environment Setup Script
# 
# Purpose: Create and configure Python virtual environment with all dependencies
#
# Usage: ./scripts/setup_venv.sh [--force]
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
FORCE_REINSTALL=false

# Parse arguments
if [ "$1" = "--force" ]; then
    FORCE_REINSTALL=true
fi

echo "=========================================="
echo "🔧 Virtual Environment Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "📌 Detected Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ ERROR: Python 3.10+ required"
    echo "   Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version compatible"
echo ""

# Handle existing virtual environment
if [ -d "$VENV_PATH" ]; then
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
    else
        echo "ℹ️  Virtual environment already exists: $VENV_PATH"
        echo "   Use --force to recreate"
        echo ""
        
        # Check if dependencies are installed
        source "$VENV_PATH/bin/activate"
        
        echo "🔍 Verifying installed dependencies..."
        MISSING_DEPS=false
        
        # Check critical dependencies
        CRITICAL_DEPS=("pytest" "playwright" "requests" "pyyaml" "opentelemetry-api")
        
        for dep in "${CRITICAL_DEPS[@]}"; do
            if ! python3 -c "import ${dep//-/_}" 2>/dev/null; then
                echo "   ❌ Missing: $dep"
                MISSING_DEPS=true
            fi
        done
        
        if [ "$MISSING_DEPS" = true ]; then
            echo ""
            echo "⚠️  Some dependencies are missing!"
            echo "   Installing missing dependencies..."
            PIP_INDEX_URL=https://pypi.org/simple "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_FILE"
            echo "✅ Dependencies installed"
        else
            echo "✅ All critical dependencies present"
        fi
        
        deactivate
        echo ""
        echo "✅ Virtual environment ready: $VENV_PATH"
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv "$VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ ERROR: Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created: $VENV_PATH"
echo ""

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet --index-url https://pypi.org/simple

# Install dependencies
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ ERROR: Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

echo "📥 Installing dependencies from requirements.txt..."
echo "   This may take 2-3 minutes..."
echo ""

if pip install -r "$REQUIREMENTS_FILE" --index-url https://pypi.org/simple; then
    echo ""
    echo "✅ All dependencies installed successfully"
else
    echo ""
    echo "❌ ERROR: Failed to install dependencies"
    exit 1
fi

# Install Playwright browsers (required for UI tests)
echo ""
echo "🌐 Installing Playwright browsers..."
if playwright install chromium; then
    echo "✅ Playwright browsers installed"
else
    echo "⚠️  Playwright browser installation failed (optional)"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
echo ""

VERIFICATION_PASSED=true

# Test critical imports
CRITICAL_MODULES=(
    "pytest:pytest"
    "playwright:playwright"
    "requests:requests"
    "yaml:pyyaml"
    "opentelemetry.trace:opentelemetry-api"
)

for module_info in "${CRITICAL_MODULES[@]}"; do
    IFS=':' read -r module package <<< "$module_info"
    if python3 -c "import $module" 2>/dev/null; then
        echo "   ✅ $package"
    else
        echo "   ❌ $package (import failed)"
        VERIFICATION_PASSED=false
    fi
done

deactivate

echo ""
echo "=========================================="

if [ "$VERIFICATION_PASSED" = true ]; then
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Virtual environment ready at: $VENV_PATH"
    echo ""
    echo "To activate:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Next steps:"
    echo "  1. Set environment: source scripts/setup_environment.sh"
    echo "  2. Run tests: ./scripts/run_tests.sh"
    echo "  3. Or run all: ./scripts/run_all.sh"
    echo ""
    exit 0
else
    echo "⚠️  Setup Complete with Warnings"
    echo "=========================================="
    echo ""
    echo "Some dependencies failed verification."
    echo "Tests may not run correctly."
    echo ""
    exit 1
fi
