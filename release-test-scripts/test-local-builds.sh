#!/usr/bin/env bash
set -e

# Requires bash 4+ for associative arrays
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "This script requires bash 4 or higher"
    echo "Current version: $BASH_VERSION"
    echo ""
    echo "You have bash 3.2 (macOS default). Install newer bash:"
    echo "  brew install bash"
    echo ""
    echo "Then run with the newer bash explicitly:"
    echo "  /opt/homebrew/bin/bash $0 $@"
    echo "  or"
    echo "  /usr/local/bin/bash $0 $@"
    exit 1
fi

# Script to test local builds of all GenAI packages
# Usage: ./release-test-scripts/test-local-builds.sh [--keep] [--output-dir <path>]
#   --keep              Keep the dist/ directories after building (don't clean up)
#   --output-dir <path> Copy all built packages to a single directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
KEEP_BUILDS=false
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP_BUILDS=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--keep] [--output-dir <path>]"
            exit 1
            ;;
    esac
done

# Package configurations: name:path
declare -A PACKAGES=(
    ["util-genai"]="util/opentelemetry-util-genai"
    ["util-genai-evals"]="util/opentelemetry-util-genai-evals"
    ["genai-emitters-splunk"]="util/opentelemetry-util-genai-emitters-splunk"
    ["genai-evals-deepeval"]="util/opentelemetry-util-genai-evals-deepeval"
    ["instrumentation-langchain"]="instrumentation-genai/opentelemetry-instrumentation-langchain"
)

FAILED_PACKAGES=()
SUCCESSFUL_PACKAGES=()

# Create output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)  # Get absolute path
    echo "Output directory: $OUTPUT_DIR"
fi

echo "=========================================="
echo "Testing Local Builds for GenAI Packages"
if [ "$KEEP_BUILDS" = true ]; then
    echo "(Keeping build artifacts)"
fi
if [ -n "$OUTPUT_DIR" ]; then
    echo "(Copying to: $OUTPUT_DIR)"
fi
echo "=========================================="
echo ""

# Check if hatch is installed
if ! command -v hatch &> /dev/null; then
    echo "❌ Error: hatch is not installed"
    echo "Install it with: pip install hatch"
    exit 1
fi

echo "✓ hatch is installed"
echo ""

# Test each package
for package_name in "${!PACKAGES[@]}"; do
    package_path="${PACKAGES[$package_name]}"
    full_path="$REPO_ROOT/$package_path"
    
    echo "=========================================="
    echo "Testing: $package_name"
    echo "Path: $package_path"
    echo "=========================================="
    
    if [ ! -d "$full_path" ]; then
        echo "❌ Error: Directory not found: $full_path"
        FAILED_PACKAGES+=("$package_name (directory not found)")
        echo ""
        continue
    fi
    
    # Check for pyproject.toml
    if [ ! -f "$full_path/pyproject.toml" ]; then
        echo "❌ Error: pyproject.toml not found in $full_path"
        FAILED_PACKAGES+=("$package_name (no pyproject.toml)")
        echo ""
        continue
    fi
    
    echo "✓ Found pyproject.toml"
    
    # Try to build
    echo "Building package..."
    cd "$full_path"
    
    if hatch build 2>&1; then
        echo "✓ Build successful"
        
        # Check dist directory
        if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
            echo "✓ Artifacts created:"
            ls -lh dist/
            
            # Count artifacts
            wheel_count=$(ls dist/*.whl 2>/dev/null | wc -l)
            tar_count=$(ls dist/*.tar.gz 2>/dev/null | wc -l)
            
            echo "  - Wheels: $wheel_count"
            echo "  - Source distributions: $tar_count"
            
            if [ "$wheel_count" -gt 0 ] && [ "$tar_count" -gt 0 ]; then
                echo "✓ Both wheel and source distribution created"
                SUCCESSFUL_PACKAGES+=("$package_name")
            else
                echo "⚠ Warning: Missing wheel or source distribution"
                FAILED_PACKAGES+=("$package_name (incomplete artifacts)")
            fi
            
            # Copy to output directory if specified
            if [ -n "$OUTPUT_DIR" ]; then
                echo "Copying artifacts to $OUTPUT_DIR..."
                cp dist/* "$OUTPUT_DIR/"
                echo "✓ Copied to output directory"
            fi
            
            # Clean up dist directory (unless --keep flag is set)
            if [ "$KEEP_BUILDS" = false ]; then
                rm -rf dist/
                echo "✓ Cleaned up dist directory"
            else
                echo "✓ Keeping dist directory at: $full_path/dist/"
            fi
        else
            echo "❌ Error: No artifacts created in dist/"
            FAILED_PACKAGES+=("$package_name (no artifacts)")
        fi
    else
        echo "❌ Build failed"
        FAILED_PACKAGES+=("$package_name (build failed)")
    fi
    
    cd "$REPO_ROOT"
    echo ""
done

# Summary
echo "=========================================="
echo "Build Test Summary"
echo "=========================================="
echo ""

if [ ${#SUCCESSFUL_PACKAGES[@]} -gt 0 ]; then
    echo "✓ Successful builds (${#SUCCESSFUL_PACKAGES[@]}):"
    for pkg in "${SUCCESSFUL_PACKAGES[@]}"; do
        echo "  - $pkg"
    done
    echo ""
fi

if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    echo "❌ Failed builds (${#FAILED_PACKAGES[@]}):"
    for pkg in "${FAILED_PACKAGES[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    echo "Please fix the failed packages before proceeding with release."
    exit 1
else
    echo "🎉 All packages built successfully!"
    echo ""
    
    if [ -n "$OUTPUT_DIR" ]; then
        echo "All build artifacts copied to: $OUTPUT_DIR"
        echo ""
        echo "Contents:"
        ls -lh "$OUTPUT_DIR"
        echo ""
        echo "Total artifacts: $(ls -1 "$OUTPUT_DIR" | wc -l)"
        echo ""
    fi
    
    if [ "$KEEP_BUILDS" = true ]; then
        echo "Build artifacts also saved in each package's dist/ directory:"
        for package_name in "${!PACKAGES[@]}"; do
            package_path="${PACKAGES[$package_name]}"
            echo "  - $REPO_ROOT/$package_path/dist/"
        done
        echo ""
    fi
    
    echo "Next steps:"
    echo "1. Review TESTING_CI_CHANGES.md for CI/CD testing strategies"
    echo "2. Test with a pre-release tag (e.g., util-genai-v0.0.1-alpha)"
    echo "3. Monitor the GitLab CI pipeline"
    exit 0
fi
