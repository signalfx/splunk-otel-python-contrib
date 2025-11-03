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

# Script to validate GitLab CI configuration
# Usage: ./release-test-scripts/validate-ci-config.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CI_FILE="$REPO_ROOT/.gitlab-ci.yml"

echo "=========================================="
echo "GitLab CI Configuration Validator"
echo "=========================================="
echo ""

# Check if .gitlab-ci.yml exists
if [ ! -f "$CI_FILE" ]; then
    echo "❌ Error: .gitlab-ci.yml not found at $CI_FILE"
    exit 1
fi

echo "✓ Found .gitlab-ci.yml"
echo ""

# Package configurations
declare -A PACKAGES=(
    ["util-genai"]="util/opentelemetry-util-genai:splunk-otel-util-genai"
    ["util-genai-evals"]="util/opentelemetry-util-genai-evals:splunk-otel-util-genai-evals"
    ["genai-emitters-splunk"]="util/opentelemetry-util-genai-emitters-splunk:splunk-otel-genai-emitters-splunk"
    ["genai-evals-deepeval"]="util/opentelemetry-util-genai-evals-deepeval:splunk-otel-genai-evals-deepeval"
    ["instrumentation-langchain"]="instrumentation-genai/opentelemetry-instrumentation-langchain:splunk-otel-instrumentation-langchain"
)

ISSUES=()

echo "Checking CI configuration..."
echo ""

# Check for each package's jobs
for package_name in "${!PACKAGES[@]}"; do
    echo "Checking jobs for: $package_name"
    
    # Check build job
    if grep -q "build-${package_name}:" "$CI_FILE"; then
        echo "  ✓ build-${package_name} job found"
    else
        echo "  ❌ build-${package_name} job NOT found"
        ISSUES+=("Missing build-${package_name} job")
    fi
    
    # Check sign job
    if grep -q "sign-${package_name}:" "$CI_FILE"; then
        echo "  ✓ sign-${package_name} job found"
    else
        echo "  ❌ sign-${package_name} job NOT found"
        ISSUES+=("Missing sign-${package_name} job")
    fi
    
    # Check deploy job
    if grep -q "deploy-${package_name}:" "$CI_FILE"; then
        echo "  ✓ deploy-${package_name} job found"
    else
        echo "  ❌ deploy-${package_name} job NOT found"
        ISSUES+=("Missing deploy-${package_name} job")
    fi
    
    # Check tag pattern
    if grep -q "${package_name}-v\[0-9\]" "$CI_FILE"; then
        echo "  ✓ Tag pattern ${package_name}-v* found"
    else
        echo "  ⚠ Warning: Tag pattern ${package_name}-v* not found (may be OK)"
    fi
    
    echo ""
done

# Check stages
echo "Checking pipeline stages..."
required_stages=("build" "sign" "deploy" "post-release")
for stage in "${required_stages[@]}"; do
    if grep -q "  - $stage" "$CI_FILE"; then
        echo "  ✓ Stage '$stage' defined"
    else
        echo "  ❌ Stage '$stage' NOT defined"
        ISSUES+=("Missing stage: $stage")
    fi
done
echo ""

# Check for common patterns
echo "Checking common patterns..."

if grep -q "hatch build" "$CI_FILE"; then
    echo "  ✓ Uses hatch for building"
else
    echo "  ⚠ Warning: 'hatch build' not found"
fi

if grep -q "sha256sum" "$CI_FILE"; then
    echo "  ✓ Creates checksums"
else
    echo "  ⚠ Warning: 'sha256sum' not found"
fi

if grep -q ".submit-signing-request" "$CI_FILE"; then
    echo "  ✓ Uses signing service"
else
    echo "  ⚠ Warning: '.submit-signing-request' not found"
fi

if grep -q "hatch.*publish" "$CI_FILE"; then
    echo "  ✓ Uses hatch for publishing"
else
    echo "  ⚠ Warning: 'hatch publish' not found"
fi

echo ""

# Check artifact paths
echo "Checking artifact paths..."
for package_name in "${!PACKAGES[@]}"; do
    if grep -q "dist/${package_name}/" "$CI_FILE"; then
        echo "  ✓ Artifact path for $package_name configured"
    else
        echo "  ⚠ Warning: Artifact path for $package_name not found"
    fi
done
echo ""

# Validate YAML syntax (basic check)
echo "Validating YAML syntax..."
if command -v python3 &> /dev/null; then
    # Check if PyYAML is installed
    if python3 -c "import yaml" 2>/dev/null; then
        # Try to parse the YAML
        yaml_error=$(python3 -c "import yaml; yaml.safe_load(open('$CI_FILE'))" 2>&1)
        if [ $? -eq 0 ]; then
            echo "  ✓ YAML syntax is valid"
        else
            echo "  ❌ YAML syntax error detected:"
            echo "$yaml_error" | head -5
            ISSUES+=("Invalid YAML syntax")
        fi
    else
        echo "  ⚠ PyYAML not installed, skipping YAML validation"
        echo "    Install with: pip install pyyaml"
    fi
else
    echo "  ⚠ Python3 not available, skipping YAML validation"
fi
echo ""

# Test tag patterns
echo "Testing tag pattern matching..."
test_tags=(
    "util-genai-v0.1.0"
    "util-genai-evals-v0.1.0"
    "genai-emitters-splunk-v0.1.0"
    "genai-evals-deepeval-v0.1.0"
    "instrumentation-langchain-v0.1.0"
)

for tag in "${test_tags[@]}"; do
    package_prefix=$(echo "$tag" | sed 's/-v[0-9].*//')
    if grep -q "${package_prefix}-v\[0-9\]" "$CI_FILE"; then
        echo "  ✓ Tag '$tag' would match pattern"
    else
        echo "  ❌ Tag '$tag' would NOT match any pattern"
        ISSUES+=("Tag pattern missing for: $tag")
    fi
done
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""

if [ ${#ISSUES[@]} -eq 0 ]; then
    echo "🎉 All checks passed!"
    echo ""
    echo "Your GitLab CI configuration looks good."
    echo ""
    echo "Next steps:"
    echo "1. Run ./scripts/test-local-builds.sh to test package builds"
    echo "2. Validate in GitLab: CI/CD → Editor → Validate"
    echo "3. Test with a pre-release tag"
    exit 0
else
    echo "❌ Found ${#ISSUES[@]} issue(s):"
    echo ""
    for issue in "${ISSUES[@]}"; do
        echo "  - $issue"
    done
    echo ""
    echo "Please fix these issues before proceeding."
    exit 1
fi
