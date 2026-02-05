# Configuration Guide

## Overview

This directory contains **common configuration templates** shared by all test applications.
App-specific configuration is defined within each app's code.

## Configuration Files

| File | Purpose |
|------|---------|
| `azure_openai.env.template` | Azure OpenAI credentials template (copy to `azure_openai.env`) |
| `.env.template` | Comprehensive reference of all available options |
| `README.md` | This documentation |

### Quick Start (Recommended)

```bash
# 1. Copy Azure credentials template
cp azure_openai.env.template azure_openai.env

# 2. Edit with your Azure OpenAI credentials
nano azure_openai.env

# 3. Source the setup script (auto-configures everything)
source ../scripts/setup_environment.sh

# 4. Run any test app
cd ../tests/apps
python direct_azure_openai_app_v2.py
```

### Configuration Reference

**`.env.template`** - Complete reference with all options documented
- All OpenTelemetry GenAI instrumentation settings
- Evaluation configuration (all 5 metrics)
- DeepEval judge model configuration
- Debug options
- **Use this for:** Understanding all available options

## Configuration Priority

Settings are loaded in this order (highest to lowest priority):

1. **Environment variables** (highest priority)
2. **`scripts/setup_environment.sh`** (auto-configures common settings)
3. **`config/azure_openai.env`** (Azure credentials)
4. **Application defaults** (lowest priority)

## Required Configuration

### Azure OpenAI Credentials

Create `azure_openai.env` from the template:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

These credentials are used for:
1. Main LLM calls in your application
2. DeepEval judge model for evaluation metrics

## Security Best Practices

- All `.env` files are gitignored (only `.template` files are tracked)
- Never commit credentials to version control
- Use secrets management in production

## See Also

- [Main README](../README.md) - Project overview
- [tests/apps/README.md](../tests/apps/README.md) - Test application documentation
