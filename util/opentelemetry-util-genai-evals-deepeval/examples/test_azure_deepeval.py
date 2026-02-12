#!/usr/bin/env python3
import os

from opentelemetry.util.evaluator.deepeval import DeepevalEvaluator
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

"""This script can be used to verify DEEPEVAL_* env vars for different LLM setups

Set the env var and python test_azure_deepeval.py
"""

# DeepEval LLM Configuration with OAuth3
# os.environ["DEEPEVAL_LLM_BASE_URL"] = "https://<YOUR_LLM_provider>/openai/deployments/gpt-4o-mini"
# os.environ["DEEPEVAL_LLM_MODEL"] = "gpt-4o-mini"
# os.environ["DEEPEVAL_LLM_PROVIDER"] = "openai"
# os.environ["DEEPEVAL_LLM_CLIENT_ID"] = "<YOUR_CLIENT_ID>"
# os.environ["DEEPEVAL_LLM_CLIENT_SECRET"] = "<YOUR_CLIENT_SECRET>"
# os.environ["DEEPEVAL_LLM_TOKEN_URL"] = "<YOUR_TOKEN_URL>"
# os.environ["DEEPEVAL_LLM_CLIENT_APP_NAME"] = "<YOUR_APP_NAME>"

# Azure OpenAI use-case scenario 1
os.environ["DEEPEVAL_LLM_BASE_URL"] = "https://<YOUR_AZURE_BASE_URL>/openai/v1"
os.environ["DEEPEVAL_LLM_MODEL"] = "gpt-4.1-nano"
os.environ["DEEPEVAL_LLM_PROVIDER"] = "openai"
os.environ["DEEPEVAL_LLM_API_KEY"] = "<YOUR_API_KEY>"
# os.environ["DEEPEVAL_LLM_EXTRA_HEADERS"] = '{"api_version":"2025-01-01-preview"}'

# Azure OpenAI use-case scenario 3
# os.environ["DEEPEVAL_LLM_BASE_URL"] = "<YOUR_AZURE_BASE_URL>"
# os.environ["DEEPEVAL_LLM_MODEL"] = "gpt-4.1-nano"
# os.environ["DEEPEVAL_LLM_PROVIDER"] = "azure"
# os.environ["DEEPEVAL_LLM_API_KEY"] = (
#     "<YOUR_API_KEY>"
# )

# GPT 5 models
# os.environ["DEEPEVAL_LLM_MODEL"] = "gpt-5-nano"
# os.environ["DEEPEVAL_LLM_PROVIDER"] = "openai"
# os.environ["TEMPERATURE"] = "1"

evaluator = DeepevalEvaluator(
    metrics=["bias"], invocation_type="LLMInvocation"
)

invocation = LLMInvocation(request_model="gpt-4o-mini")
invocation.input_messages.append(
    InputMessage(
        role="user", parts=[Text(content="What is the capital of France?")]
    )
)
invocation.output_messages.append(
    OutputMessage(
        role="assistant",
        parts=[Text(content="The capital of France is Paris.")],
        finish_reason="stop",
    )
)

try:
    results = evaluator.evaluate(invocation)
    for r in results:
        print(f"\nMetric: {r.metric_name}")
        print(f"Score: {getattr(r, 'score', 'N/A')}")
        print(f"Label: {getattr(r, 'label', 'N/A')}")
        if r.explanation:
            print(f"Explanation: {r.explanation[:300]}...")
        if r.error:
            print(f"Error: {r.error}")
    print("=" * 80)
except Exception as e:
    import traceback

    print("\n" + "=" * 80)
    print(f"ERROR: {type(e).__name__}: {e}")
    print("=" * 80)
    traceback.print_exc()
