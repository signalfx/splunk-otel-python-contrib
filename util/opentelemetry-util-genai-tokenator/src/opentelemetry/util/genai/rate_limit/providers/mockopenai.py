# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock OpenAI rate limit provider with artificially low limits.

Designed for demo and testing purposes. Limits are set so that a single
CrewAI workflow run (~80-100k tokens) will:
- Breach tokens_per_minute within a few LLM calls
- Breach requests_per_minute quickly (CrewAI makes many calls)
- Hit ≥80% of the weekly_tokens quota, triggering WARNING events

Usage:
    export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_PROVIDER=mockopenai
"""

from __future__ import annotations

from opentelemetry.util.genai.rate_limit.providers.base import (
    ModelLimits,
    RateLimitProvider,
)

# Artificially tight limits — one workflow run (~83k tokens) hits 80%+ weekly.
#
# Real gpt-4o-mini free tier:  tokens_per_minute=200k, requests_per_minute=30, weekly=100M
# Mock limits:                 tokens_per_minute=2k,   requests_per_minute=5,  weekly=100k
#
# A typical customer_support workflow:
#   ~27 LLM calls, ~83k total tokens
#   → tokens_per_minute breached after ~2-3 calls (2k limit)
#   → requests_per_minute breached after 5 calls (5 limit)
#   → weekly at 83% after one run (100k limit, 80% threshold fires)
_MOCK_OPENAI_LIMITS: dict[str, ModelLimits] = {
    "gpt-4o-mini": ModelLimits(
        requests_per_minute=5,
        tokens_per_minute=2_000,
        monthly_input_tokens=200_000,
        monthly_output_tokens=50_000,
        weekly_tokens=100_000,
    ),
    "gpt-4.1": ModelLimits(
        requests_per_minute=3,
        tokens_per_minute=1_500,
        monthly_input_tokens=100_000,
        monthly_output_tokens=25_000,
        weekly_tokens=50_000,
    ),
    "gpt-4.1-mini": ModelLimits(
        requests_per_minute=5,
        tokens_per_minute=2_000,
        monthly_input_tokens=200_000,
        monthly_output_tokens=50_000,
        weekly_tokens=100_000,
    ),
    "gpt-4.1-nano": ModelLimits(
        requests_per_minute=5,
        tokens_per_minute=2_000,
        monthly_input_tokens=200_000,
        monthly_output_tokens=50_000,
        weekly_tokens=100_000,
    ),
}


class MockOpenAIRateLimitProvider(RateLimitProvider):
    """Mock OpenAI provider with artificially low limits for testing.

    Use this provider to validate that rate limit warnings fire correctly
    during a real workflow run without needing to exhaust actual API quotas.

    Limits are calibrated so a single CrewAI customer_support workflow
    (~83k tokens, ~27 LLM calls) will:
    - CRITICAL: breach tokens_per_minute (2k) and requests_per_minute (5)
    - WARNING: hit ≥80% of weekly_tokens (100k)
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_limits(self, model: str) -> ModelLimits | None:
        """Get mock rate limits for an OpenAI model.

        Lookup is case-insensitive.
        """
        return _MOCK_OPENAI_LIMITS.get(model.lower())

    def supported_models(self) -> list[str]:
        """Return list of supported model names."""
        return list(_MOCK_OPENAI_LIMITS.keys())
