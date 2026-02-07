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

"""OpenAI rate limit provider with hardcoded free tier limits.

Limits are based on OpenAI free tier documentation as of 2026-02-06.
For production use, these should be fetched from the OpenAI management
API or configured via environment variables.
"""

from __future__ import annotations

from opentelemetry.util.genai.rate_limit.providers.base import (
    ModelLimits,
    RateLimitProvider,
)

# Hardcoded free tier limits as of 2026-02-06
_OPENAI_FREE_TIER_LIMITS: dict[str, ModelLimits] = {
    "gpt-4o-mini": ModelLimits(
        requests_per_minute=30,
        tokens_per_minute=200_000,
        monthly_input_tokens=500_000_000,
        monthly_output_tokens=50_000_000,
        weekly_tokens=100_000_000,
    ),
    "gpt-4.1": ModelLimits(
        requests_per_minute=15,
        tokens_per_minute=1_000_000,
        monthly_input_tokens=50_000_000,
        monthly_output_tokens=5_000_000,
        weekly_tokens=15_000_000,
    ),
    "gpt-4.1-mini": ModelLimits(
        requests_per_minute=30,
        tokens_per_minute=200_000,
        monthly_input_tokens=500_000_000,
        monthly_output_tokens=50_000_000,
        weekly_tokens=100_000_000,
    ),
    "gpt-4.1-nano": ModelLimits(
        requests_per_minute=30,
        tokens_per_minute=200_000,
        monthly_input_tokens=500_000_000,
        monthly_output_tokens=50_000_000,
        weekly_tokens=100_000_000,
    ),
}


class OpenAIRateLimitProvider(RateLimitProvider):
    """OpenAI rate limit provider using hardcoded free tier limits.

    Note: These limits are from OpenAI free tier as of 2026-02-06.
    Production deployments should fetch limits from the OpenAI API
    or configure them via environment variables.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_limits(self, model: str) -> ModelLimits | None:
        """Get rate limits for an OpenAI model.

        Lookup is case-insensitive.
        """
        return _OPENAI_FREE_TIER_LIMITS.get(model.lower())

    def supported_models(self) -> list[str]:
        """Return list of supported OpenAI model names."""
        return list(_OPENAI_FREE_TIER_LIMITS.keys())
