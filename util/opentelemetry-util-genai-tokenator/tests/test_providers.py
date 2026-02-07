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

"""Tests for providers/ - Rate limit provider implementations."""

import pytest

from opentelemetry.util.genai.rate_limit.providers.base import (
    ModelLimits,
    RateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.providers.mockopenai import (
    MockOpenAIRateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.providers.openai import (
    OpenAIRateLimitProvider,
)


class TestModelLimits:
    """Test the ModelLimits dataclass."""

    def test_model_limits_creation(self) -> None:
        limits = ModelLimits(
            requests_per_minute=30,
            tokens_per_minute=200_000,
            monthly_input_tokens=500_000_000,
            monthly_output_tokens=50_000_000,
            weekly_tokens=100_000_000,
        )
        assert limits.requests_per_minute == 30
        assert limits.tokens_per_minute == 200_000
        assert limits.monthly_input_tokens == 500_000_000
        assert limits.monthly_output_tokens == 50_000_000
        assert limits.weekly_tokens == 100_000_000

    def test_model_limits_defaults(self) -> None:
        """Optional fields should default to None."""
        limits = ModelLimits(requests_per_minute=10, tokens_per_minute=100_000)
        assert limits.monthly_input_tokens is None
        assert limits.monthly_output_tokens is None
        assert limits.weekly_tokens is None


class TestOpenAIRateLimitProvider:
    """Test OpenAI rate limit provider with hardcoded free tier limits."""

    def test_provider_name(self) -> None:
        provider = OpenAIRateLimitProvider()
        assert provider.provider_name == "openai"

    def test_get_limits_gpt4o_mini(self) -> None:
        provider = OpenAIRateLimitProvider()
        limits = provider.get_limits("gpt-4o-mini")
        assert limits is not None
        assert limits.requests_per_minute == 30
        assert limits.tokens_per_minute == 200_000
        assert limits.monthly_input_tokens == 500_000_000
        assert limits.monthly_output_tokens == 50_000_000
        assert limits.weekly_tokens == 100_000_000

    def test_get_limits_gpt41(self) -> None:
        provider = OpenAIRateLimitProvider()
        limits = provider.get_limits("gpt-4.1")
        assert limits is not None
        assert limits.requests_per_minute == 15
        assert limits.tokens_per_minute == 1_000_000

    def test_get_limits_unknown_model_returns_none(self) -> None:
        provider = OpenAIRateLimitProvider()
        limits = provider.get_limits("unknown-model-xyz")
        assert limits is None

    def test_get_limits_case_insensitive(self) -> None:
        """Model lookup should be case-insensitive."""
        provider = OpenAIRateLimitProvider()
        limits = provider.get_limits("GPT-4O-MINI")
        assert limits is not None
        assert limits.tokens_per_minute == 200_000

    def test_supported_models(self) -> None:
        provider = OpenAIRateLimitProvider()
        models = provider.supported_models()
        assert "gpt-4o-mini" in models
        assert "gpt-4.1" in models
        assert len(models) >= 2


class TestMockOpenAIRateLimitProvider:
    """Test mock OpenAI provider with artificially low limits."""

    def test_provider_name(self) -> None:
        provider = MockOpenAIRateLimitProvider()
        assert provider.provider_name == "openai"

    def test_get_limits_gpt4o_mini(self) -> None:
        provider = MockOpenAIRateLimitProvider()
        limits = provider.get_limits("gpt-4o-mini")
        assert limits is not None
        assert limits.requests_per_minute == 5
        assert limits.tokens_per_minute == 2_000
        assert limits.weekly_tokens == 100_000
        assert limits.monthly_input_tokens == 200_000
        assert limits.monthly_output_tokens == 50_000

    def test_limits_much_lower_than_real(self) -> None:
        """Mock limits should be significantly lower than real OpenAI limits."""
        real = OpenAIRateLimitProvider().get_limits("gpt-4o-mini")
        mock = MockOpenAIRateLimitProvider().get_limits("gpt-4o-mini")
        assert real is not None and mock is not None
        assert mock.tokens_per_minute < real.tokens_per_minute / 10
        assert mock.requests_per_minute < real.requests_per_minute
        assert mock.weekly_tokens < real.weekly_tokens / 100

    def test_workflow_triggers_weekly_warning(self) -> None:
        """A single ~83k token workflow should exceed 80% of weekly_tokens."""
        provider = MockOpenAIRateLimitProvider()
        limits = provider.get_limits("gpt-4o-mini")
        assert limits is not None
        typical_workflow_tokens = 83_000
        utilization = typical_workflow_tokens / limits.weekly_tokens
        assert utilization >= 0.80, (
            f"Expected ≥80% weekly utilization, got {utilization:.1%}"
        )

    def test_get_limits_unknown_model_returns_none(self) -> None:
        provider = MockOpenAIRateLimitProvider()
        assert provider.get_limits("unknown-model") is None

    def test_supported_models_same_as_real(self) -> None:
        """Mock should support the same models as the real provider."""
        real_models = set(OpenAIRateLimitProvider().supported_models())
        mock_models = set(MockOpenAIRateLimitProvider().supported_models())
        assert mock_models == real_models

    def test_case_insensitive_lookup(self) -> None:
        provider = MockOpenAIRateLimitProvider()
        limits = provider.get_limits("GPT-4O-MINI")
        assert limits is not None
        assert limits.tokens_per_minute == 2_000


class TestRateLimitProviderABC:
    """Test that RateLimitProvider is a proper abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            RateLimitProvider()  # type: ignore[abstract]

    def test_subclass_must_implement(self) -> None:
        """Subclass must implement abstract methods."""

        class IncompleteProvider(RateLimitProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]
