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

"""Abstract base class for rate limit providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelLimits:
    """Rate limits for a specific model.

    Args:
        requests_per_minute: Requests per minute.
        tokens_per_minute: Tokens per minute.
        monthly_input_tokens: Monthly input token limit.
        monthly_output_tokens: Monthly output (completion) token limit.
        weekly_tokens: Weekly total token limit.
    """

    requests_per_minute: int
    tokens_per_minute: int
    monthly_input_tokens: Optional[int] = None
    monthly_output_tokens: Optional[int] = None
    weekly_tokens: Optional[int] = None


class RateLimitProvider(ABC):
    """Abstract base class for LLM provider rate limit information.

    Subclasses provide model-specific rate limits, either hardcoded
    or fetched from provider management APIs.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'openai')."""
        ...

    @abstractmethod
    def get_limits(self, model: str) -> ModelLimits | None:
        """Get rate limits for a specific model.

        Args:
            model: The model name (e.g., 'gpt-4o-mini').

        Returns:
            ModelLimits if the model is known, None otherwise.
        """
        ...

    @abstractmethod
    def supported_models(self) -> list[str]:
        """Return list of supported model names."""
        ...
