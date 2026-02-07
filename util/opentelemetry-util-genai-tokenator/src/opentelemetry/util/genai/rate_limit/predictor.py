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

"""Rate limit prediction engine.

Provides multi-limit predictions (tokens_per_minute, requests_per_minute,
weekly, monthly) and workflow completion predictions using learned patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from opentelemetry.util.genai.rate_limit.providers.base import (
    RateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.tracker import TokenTracker

_logger = logging.getLogger(__name__)


@dataclass
class RateLimitPrediction:
    """A prediction for a specific rate limit type.

    Args:
        limit_type: The type of limit (tokens_per_minute, requests_per_minute, weekly, monthly_input, monthly_output).
        current_usage: Current usage value.
        limit: The limit value.
        utilization: Current utilization as a fraction (0.0-1.0).
        will_breach: Whether the limit is expected to be breached.
        time_to_breach_seconds: Estimated seconds until breach, if applicable.
        recommendation: Human-readable recommendation string.
    """

    limit_type: str
    current_usage: int
    limit: int
    utilization: float
    will_breach: bool = False
    time_to_breach_seconds: Optional[float] = None
    recommendation: str = ""

    def is_warning(self, threshold: float = 0.8) -> bool:
        """Check if this prediction exceeds the warning threshold."""
        return self.utilization >= threshold


@dataclass
class WorkflowPrediction:
    """Prediction for whether a workflow can complete within rate limits.

    Args:
        trace_id: The trace ID of the current workflow run.
        workflow_name: Name of the workflow.
        current_tokens: Tokens consumed so far in this run.
        predicted_total_tokens: Predicted total tokens for the full workflow.
        progress_percent: Estimated completion percentage.
        tokens_remaining_estimate: Estimated tokens still needed.
        rate_limit_remaining: Tokens remaining before hitting rate limit.
        can_complete: Whether the workflow can complete without hitting limits.
        recommendation: Human-readable recommendation.
    """

    trace_id: str
    workflow_name: str
    current_tokens: int
    predicted_total_tokens: float
    progress_percent: float
    tokens_remaining_estimate: float
    rate_limit_remaining: int
    can_complete: bool
    recommendation: str = ""


class RateLimitPredictor:
    """Multi-limit rate limit prediction engine.

    Generates predictions for tokens_per_minute, requests_per_minute,
    weekly, and monthly limits based on current tracked usage and
    provider-specific limits.

    Args:
        tracker: Token usage tracker instance.
        provider: Rate limit provider for limit values.
        warning_threshold: Utilization fraction that triggers warnings (0.0-1.0).
    """

    def __init__(
        self,
        tracker: TokenTracker,
        provider: RateLimitProvider,
        warning_threshold: float = 0.8,
    ) -> None:
        self._tracker = tracker
        self._provider = provider
        self._warning_threshold = warning_threshold

    def predict_all(
        self, *, provider: str, model: str
    ) -> list[RateLimitPrediction]:
        """Generate predictions for all applicable rate limits.

        Args:
            provider: Provider name (e.g., 'openai').
            model: Model name (e.g., 'gpt-4o-mini').

        Returns:
            List of RateLimitPrediction objects, empty if model is unknown.
        """
        limits = self._provider.get_limits(model)
        if limits is None:
            return []

        predictions: list[RateLimitPrediction] = []

        # tokens_per_minute prediction
        predictions.append(
            self._predict_tokens_per_minute(
                provider, model, limits.tokens_per_minute
            )
        )

        # requests_per_minute prediction
        predictions.append(
            self._predict_requests_per_minute(
                provider, model, limits.requests_per_minute
            )
        )

        # Weekly prediction
        if limits.weekly_tokens is not None:
            predictions.append(
                self._predict_weekly(provider, model, limits.weekly_tokens)
            )

        # Monthly input prediction
        if limits.monthly_input_tokens is not None:
            predictions.append(
                self._predict_monthly_input(
                    provider, model, limits.monthly_input_tokens
                )
            )

        # Monthly output prediction
        if limits.monthly_output_tokens is not None:
            predictions.append(
                self._predict_monthly_output(
                    provider, model, limits.monthly_output_tokens
                )
            )

        return predictions

    def predict_workflow_completion(
        self,
        *,
        trace_id: str,
        workflow_name: str,
        provider: str,
        model: str,
    ) -> WorkflowPrediction | None:
        """Predict whether a running workflow can complete within rate limits.

        Returns None if there's no learned pattern for this workflow.
        """
        pattern = self._tracker.get_workflow_pattern(workflow_name)
        if pattern is None:
            return None

        trace_usage = self._tracker.get_trace_usage(trace_id)
        current_tokens = trace_usage["total_tokens"] if trace_usage else 0
        predicted_total = pattern["avg_total_tokens"]

        progress = (
            (current_tokens / predicted_total * 100)
            if predicted_total > 0
            else 0.0
        )
        tokens_remaining = max(0.0, predicted_total - current_tokens)

        # Check if remaining tokens would breach tokens_per_minute
        limits = self._provider.get_limits(model)
        tokens_per_minute_remaining = 0
        can_complete = True
        if limits is not None:
            current_tokens_per_minute = (
                self._tracker.get_current_tokens_per_minute(
                    provider=provider, model=model
                )
            )
            tokens_per_minute_remaining = (
                limits.tokens_per_minute - current_tokens_per_minute
            )
            can_complete = tokens_remaining <= tokens_per_minute_remaining

        recommendation = self._workflow_recommendation(
            can_complete,
            tokens_remaining,
            tokens_per_minute_remaining,
            progress,
        )

        return WorkflowPrediction(
            trace_id=trace_id,
            workflow_name=workflow_name,
            current_tokens=current_tokens,
            predicted_total_tokens=predicted_total,
            progress_percent=progress,
            tokens_remaining_estimate=tokens_remaining,
            rate_limit_remaining=tokens_per_minute_remaining,
            can_complete=can_complete,
            recommendation=recommendation,
        )

    def _predict_tokens_per_minute(
        self, provider: str, model: str, limit: int
    ) -> RateLimitPrediction:
        current = self._tracker.get_current_tokens_per_minute(
            provider=provider, model=model
        )
        utilization = current / limit if limit > 0 else 0.0
        return RateLimitPrediction(
            limit_type="tokens_per_minute",
            current_usage=current,
            limit=limit,
            utilization=utilization,
            will_breach=utilization >= 1.0,
            recommendation=self._generate_recommendation(
                "tokens_per_minute", current, limit, utilization
            ),
        )

    def _predict_requests_per_minute(
        self, provider: str, model: str, limit: int
    ) -> RateLimitPrediction:
        current = self._tracker.get_current_requests_per_minute(
            provider=provider, model=model
        )
        utilization = current / limit if limit > 0 else 0.0
        return RateLimitPrediction(
            limit_type="requests_per_minute",
            current_usage=current,
            limit=limit,
            utilization=utilization,
            will_breach=utilization >= 1.0,
            recommendation=self._generate_recommendation(
                "requests_per_minute", current, limit, utilization
            ),
        )

    def _predict_weekly(
        self, provider: str, model: str, limit: int
    ) -> RateLimitPrediction:
        usage = self._tracker.get_weekly_usage(provider=provider, model=model)
        current = usage["total_tokens"]
        utilization = current / limit if limit > 0 else 0.0
        return RateLimitPrediction(
            limit_type="weekly",
            current_usage=current,
            limit=limit,
            utilization=utilization,
            will_breach=utilization >= 1.0,
            recommendation=self._generate_recommendation(
                "weekly", current, limit, utilization
            ),
        )

    def _predict_monthly_input(
        self, provider: str, model: str, limit: int
    ) -> RateLimitPrediction:
        usage = self._tracker.get_monthly_usage(provider=provider, model=model)
        current = usage["total_input_tokens"]
        utilization = current / limit if limit > 0 else 0.0
        return RateLimitPrediction(
            limit_type="monthly_input",
            current_usage=current,
            limit=limit,
            utilization=utilization,
            will_breach=utilization >= 1.0,
            recommendation=self._generate_recommendation(
                "monthly_input", current, limit, utilization
            ),
        )

    def _predict_monthly_output(
        self, provider: str, model: str, limit: int
    ) -> RateLimitPrediction:
        usage = self._tracker.get_monthly_usage(provider=provider, model=model)
        current = usage["total_output_tokens"]
        utilization = current / limit if limit > 0 else 0.0
        return RateLimitPrediction(
            limit_type="monthly_output",
            current_usage=current,
            limit=limit,
            utilization=utilization,
            will_breach=utilization >= 1.0,
            recommendation=self._generate_recommendation(
                "monthly_output", current, limit, utilization
            ),
        )

    def _generate_recommendation(
        self,
        limit_type: str,
        current: int,
        limit: int,
        utilization: float,
    ) -> str:
        """Generate a human-readable recommendation based on utilization."""
        pct = utilization * 100
        if utilization >= 1.0:
            return (
                f"CRITICAL: {limit_type.upper()} limit breached! "
                f"Current: {current:,} / Limit: {limit:,} ({pct:.1f}%)"
            )
        if utilization >= self._warning_threshold:
            return (
                f"WARNING: Approaching {limit_type.upper()} limit. "
                f"Current: {current:,} / Limit: {limit:,} ({pct:.1f}%)"
            )
        return (
            f"OK: {limit_type.upper()} usage normal. "
            f"Current: {current:,} / Limit: {limit:,} ({pct:.1f}%)"
        )

    @staticmethod
    def _workflow_recommendation(
        can_complete: bool,
        tokens_remaining: float,
        tokens_per_minute_remaining: int,
        progress: float,
    ) -> str:
        if can_complete:
            return (
                f"OK: Workflow can complete. "
                f"Progress: {progress:.1f}%, "
                f"Tokens remaining: ~{tokens_remaining:,.0f}, "
                f"Rate limit headroom: {tokens_per_minute_remaining:,}"
            )
        return (
            f"CRITICAL: Workflow may hit rate limit before completion! "
            f"Progress: {progress:.1f}%, "
            f"Tokens needed: ~{tokens_remaining:,.0f}, "
            f"Rate limit remaining: {tokens_per_minute_remaining:,}"
        )
