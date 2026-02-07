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

"""Tokenator emitter - rate limit prediction via the GenAI emitter pipeline.

Integrates with the opentelemetry-util-genai emitter system via entry points.
Records token usage to SQLite, runs predictions, and emits OTel warning events.

Entry point: rate_limit_predictor
Usage: OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

from opentelemetry.util.genai.emitters.spec import EmitterSpec
from opentelemetry.util.genai.interfaces import EmitterMeta
from opentelemetry.util.genai.rate_limit.predictor import RateLimitPredictor
from opentelemetry.util.genai.rate_limit.providers.base import (
    RateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.providers.mockopenai import (
    MockOpenAIRateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.providers.openai import (
    OpenAIRateLimitProvider,
)
from opentelemetry.util.genai.rate_limit.tracker import TokenTracker
from opentelemetry.util.genai.types import (
    EvaluationResult,
    LLMInvocation,
    Workflow,
)

_logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_DB_PATH = os.path.join(
    Path.home(), ".opentelemetry_genai_rate_limit.db"
)
_DEFAULT_WARNING_THRESHOLD = 0.8
_DEFAULT_EMA_ALPHA = 0.3

# Environment variable names
_ENV_DB_PATH = "OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_DB_PATH"
_ENV_WARNING_THRESHOLD = (
    "OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_WARNING_THRESHOLD"
)
_ENV_EMA_ALPHA = "OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_EMA_ALPHA"
_ENV_PROVIDER = "OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_PROVIDER"

_PROVIDERS: dict[str, type[RateLimitProvider]] = {
    "openai": OpenAIRateLimitProvider,
    "mockopenai": MockOpenAIRateLimitProvider,
}

_EVENT_NAME_RATE_LIMIT = "gen_ai.rate_limit.warning"
_EVENT_NAME_WORKFLOW_COMPLETION = "gen_ai.workflow.completion.warning"


class RateLimitPredictorEmitter(EmitterMeta):
    """Emitter that tracks token usage and predicts rate limit breaches.

    Records every LLM invocation's token usage to SQLite, aggregates
    at trace/workflow level, and emits OpenTelemetry warning events
    when approaching rate limits.

    Codename: Tokenator — "I'll be back... before you hit your rate limit"
    """

    role = "content_event"
    name = "rate_limit_predictor"

    def __init__(
        self,
        *,
        db_path: str | None = None,
        warning_threshold: float | None = None,
        ema_alpha: float | None = None,
        event_logger: Any = None,
    ) -> None:
        resolved_db_path = db_path or os.environ.get(
            _ENV_DB_PATH, _DEFAULT_DB_PATH
        )
        resolved_threshold = warning_threshold
        if resolved_threshold is None:
            try:
                resolved_threshold = float(
                    os.environ.get(
                        _ENV_WARNING_THRESHOLD,
                        str(_DEFAULT_WARNING_THRESHOLD),
                    )
                )
            except (TypeError, ValueError):
                resolved_threshold = _DEFAULT_WARNING_THRESHOLD

        resolved_alpha = ema_alpha
        if resolved_alpha is None:
            try:
                resolved_alpha = float(
                    os.environ.get(_ENV_EMA_ALPHA, str(_DEFAULT_EMA_ALPHA))
                )
            except (TypeError, ValueError):
                resolved_alpha = _DEFAULT_EMA_ALPHA

        self._tracker = TokenTracker(
            db_path=resolved_db_path, ema_alpha=resolved_alpha
        )
        provider_key = os.environ.get(_ENV_PROVIDER, "openai").lower()
        provider_cls = _PROVIDERS.get(provider_key, OpenAIRateLimitProvider)
        self._provider = provider_cls()
        self._predictor = RateLimitPredictor(
            tracker=self._tracker,
            provider=self._provider,
            warning_threshold=resolved_threshold,
        )
        self._event_logger = event_logger
        self._warning_threshold = resolved_threshold

    def handles(self, obj: Any) -> bool:
        """Accept LLMInvocation and Workflow objects."""
        return isinstance(obj, (LLMInvocation, Workflow))

    def on_start(self, obj: Any) -> None:
        """No-op on start; tracking happens on end."""
        return None

    def on_end(self, obj: Any) -> None:
        """Process completed LLM invocations and workflows.

        For LLMInvocation: records token usage and runs predictions.
        For Workflow: marks trace complete and updates workflow patterns.
        """
        try:
            if isinstance(obj, LLMInvocation):
                self._handle_llm_end(obj)
            elif isinstance(obj, Workflow):
                self._handle_workflow_end(obj)
        except Exception:
            _logger.debug(
                "RateLimitPredictorEmitter.on_end failed",
                exc_info=True,
            )

    def on_error(self, error: Any, obj: Any) -> None:
        """No-op on error; errors don't affect rate limit tracking."""
        return None

    def on_evaluation_results(
        self,
        results: Sequence[EvaluationResult],
        obj: Any | None = None,
    ) -> None:
        """No-op; evaluations don't affect rate limit tracking."""
        return None

    def _handle_llm_end(self, obj: LLMInvocation) -> None:
        """Record token usage and run predictions for an LLM invocation."""
        provider = obj.provider or "unknown"
        model = obj.request_model or "unknown"
        input_tokens = int(obj.input_tokens or 0)
        output_tokens = int(obj.output_tokens or 0)

        if input_tokens == 0 and output_tokens == 0:
            return

        trace_id = self._format_trace_id(obj.trace_id)
        span_id = self._format_span_id(obj.span_id)

        # Get workflow name from attributes or parent context
        workflow_name = (
            obj.attributes.get("gen_ai.workflow.name")
            if obj.attributes
            else None
        )

        self._tracker.record(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            trace_id=trace_id,
            span_id=span_id,
            workflow_name=workflow_name,
        )

        # Run predictions
        predictions = self._predictor.predict_all(
            provider=provider, model=model
        )

        # Emit warnings for predictions above threshold
        for pred in predictions:
            if pred.is_warning(self._warning_threshold):
                self._emit_rate_limit_warning(pred, provider, model, trace_id)

    def _handle_workflow_end(self, obj: Workflow) -> None:
        """Mark trace complete and update workflow patterns."""
        trace_id = self._format_trace_id(obj.trace_id)
        if trace_id is None:
            return

        self._tracker.mark_trace_complete(trace_id)

        # Update workflow pattern from the completed trace
        trace_usage = self._tracker.get_trace_usage(trace_id)
        if trace_usage and trace_usage["total_tokens"] > 0:
            self._tracker.update_workflow_pattern(
                workflow_name=obj.name,
                provider=trace_usage["provider"],
                model=trace_usage["model"],
                total_tokens=trace_usage["total_tokens"],
            )

    def _emit_rate_limit_warning(
        self,
        pred: Any,
        provider: str,
        model: str,
        trace_id: str | None,
    ) -> None:
        """Emit a rate limit warning as an OpenTelemetry log event."""
        if self._event_logger is None:
            return

        try:
            from opentelemetry._logs import LogRecord

            attributes: dict[str, Any] = {
                "event.name": _EVENT_NAME_RATE_LIMIT,
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model,
                "rate_limit.type": pred.limit_type,
                "rate_limit.current_usage": pred.current_usage,
                "rate_limit.limit": pred.limit,
                "rate_limit.utilization_percent": round(
                    pred.utilization * 100, 1
                ),
                "rate_limit.will_breach": pred.will_breach,
                "rate_limit.recommendation": pred.recommendation,
            }
            if trace_id:
                attributes["trace_id"] = trace_id

            record = LogRecord(
                body={"event.name": _EVENT_NAME_RATE_LIMIT},
                attributes=attributes,
            )
            self._event_logger.emit(record)
        except Exception:
            _logger.debug(
                "Failed to emit rate limit warning event", exc_info=True
            )

    @staticmethod
    def _format_trace_id(trace_id: int | None) -> str | None:
        """Format trace_id int as 32-char hex string."""
        if trace_id is None or trace_id == 0:
            return None
        return f"{trace_id:032x}"

    @staticmethod
    def _format_span_id(span_id: int | None) -> str | None:
        """Format span_id int as 16-char hex string."""
        if span_id is None or span_id == 0:
            return None
        return f"{span_id:016x}"


def load_emitters() -> list[EmitterSpec]:
    """Entry point loader for the rate limit predictor emitter.

    Returns a list of EmitterSpec for registration with the GenAI emitter pipeline.

    Usage:
        OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"
    """

    def _factory(ctx: Any) -> RateLimitPredictorEmitter:
        return RateLimitPredictorEmitter(
            event_logger=getattr(ctx, "content_logger", None),
        )

    return [
        EmitterSpec(
            name="RateLimitPredictor",
            category="content_events",
            factory=_factory,
        ),
    ]
