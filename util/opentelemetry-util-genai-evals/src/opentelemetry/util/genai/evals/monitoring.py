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

"""Evaluator monitoring instruments for GenAI evaluations.

This module owns the evaluator-side metrics instruments used to monitor the
evaluation pipeline (queue/backpressure) and LLM-as-a-judge client activity.

Metric names are evaluation-prefixed variants intended to mirror GenAI client
metric semantics from OpenTelemetry semantic conventions.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Mapping

from opentelemetry.metrics import get_meter
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.util.genai.attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    SERVER_ADDRESS,
    SERVER_PORT,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING,
)

from .version import __version__

GEN_AI_TOKEN_TYPE = "gen_ai.token.type"  # semconv attribute for token metrics

EVAL_CLIENT_OPERATION_DURATION = "gen_ai.evaluation.client.operation.duration"
EVAL_CLIENT_TOKEN_USAGE = "gen_ai.evaluation.client.token.usage"
EVAL_CLIENT_QUEUE_SIZE = "gen_ai.evaluation.client.queue.size"
EVAL_CLIENT_ENQUEUE_ERRORS = "gen_ai.evaluation.client.enqueue.errors"

_METER_NAME = "opentelemetry.util.genai.evals"

_LOCK = threading.Lock()
_DEFAULT_INSTRUMENTS: "EvaluationMonitoringInstruments | None" = None
_INSTRUMENTS_BY_PROVIDER: weakref.WeakKeyDictionary[
    object, "EvaluationMonitoringInstruments"
] = weakref.WeakKeyDictionary()

_TRUTHY = {"1", "true", "yes", "on"}


def monitoring_enabled() -> bool:
    value = os.getenv(OTEL_INSTRUMENTATION_GENAI_EVALS_MONITORING)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


class _NoopInstrument:
    def add(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass(frozen=True)
class EvaluationMonitoringInstruments:
    client_operation_duration: Any
    client_token_usage: Any
    queue_size: Any
    enqueue_errors: Any


_NOOP_INSTRUMENTS = EvaluationMonitoringInstruments(
    client_operation_duration=_NoopInstrument(),
    client_token_usage=_NoopInstrument(),
    queue_size=_NoopInstrument(),
    enqueue_errors=_NoopInstrument(),
)


def get_instruments(
    meter_provider: Any | None = None,
) -> EvaluationMonitoringInstruments:
    if not monitoring_enabled():
        return _NOOP_INSTRUMENTS

    global _DEFAULT_INSTRUMENTS

    if meter_provider is None:
        if _DEFAULT_INSTRUMENTS is not None:
            return _DEFAULT_INSTRUMENTS
        with _LOCK:
            if _DEFAULT_INSTRUMENTS is not None:
                return _DEFAULT_INSTRUMENTS
            _DEFAULT_INSTRUMENTS = _create_instruments(meter_provider=None)
            return _DEFAULT_INSTRUMENTS

    try:
        existing = _INSTRUMENTS_BY_PROVIDER.get(meter_provider)
        if existing is not None:
            return existing
        with _LOCK:
            existing = _INSTRUMENTS_BY_PROVIDER.get(meter_provider)
            if existing is not None:
                return existing
            instruments = _create_instruments(meter_provider=meter_provider)
            _INSTRUMENTS_BY_PROVIDER[meter_provider] = instruments
            return instruments
    except TypeError:
        # Meter provider doesn't support weak references; fall back to an
        # id-based cache for this provider instance.
        fallback_key = id(meter_provider)
        with _LOCK:
            existing_fallback = getattr(
                get_instruments, "_fallback_cache", {}
            ).get(fallback_key)
            if existing_fallback is not None:
                return existing_fallback
            instruments = _create_instruments(meter_provider=meter_provider)
            cache = getattr(get_instruments, "_fallback_cache", {})
            cache[fallback_key] = instruments
            setattr(get_instruments, "_fallback_cache", cache)
            return instruments


def _create_instruments(
    *, meter_provider: Any | None
) -> EvaluationMonitoringInstruments:
    meter = get_meter(
        _METER_NAME,
        __version__,
        meter_provider=meter_provider,
        schema_url=Schemas.V1_37_0.value,
    )
    return EvaluationMonitoringInstruments(
        client_operation_duration=meter.create_histogram(
            name=EVAL_CLIENT_OPERATION_DURATION,
            unit="s",
            description="Duration of evaluation calls",
        ),
        client_token_usage=meter.create_histogram(
            name=EVAL_CLIENT_TOKEN_USAGE,
            unit="{token}",
            description="Token usage in evaluation calls",
        ),
        queue_size=meter.create_up_down_counter(
            name=EVAL_CLIENT_QUEUE_SIZE,
            unit="1",
            description="Current evaluation queue size",
        ),
        enqueue_errors=meter.create_counter(
            name=EVAL_CLIENT_ENQUEUE_ERRORS,
            unit="1",
            description="Number of sampled spans that failed to enqueue for evaluation",
        ),
    )


def _build_client_metric_attributes(
    *,
    operation_name: str,
    provider_name: str,
    request_model: str | None = None,
    response_model: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    error_type: str | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: operation_name,
        GEN_AI_PROVIDER_NAME: provider_name,
    }
    if request_model:
        attrs[GEN_AI_REQUEST_MODEL] = request_model
    if response_model:
        attrs[GEN_AI_RESPONSE_MODEL] = response_model
    if server_address:
        attrs[SERVER_ADDRESS] = server_address
    if server_port is not None:
        attrs[SERVER_PORT] = server_port
    if error_type:
        attrs[ErrorAttributes.ERROR_TYPE] = error_type
    if extra_attributes:
        try:
            attrs.update(dict(extra_attributes))
        except Exception:
            pass
    return attrs


def record_client_operation_duration(
    duration_seconds: float,
    *,
    meter_provider: Any | None = None,
    operation_name: str,
    provider_name: str,
    request_model: str | None = None,
    response_model: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    error_type: str | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    if not monitoring_enabled():
        return
    if (
        not isinstance(duration_seconds, (int, float))
        or duration_seconds < 0
        or not operation_name
        or not provider_name
    ):
        return
    instruments = get_instruments(meter_provider)
    attrs = _build_client_metric_attributes(
        operation_name=operation_name,
        provider_name=provider_name,
        request_model=request_model,
        response_model=response_model,
        server_address=server_address,
        server_port=server_port,
        error_type=error_type,
        extra_attributes=extra_attributes,
    )
    try:
        instruments.client_operation_duration.record(duration_seconds, attrs)
    except Exception:
        return


def record_client_token_usage(
    tokens: int,
    *,
    meter_provider: Any | None = None,
    token_type: str,
    operation_name: str,
    provider_name: str,
    request_model: str | None = None,
    response_model: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    if not monitoring_enabled():
        return
    if (
        not isinstance(tokens, int)
        or tokens < 0
        or token_type not in {"input", "output"}
        or not operation_name
        or not provider_name
    ):
        return
    instruments = get_instruments(meter_provider)
    attrs = _build_client_metric_attributes(
        operation_name=operation_name,
        provider_name=provider_name,
        request_model=request_model,
        response_model=response_model,
        server_address=server_address,
        server_port=server_port,
        error_type=None,
        extra_attributes=extra_attributes,
    )
    attrs[GEN_AI_TOKEN_TYPE] = token_type
    try:
        instruments.client_token_usage.record(tokens, attrs)
    except Exception:
        return


def time_client_operation(
    *,
    meter_provider: Any | None = None,
    operation_name: str,
    provider_name: str,
    request_model: str | None = None,
    response_model: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> tuple[float, Any]:
    """Return (start_time, finish_fn) for manual timing without context managers.

    This is a small helper to keep evaluator integrations dependency-light and
    avoid forcing a particular exception handling strategy.
    """

    start = time.monotonic()
    if not monitoring_enabled():
        return start, (lambda _error_type=None: None)

    def _finish(error_type: str | None = None) -> None:
        duration = time.monotonic() - start
        record_client_operation_duration(
            duration,
            meter_provider=meter_provider,
            operation_name=operation_name,
            provider_name=provider_name,
            request_model=request_model,
            response_model=response_model,
            server_address=server_address,
            server_port=server_port,
            error_type=error_type,
            extra_attributes=extra_attributes,
        )

    return start, _finish


__all__ = [
    "EVAL_CLIENT_OPERATION_DURATION",
    "EVAL_CLIENT_TOKEN_USAGE",
    "EVAL_CLIENT_QUEUE_SIZE",
    "EVAL_CLIENT_ENQUEUE_ERRORS",
    "GEN_AI_TOKEN_TYPE",
    "EvaluationMonitoringInstruments",
    "get_instruments",
    "record_client_operation_duration",
    "record_client_token_usage",
    "time_client_operation",
]
