# Copyright Splunk Inc.
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

"""Bedrock Runtime example with optional AgentCore composition."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import os
from typing import Any, Callable

from opentelemetry import _logs as logs
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

try:
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
except (ImportError, ModuleNotFoundError):
    OTLPLogExporter = None
    OTLPMetricExporter = None
    OTLPSpanExporter = None

DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_PROMPT = (
    "Explain in two sentences how OpenTelemetry helps debug agentic AI."
)
DEFAULT_REGION = "us-west-2"
DEFAULT_EXPORTER = "otlp"
DEFAULT_SERVICE_NAME = "bedrock-runtime-agentcore-example"
DEFAULT_CAPTURE_CONTENT = "true"
DEFAULT_CAPTURE_CONTENT_MODE = "SPAN_AND_EVENT"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4317"
DEFAULT_EMITTERS = "span_metric_event"
DEFAULT_ENABLE_AGENTCORE = "false"
DEFAULT_SERVE_AGENTCORE = "false"
DEFAULT_EVAL_WAIT_SECONDS = "60"
ENVIRONMENT_HELP = f"""Environment variables:
  AWS_REGION / AWS_DEFAULT_REGION (default: {DEFAULT_REGION})
  BEDROCK_MODEL_ID (default: {DEFAULT_MODEL_ID})
  BEDROCK_PROMPT (default: {DEFAULT_PROMPT})
  BEDROCK_EXAMPLE_EXPORTER=otlp|console (default: {DEFAULT_EXPORTER})
  BEDROCK_EXAMPLE_ENABLE_AGENTCORE=true|false (default: {DEFAULT_ENABLE_AGENTCORE})
  BEDROCK_EXAMPLE_SERVE_AGENTCORE=true|false (default: {DEFAULT_SERVE_AGENTCORE})
  BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS (default: {DEFAULT_EVAL_WAIT_SECONDS})
  OTEL_SERVICE_NAME (default: {DEFAULT_SERVICE_NAME})
  OTEL_EXPORTER_OTLP_ENDPOINT (default: {DEFAULT_OTLP_ENDPOINT})
  OTEL_INSTRUMENTATION_GENAI_EMITTERS (default: {DEFAULT_EMITTERS})
  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT (default: {DEFAULT_CAPTURE_CONTENT})
  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE (default: {DEFAULT_CAPTURE_CONTENT_MODE})
  OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS (optional, for evals)
"""


def _set_default_environment() -> None:
    defaults = {
        "BEDROCK_MODEL_ID": DEFAULT_MODEL_ID,
        "BEDROCK_PROMPT": DEFAULT_PROMPT,
        "BEDROCK_EXAMPLE_EXPORTER": DEFAULT_EXPORTER,
        "BEDROCK_EXAMPLE_ENABLE_AGENTCORE": DEFAULT_ENABLE_AGENTCORE,
        "BEDROCK_EXAMPLE_SERVE_AGENTCORE": DEFAULT_SERVE_AGENTCORE,
        "BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS": DEFAULT_EVAL_WAIT_SECONDS,
        "OTEL_SERVICE_NAME": DEFAULT_SERVICE_NAME,
        "OTEL_EXPORTER_OTLP_ENDPOINT": DEFAULT_OTLP_ENDPOINT,
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS": DEFAULT_EMITTERS,
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": DEFAULT_CAPTURE_CONTENT,
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE": DEFAULT_CAPTURE_CONTENT_MODE,
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    if (
        "AWS_REGION" not in os.environ
        and "AWS_DEFAULT_REGION" not in os.environ
    ):
        os.environ["AWS_REGION"] = DEFAULT_REGION


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Bedrock Runtime Converse call with optional AgentCore "
            "instrumentation composition."
        ),
        epilog=ENVIRONMENT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--with-agentcore",
        action="store_true",
        help=(
            "Also enable Bedrock AgentCore instrumentation and run the "
            "Bedrock call from an AgentCore entrypoint."
        ),
    )
    return parser.parse_args()


def _configure_telemetry(
    exporter: str,
) -> tuple[TracerProvider, MeterProvider, LoggerProvider]:
    resource = Resource.create(
        {
            "service.name": os.getenv(
                "OTEL_SERVICE_NAME",
                DEFAULT_SERVICE_NAME,
            )
        }
    )
    provider = TracerProvider(resource=resource)

    if exporter == "otlp":
        if (
            OTLPSpanExporter is None
            or OTLPMetricExporter is None
            or OTLPLogExporter is None
        ):
            raise SystemExit(
                "OTLP export requires opentelemetry-exporter-otlp-proto-grpc."
            )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=5000,
        )
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(OTLPLogExporter())
        )
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=5000,
        )
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(ConsoleLogExporter())
        )

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
    )
    trace.set_tracer_provider(provider)
    metrics.set_meter_provider(meter_provider)
    logs.set_logger_provider(logger_provider)
    return provider, meter_provider, logger_provider


def _instrument(
    enable_agentcore: bool,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
) -> None:
    instrumentor_kwargs = {
        "tracer_provider": tracer_provider,
        "meter_provider": meter_provider,
        "logger_provider": logger_provider,
    }
    if enable_agentcore:
        try:
            from opentelemetry.instrumentation.bedrock_agentcore import (
                BedrockAgentCoreInstrumentor,
            )
        except ImportError as exc:
            raise SystemExit(
                "AgentCore mode requires "
                "opentelemetry.instrumentation.bedrock_agentcore. "
                "Install the AgentCore instrumentation package, or run "
                "without --with-agentcore."
            ) from exc

        BedrockAgentCoreInstrumentor().instrument(**instrumentor_kwargs)

    BedrockInstrumentor().instrument(**instrumentor_kwargs)


def _bedrock_client(region: str) -> Any:
    try:
        boto3 = importlib.import_module("boto3")
    except ImportError as exc:
        raise SystemExit(
            "This example requires boto3. Install requirements.txt before "
            "running Bedrock calls."
        ) from exc

    return boto3.client("bedrock-runtime", region_name=region)


def _call_converse(client: Any, model_id: str, prompt: str) -> dict[str, Any]:
    return client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={
            "maxTokens": 256,
            "temperature": 0.2,
            "topP": 0.9,
        },
    )


def _extract_text(response: dict[str, Any]) -> str:
    message = (response.get("output") or {}).get("message") or {}
    content = message.get("content") or []
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("text")
    ]
    return "".join(text_parts)


def _run_bedrock_turn(
    client: Any,
    model_id: str,
    prompt: str,
) -> dict[str, Any]:
    response = _call_converse(client, model_id, prompt)
    answer = _extract_text(response)
    print(answer)
    return {
        "model_id": model_id,
        "answer": answer,
        "request_id": (response.get("ResponseMetadata") or {}).get(
            "RequestId"
        ),
    }


def _prompt_from_payload(payload: Any, default_prompt: str) -> str:
    if isinstance(payload, dict):
        value = payload.get("prompt")
        if value:
            return str(value)
    if payload:
        return str(payload)
    return default_prompt


def _build_agentcore_app(
    client: Any,
    model_id: str,
    prompt: str,
) -> tuple[Any, Callable[[dict[str, Any]], Any]]:
    try:
        from bedrock_agentcore.runtime import BedrockAgentCoreApp
    except ImportError as exc:
        raise SystemExit(
            "AgentCore mode requires the Bedrock AgentCore SDK "
            "providing bedrock_agentcore.runtime.BedrockAgentCoreApp. "
            "Install the SDK, or run without --with-agentcore."
        ) from exc

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def bedrock_runtime_agent(payload: dict[str, Any]) -> dict[str, Any]:
        user_prompt = _prompt_from_payload(payload, prompt)
        return _run_bedrock_turn(client, model_id, user_prompt)

    return app, bedrock_runtime_agent


def _resolve_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _configured_exporter() -> str:
    exporter = os.environ["BEDROCK_EXAMPLE_EXPORTER"].strip().lower()
    if exporter not in {"otlp", "console"}:
        raise SystemExit(
            "BEDROCK_EXAMPLE_EXPORTER must be either 'otlp' or 'console'."
        )
    return exporter


def _configured_region() -> str:
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )


def _configured_eval_wait_seconds() -> float:
    value = os.environ["BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS"]
    try:
        return max(0.0, float(value))
    except ValueError as exc:
        raise SystemExit(
            "BEDROCK_EXAMPLE_EVAL_WAIT_SECONDS must be a number."
        ) from exc


def _wait_for_evaluations() -> None:
    timeout = _configured_eval_wait_seconds()
    if timeout <= 0:
        return
    try:
        from opentelemetry.util.genai.handler import get_telemetry_handler
    except ImportError:
        return

    get_telemetry_handler().wait_for_evaluations(timeout)


def main() -> None:
    _set_default_environment()
    args = _parse_args()
    enable_agentcore = args.with_agentcore or _env_flag(
        "BEDROCK_EXAMPLE_ENABLE_AGENTCORE"
    )
    serve_agentcore = _env_flag("BEDROCK_EXAMPLE_SERVE_AGENTCORE")
    if serve_agentcore and not enable_agentcore:
        raise SystemExit(
            "BEDROCK_EXAMPLE_SERVE_AGENTCORE=true requires "
            "--with-agentcore or BEDROCK_EXAMPLE_ENABLE_AGENTCORE=true."
        )

    tracer_provider, meter_provider, logger_provider = _configure_telemetry(
        _configured_exporter()
    )
    _instrument(
        enable_agentcore,
        tracer_provider,
        meter_provider,
        logger_provider,
    )

    model_id = os.environ["BEDROCK_MODEL_ID"]
    prompt = os.environ["BEDROCK_PROMPT"]
    client = _bedrock_client(_configured_region())
    try:
        if enable_agentcore:
            app, entrypoint = _build_agentcore_app(client, model_id, prompt)
            if serve_agentcore:
                app.run()
            else:
                result = _resolve_result(entrypoint({"prompt": prompt}))
                if result is not None:
                    print(f"AgentCore result: {result}")
        else:
            _run_bedrock_turn(client, model_id, prompt)
    finally:
        _wait_for_evaluations()
        tracer_provider.force_flush()
        meter_provider.force_flush()
        logger_provider.force_flush()
        tracer_provider.shutdown()
        meter_provider.shutdown()
        logger_provider.shutdown()


if __name__ == "__main__":
    main()
