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

from opentelemetry import trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
except (ImportError, ModuleNotFoundError):
    OTLPSpanExporter = None

DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_PROMPT = (
    "Explain in two sentences how OpenTelemetry helps debug agentic AI."
)


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
        )
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        help="Bedrock Runtime model ID.",
    )
    parser.add_argument(
        "--region",
        default=(
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-west-2"
        ),
        help="AWS region for the Bedrock Runtime client.",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("BEDROCK_PROMPT", DEFAULT_PROMPT),
        help="Prompt sent to Bedrock Runtime.",
    )
    parser.add_argument(
        "--agentcore",
        action="store_true",
        default=_env_flag("BEDROCK_EXAMPLE_ENABLE_AGENTCORE"),
        help=(
            "Also enable Bedrock AgentCore instrumentation and run the "
            "Bedrock call from an AgentCore entrypoint."
        ),
    )
    parser.add_argument(
        "--serve-agentcore",
        action="store_true",
        default=_env_flag("BEDROCK_EXAMPLE_SERVE_AGENTCORE"),
        help=(
            "Start the AgentCore app server instead of invoking the "
            "entrypoint locally. Requires --agentcore."
        ),
    )
    parser.add_argument(
        "--exporter",
        choices=("console", "otlp"),
        default=os.getenv("BEDROCK_EXAMPLE_EXPORTER", "console"),
        help="Trace exporter. Console is useful for local parent/child checks.",
    )
    return parser.parse_args()


def _configure_tracing(exporter: str) -> TracerProvider:
    resource = Resource.create(
        {
            "service.name": os.getenv(
                "OTEL_SERVICE_NAME",
                "bedrock-runtime-agentcore-example",
            )
        }
    )
    provider = TracerProvider(resource=resource)

    if exporter == "otlp":
        if OTLPSpanExporter is None:
            raise SystemExit(
                "OTLP export requires opentelemetry-exporter-otlp-proto-grpc."
            )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def _instrument(enable_agentcore: bool) -> None:
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
                "without --agentcore."
            ) from exc

        BedrockAgentCoreInstrumentor().instrument()

    BedrockInstrumentor().instrument()


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
            "Install the SDK, or run without --agentcore."
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


def main() -> None:
    args = _parse_args()
    if args.serve_agentcore and not args.agentcore:
        raise SystemExit("--serve-agentcore requires --agentcore.")

    provider = _configure_tracing(args.exporter)
    _instrument(args.agentcore)

    client = _bedrock_client(args.region)
    try:
        if args.agentcore:
            app, entrypoint = _build_agentcore_app(
                client, args.model_id, args.prompt
            )
            if args.serve_agentcore:
                app.run()
            else:
                result = _resolve_result(entrypoint({"prompt": args.prompt}))
                if result is not None:
                    print(f"AgentCore result: {result}")
        else:
            _run_bedrock_turn(client, args.model_id, args.prompt)
    finally:
        provider.force_flush()
        provider.shutdown()


if __name__ == "__main__":
    main()
