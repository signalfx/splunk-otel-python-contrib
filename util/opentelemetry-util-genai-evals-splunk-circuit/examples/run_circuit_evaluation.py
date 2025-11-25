#!/usr/bin/env python3
# pyright: ignore
"""Run a Deepeval assessment using Cisco CircuIT as the judge model."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Callable, Iterable, cast


def _load_dependency(module: str, attr: str | None = None):
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "This example requires splunk-otel-genai-evals-deepeval and the CircuIT package on PYTHONPATH."
        ) from exc
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise SystemExit(
            f"Module '{module}' is missing expected attribute '{attr}'."
        ) from exc


create_circuit_llm = cast(
    Callable[[], Any],
    _load_dependency(
        "opentelemetry.util.evaluator.circuit_deepeval", "create_circuit_llm"
    ),
)
DeepevalEvaluator = cast(
    Any,
    _load_dependency(
        "opentelemetry.util.evaluator.deepeval", "DeepevalEvaluator"
    ),
)
register_model = cast(
    Callable[[str, Callable[[], Any]], None],
    _load_dependency(
        "opentelemetry.util.evaluator.deepeval_models", "register_model"
    ),
)
_types_mod = _load_dependency("opentelemetry.util.genai.types")
InputMessage = getattr(_types_mod, "InputMessage")
LLMInvocation = getattr(_types_mod, "LLMInvocation")
OutputMessage = getattr(_types_mod, "OutputMessage")
Text = getattr(_types_mod, "Text")


def _ensure_circuit_registration(model_alias: str) -> None:
    """Register CircuIT aliases and set the active Deepeval model."""

    previous = os.environ.get("DEEPEVAL_MODEL")
    os.environ["DEEPEVAL_MODEL"] = model_alias
    if previous != model_alias:
        print(f"Set DEEPEVAL_MODEL to '{model_alias}' (was {previous!r})")
    else:
        print(f"Using existing DEEPEVAL_MODEL='{model_alias}'")

    for alias in ("splunk-circuit", "circuit"):
        register_model(alias, create_circuit_llm)
    print("Registered CircuIT model aliases: splunk-circuit, circuit")


def _build_invocation(prompt: str, response: str, model_name: str):
    invocation = LLMInvocation(request_model=model_name)
    invocation.input_messages.append(
        InputMessage(role="user", parts=[Text(content=prompt)])
    )
    invocation.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content=response)],
            finish_reason="stop",
        )
    )
    return invocation


def _format_metrics(
    results: Iterable[Any], show_details: bool = False
) -> Iterable[str]:
    for result in results:
        label = result.label or "-"
        score = "-" if result.score is None else f"{result.score:.3f}"
        status = result.attributes.get("gen_ai.evaluation.passed")
        status_text = "?" if status is None else ("pass" if status else "fail")
        line = f"{result.metric_name:20s} score={score:>6s} label={label:>10s} status={status_text}"
        if show_details:
            explanation = getattr(result, "explanation", None)
            error = getattr(result, "error", None)
            if explanation:
                line += f" explanation={explanation!r}"
            if error:
                line += f" error={getattr(error, 'message', error)!r}"
        yield line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="Summarize the assistant reply.",
        help="User prompt text supplied to the LLM under evaluation.",
    )
    parser.add_argument(
        "--response",
        default="The assistant responded with this placeholder answer.",
        help="Assistant response text to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional list of Deepeval metrics to run (defaults to plugin defaults).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Deepeval model alias to use (default: splunk-circuit).",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Include Deepeval error/explanation details in the output table.",
    )
    args = parser.parse_args()

    desired_model = (
        args.model or os.getenv("DEEPEVAL_MODEL") or "splunk-circuit"
    )
    # Remove conflicting overrides so our CircuIT model is respected.
    if args.model:
        os.environ.pop("DEEPEVAL_EVALUATION_MODEL", None)
    if os.getenv("OPENAI_MODEL") and desired_model in {
        "splunk-circuit",
        "circuit",
    }:
        print(
            "OPENAI_MODEL is set; ignoring it in favour of CircuIT model alias."
        )
        os.environ.pop("OPENAI_MODEL", None)

    _ensure_circuit_registration(desired_model)

    model_name = os.getenv("DEEPEVAL_MODEL", desired_model)
    invocation = _build_invocation(args.prompt, args.response, model_name)

    evaluator = DeepevalEvaluator(
        tuple(args.metrics) if args.metrics else None,
        invocation_type="LLMInvocation",
    )
    resolved_model = None
    try:
        resolved_model = evaluator._default_model()
    except Exception:
        pass
    if resolved_model is not None:
        resolved_desc = getattr(
            resolved_model, "get_model_name", lambda: repr(resolved_model)
        )()
        print(
            f"Resolved Deepeval judge: {resolved_desc} ({resolved_model.__class__.__name__})"
        )
    print(
        f"Invoking Deepeval with model '{model_name}' and metrics {evaluator.metrics}"
    )
    try:
        results = list(evaluator.evaluate(invocation))
    except Exception as exc:
        print(
            f"Deepeval evaluation raised an exception: {exc}", file=sys.stderr
        )
        return 2

    if not results:
        print(
            "No evaluation results were produced. Check your environment settings.",
            file=sys.stderr,
        )
        return 1

    print("Deepeval results via Cisco CircuIT:")
    for line in _format_metrics(results, show_details=args.show_errors):
        print("  " + line)
    error_count = sum(1 for item in results if getattr(item, "error", None))
    if error_count:
        print(
            f"Encountered errors for {error_count} metric(s); inspect the logs above for details."
        )
    print("Evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
