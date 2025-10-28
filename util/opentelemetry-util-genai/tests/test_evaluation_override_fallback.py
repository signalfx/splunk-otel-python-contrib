from __future__ import annotations

from opentelemetry.util.genai.config import Settings
from opentelemetry.util.genai.emitters.configuration import (
    build_emitter_pipeline,
)
from opentelemetry.util.genai.emitters.spec import CategoryOverride


class _MeterProvider:
    def get_meter(self, name):  # type: ignore[no-untyped-def]
        from opentelemetry import metrics

        return metrics.get_meter(name)


def test_replace_category_empty_fallback():  # type: ignore[no-untyped-def]
    """If replace-category references only unknown emitters, built-ins must remain."""
    settings = Settings(
        enable_span=False,
        enable_metrics=False,
        enable_content_events=False,
        extra_emitters=[],
        only_traceloop_compat=False,
        raw_tokens=[],
        capture_messages_mode=None,  # type: ignore[arg-type]
        capture_messages_override=False,
        legacy_capture_request=False,
        emit_legacy_evaluation_event=False,
        category_overrides={
            "evaluation": CategoryOverride(
                mode="replace-category",
                emitter_names=("DoesNotExistOne", "DoesNotExistTwo"),
            )
        },
    )
    tracer = None
    meter = _MeterProvider().get_meter(__name__)
    composite, _ = build_emitter_pipeline(
        tracer=tracer,
        meter=meter,
        event_logger=None,
        content_logger=None,
        evaluation_histogram=lambda name: None,
        settings=settings,
    )
    eval_names = {
        getattr(e, "name", "") for e in composite.emitters_for("evaluation")
    }
    assert "EvaluationMetrics" in eval_names
    assert "EvaluationEvents" in eval_names
