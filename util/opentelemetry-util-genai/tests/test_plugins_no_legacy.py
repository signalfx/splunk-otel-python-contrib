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


def test_loader_ignores_legacy_group(monkeypatch):  # type: ignore[no-untyped-def]
    # Patch entry_points to supply only legacy group; expect no emitters loaded beyond built-ins.
    from opentelemetry.util.genai import plugins as plugins_mod

    class _EP:
        def __init__(self, name):  # type: ignore[no-untyped-def]
            self.name = name

        def load(self):  # type: ignore[no-untyped-def]
            return []  # would have returned specs if legacy group were still supported

    def _fake_entry_points(group=None):  # type: ignore[no-untyped-def]
        if group == "opentelemetry_util_genai_emitters":
            return []  # no modern group entries
        if group == "opentelemetry_genai_emitters":
            return [_EP("splunk")]  # legacy only
        return []

    monkeypatch.setattr(plugins_mod, "entry_points", _fake_entry_points)

    settings = Settings(
        enable_span=True,
        enable_metrics=True,
        enable_content_events=True,
        extra_emitters=["splunk"],  # request splunk explicitly
        only_traceloop_compat=False,
        raw_tokens=["span_metric_event", "splunk"],
        capture_messages_mode=None,  # type: ignore[arg-type]
        capture_messages_override=False,
        legacy_capture_request=False,
        emit_legacy_evaluation_event=False,
        category_overrides={
            # ensure evaluation override referencing legacy spec does not crash; will be ignored
            "evaluation": CategoryOverride(
                mode="append", emitter_names=("SplunkEvaluationResults",)
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

    # Built-in evaluation emitters should still be present, but no SplunkEvaluationResults
    eval_names = {
        getattr(e, "name", "") for e in composite.emitters_for("evaluation")
    }
    assert "EvaluationMetrics" in eval_names
    assert "EvaluationEvents" in eval_names
    assert "SplunkEvaluationResults" not in eval_names
