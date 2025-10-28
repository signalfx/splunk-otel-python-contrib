from __future__ import annotations

from opentelemetry.util.genai.config import Settings
from opentelemetry.util.genai.emitters.configuration import (
    build_emitter_pipeline,
)


class _MeterProvider:
    def get_meter(self, name):  # type: ignore[no-untyped-def]
        from opentelemetry import metrics

        return metrics.get_meter(name)


def test_replace_category_only_splunk_auto_augmented(monkeypatch):  # type: ignore[no-untyped-def]
    # Simulate environment where SplunkEvaluationResults spec is registered.
    # We'll monkeypatch load_emitter_specs to return a Splunk spec imitation.
    from opentelemetry.util.genai import emitters as emitters_pkg
    from opentelemetry.util.genai.emitters.spec import EmitterSpec

    class DummySplunkEval:
        role = "evaluation_results"
        name = "splunk_evaluation_results"

        def handles(self, obj):  # type: ignore[no-untyped-def]
            return False

    def _factory(ctx):  # type: ignore[no-untyped-def]
        return DummySplunkEval()

    splunk_spec = EmitterSpec(
        name="SplunkEvaluationResults",
        category="evaluation",
        factory=_factory,
    )

    def _fake_load(names):  # type: ignore[no-untyped-def]
        return [splunk_spec] if "splunk" in names else []

    monkeypatch.setattr(
        emitters_pkg.configuration,  # type: ignore[attr-defined]
        "load_emitter_specs",
        _fake_load,
        raising=True,
    )

    settings = Settings(
        enable_span=False,
        enable_metrics=False,
        enable_content_events=False,
        extra_emitters=["splunk"],
        only_traceloop_compat=False,
        raw_tokens=["splunk"],
        capture_messages_mode=None,  # type: ignore[arg-type]
        capture_messages_override=False,
        legacy_capture_request=False,
        emit_legacy_evaluation_event=False,
        category_overrides={
            "evaluation": emitters_pkg.spec.CategoryOverride(
                mode="replace-category",
                emitter_names=("SplunkEvaluationResults",),
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
    # Auto augmentation should ensure EvaluationMetrics and EvaluationEvents restored
    assert "EvaluationMetrics" in eval_names
    assert "EvaluationEvents" in eval_names
    # Instance name is 'splunk_evaluation_results' (class attribute), not spec name.
    assert "splunk_evaluation_results" in eval_names
