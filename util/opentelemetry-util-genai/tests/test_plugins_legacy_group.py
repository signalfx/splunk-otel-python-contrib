from __future__ import annotations

from typing import Any, cast

from opentelemetry.util.genai.config import Settings
from opentelemetry.util.genai.emitters.configuration import (
    build_emitter_pipeline,
)
from opentelemetry.util.genai.emitters.spec import (
    CategoryOverride,
    EmitterSpec,
)


class _Dummy:  # minimal meter provider stub
    def get_meter(self, name):  # type: ignore[no-untyped-def]
        from opentelemetry import metrics

        return metrics.get_meter(name)


def _base_settings() -> Settings:
    return Settings(
        enable_span=True,
        enable_metrics=True,
        enable_content_events=True,
        extra_emitters=["splunk"],
        only_traceloop_compat=False,
        raw_tokens=["span_metric_event", "splunk"],
        capture_messages_mode=None,  # type: ignore[arg-type]
        capture_messages_override=False,
        legacy_capture_request=False,
        emit_legacy_evaluation_event=False,
        category_overrides={
            "evaluation": CategoryOverride(
                mode="replace-category",
                emitter_names=("SplunkEvaluationResults",),
            )
        },
    )


def test_legacy_entry_point_group_loading(monkeypatch):  # type: ignore[no-untyped-def]
    # Simulate legacy group by patching entry_points to return a 'splunk' ep in legacy group call
    from opentelemetry.util.genai import plugins as plugins_mod

    class _EP:  # simple entry point stub
        def __init__(self, name):  # type: ignore[no-untyped-def]
            self.name = name

        def load(self):  # type: ignore[no-untyped-def]
            # Return list of specs mimicking the real splunk_emitters output
            def _fake_factory(_ctx):  # type: ignore[no-untyped-def]
                class _FakeEmitter:
                    role = "evaluation_results"
                    name = "SplunkEvaluationResults"

                    def handles(self, obj):  # type: ignore[no-untyped-def]
                        return True

                    def on_start(self, obj):  # type: ignore[no-untyped-def]
                        return None

                    def on_end(self, obj):  # type: ignore[no-untyped-def]
                        return None

                    def on_error(self, error, obj):  # type: ignore[no-untyped-def]
                        return None

                    def on_evaluation_results(self, results, obj=None):  # type: ignore[no-untyped-def]
                        return None

                return cast(Any, _FakeEmitter())

            return [
                EmitterSpec(
                    name="SplunkEvaluationResults",
                    category="evaluation",
                    factory=_fake_factory,
                )
            ]

    def _fake_entry_points(group=None):  # type: ignore[no-untyped-def]
        if group == "opentelemetry_genai_emitters":
            return [_EP("splunk")]
        if group == "opentelemetry_util_genai_emitters":
            return []
        return []

    monkeypatch.setattr(plugins_mod, "entry_points", _fake_entry_points)

    settings = _base_settings()
    tracer = None
    meter = _Dummy().get_meter(__name__)
    event_logger = None
    content_logger = None

    composite, _ = build_emitter_pipeline(
        tracer=tracer,
        meter=meter,
        event_logger=event_logger,
        content_logger=content_logger,
        evaluation_histogram=lambda name: None,
        settings=settings,
    )

    eval_emitters = [
        getattr(e, "name", None) for e in composite.emitters_for("evaluation")
    ]
    assert "SplunkEvaluationResults" in eval_emitters
