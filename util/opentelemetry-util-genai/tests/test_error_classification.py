"""Tests for ErrorClassification enum and classification-aware error handling."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.attributes import (
    FINISH_REASON_CANCELLED,
    FINISH_REASON_FAILED,
    FINISH_REASON_INTERRUPTED,
    GEN_AI_COMMAND,
    GEN_AI_FINISH_REASON,
    GEN_AI_FINISH_REASON_DESCRIPTION,
    GEN_AI_STEP_STATUS,
)
from opentelemetry.util.genai.emitters.span import SpanEmitter
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Error,
    ErrorClassification,
    Step,
    Workflow,
)

# --- ErrorClassification enum unit tests ---


def test_error_classification_values():
    assert ErrorClassification.REAL_ERROR.value == "error"
    assert ErrorClassification.INTERRUPT.value == "interrupted"
    assert ErrorClassification.CANCELLATION.value == "cancelled"


def test_error_default_classification():
    err = Error(message="boom", type=RuntimeError)
    assert err.classification == ErrorClassification.REAL_ERROR


def test_error_explicit_classification():
    err = Error(
        message="interrupted",
        type=RuntimeError,
        classification=ErrorClassification.INTERRUPT,
    )
    assert err.classification == ErrorClassification.INTERRUPT


# --- Span emitter classification-aware tests ---


def _make_emitter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        __import__(
            "opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]
        ).SimpleSpanProcessor(exporter)
    )
    tracer = provider.get_tracer(__name__)
    emitter = SpanEmitter(tracer=tracer, capture_content=False)
    return emitter, exporter, provider


def test_error_workflow_real_error():
    emitter, exporter, provider = _make_emitter()
    wf = Workflow(name="test-wf")
    emitter.on_start(wf)
    err = Error(message="fail", type=ValueError)
    emitter.on_error(err, wf)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_error_workflow_interrupt():
    emitter, exporter, provider = _make_emitter()
    wf = Workflow(name="test-wf")
    emitter.on_start(wf)
    err = Error(
        message="interrupted",
        type=RuntimeError,
        classification=ErrorClassification.INTERRUPT,
    )
    emitter.on_error(err, wf)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON)
        == FINISH_REASON_INTERRUPTED
    )
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON_DESCRIPTION)
        == "interrupted"
    )


def test_error_workflow_cancellation():
    emitter, exporter, provider = _make_emitter()
    wf = Workflow(name="test-wf")
    emitter.on_start(wf)
    err = Error(
        message="cancelled",
        type=RuntimeError,
        classification=ErrorClassification.CANCELLATION,
    )
    emitter.on_error(err, wf)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON)
        == FINISH_REASON_CANCELLED
    )


def test_error_step_interrupt_status():
    emitter, exporter, provider = _make_emitter()
    step = Step(name="test-step", step_type="chain")
    emitter.on_start(step)
    err = Error(
        message="interrupted",
        type=RuntimeError,
        classification=ErrorClassification.INTERRUPT,
    )
    emitter.on_error(err, step)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert (
        spans[0].attributes.get(GEN_AI_STEP_STATUS)
        == FINISH_REASON_INTERRUPTED
    )
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON)
        == FINISH_REASON_INTERRUPTED
    )


def test_error_step_cancellation_status():
    emitter, exporter, provider = _make_emitter()
    step = Step(name="test-step", step_type="chain")
    emitter.on_start(step)
    err = Error(
        message="cancelled",
        type=RuntimeError,
        classification=ErrorClassification.CANCELLATION,
    )
    emitter.on_error(err, step)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert (
        spans[0].attributes.get(GEN_AI_STEP_STATUS) == FINISH_REASON_CANCELLED
    )
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON)
        == FINISH_REASON_CANCELLED
    )


def test_error_step_real_error_status():
    emitter, exporter, provider = _make_emitter()
    step = Step(name="test-step", step_type="chain")
    emitter.on_start(step)
    err = Error(message="fail", type=ValueError)
    emitter.on_error(err, step)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes.get(GEN_AI_STEP_STATUS) == FINISH_REASON_FAILED
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON) == FINISH_REASON_FAILED
    )
    assert spans[0].attributes.get(GEN_AI_FINISH_REASON_DESCRIPTION) == "fail"


def test_error_agent_interrupt():
    emitter, exporter, provider = _make_emitter()
    agent = AgentInvocation(name="test-agent")
    emitter.on_start(agent)
    err = Error(
        message="interrupted",
        type=RuntimeError,
        classification=ErrorClassification.INTERRUPT,
    )
    emitter.on_error(err, agent)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON)
        == FINISH_REASON_INTERRUPTED
    )


# --- Workflow command (resume) tests ---


def test_workflow_command_resume():
    emitter, exporter, provider = _make_emitter()
    wf = Workflow(name="test-wf")
    wf.attributes[GEN_AI_COMMAND] = "resume"
    emitter.on_start(wf)
    emitter.on_end(wf)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_COMMAND) == "resume"


def test_workflow_command_not_set_on_fresh():
    emitter, exporter, provider = _make_emitter()
    wf = Workflow(name="test-wf")
    emitter.on_start(wf)
    emitter.on_end(wf)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_COMMAND) is None


def test_error_real_error_sets_finish_reason_description():
    emitter, exporter, provider = _make_emitter()
    wf = Workflow(name="test-wf")
    emitter.on_start(wf)
    err = Error(message="connection timed out", type=TimeoutError)
    emitter.on_error(err, wf)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON) == FINISH_REASON_FAILED
    )
    assert (
        spans[0].attributes.get(GEN_AI_FINISH_REASON_DESCRIPTION)
        == "connection timed out"
    )
