"""Unit tests configuration module."""

import json
import os

import pytest

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    yaml = None

# from openai import AsyncOpenAI, OpenAI
try:  # pragma: no cover - optional dependency in CI
    from langchain_openai import ChatOpenAI
except (
    ModuleNotFoundError
):  # pragma: no cover - allow minimal test runs without package
    ChatOpenAI = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in CI
    from opentelemetry.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.instrumentation.langchain.utils import (
        set_prompt_capture_enabled,
    )
except (
    ModuleNotFoundError
):  # pragma: no cover - allow subset of tests without full instrumentation
    LangChainInstrumentor = None  # type: ignore[assignment]

    def set_prompt_capture_enabled(_value: bool) -> None:  # type: ignore[override]
        return None


from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import (
    MeterProvider,
)
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import get_telemetry_handler


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    exporter = InMemoryMetricReader()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="event_logger_provider")
def fixture_event_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    event_logger_provider = EventLoggerProvider(provider)

    return event_logger_provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
    )

    return meter_provider


@pytest.fixture(autouse=True)
def environment():
    original_api_key = os.environ.get("OPENAI_API_KEY")
    original_evals = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS")
    original_emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS")

    if not original_api_key:
        os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"
    setattr(get_telemetry_handler, "_default_handler", None)

    yield

    if original_api_key is None:
        os.environ.pop("OPENAI_API_KEY", None)
    else:
        os.environ["OPENAI_API_KEY"] = original_api_key

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = original_evals

    if original_emitters is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = original_emitters

    setattr(get_telemetry_handler, "_default_handler", None)


@pytest.fixture
def chatOpenAI_client():
    if ChatOpenAI is None:  # pragma: no cover - guard when dependency missing
        pytest.skip("langchain_openai is not available in this environment")
    return ChatOpenAI()


@pytest.fixture(scope="function")
def vcr_config():
    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
        "serializer": "yaml",
    }


@pytest.fixture(scope="session")
def vcr_cassette_dir():
    """Override the default cassette directory to avoid nested subdirectories."""
    import os

    # Return the cassettes directory path
    return os.path.join(os.path.dirname(__file__), "cassettes")


@pytest.fixture(scope="function")
def instrument_no_content(tracer_provider, event_logger_provider, meter_provider):
    if LangChainInstrumentor is None:  # pragma: no cover - skip when dependency missing
        pytest.skip("opentelemetry-instrumentation-langchain not available")
    set_prompt_capture_enabled(False)

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    set_prompt_capture_enabled(True)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, event_logger_provider, meter_provider):
    if LangChainInstrumentor is None:  # pragma: no cover
        pytest.skip("opentelemetry-instrumentation-langchain not available")
    set_prompt_capture_enabled(True)

    # Reset util-genai singleton handler to ensure clean state
    import opentelemetry.util.genai.handler as _util_handler_mod  # noqa: PLC0415

    if hasattr(_util_handler_mod.get_telemetry_handler, "_default_handler"):
        setattr(_util_handler_mod.get_telemetry_handler, "_default_handler", None)

    # Create new instrumentor for each test
    instrumentor = LangChainInstrumentor()

    # If already instrumented (from previous test), uninstrument first
    if instrumentor._is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    set_prompt_capture_enabled(True)
    # Clean up: uninstrument and reset singleton
    if instrumentor._is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    if hasattr(_util_handler_mod.get_telemetry_handler, "_default_handler"):
        setattr(_util_handler_mod.get_telemetry_handler, "_default_handler", None)


@pytest.fixture(scope="function")
def instrument_with_content_unsampled(
    span_exporter, event_logger_provider, meter_provider
):
    if LangChainInstrumentor is None:  # pragma: no cover
        pytest.skip("opentelemetry-instrumentation-langchain not available")
    set_prompt_capture_enabled(True)

    tracer_provider = TracerProvider(sampler=ALWAYS_OFF)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    set_prompt_capture_enabled(True)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content_util(
    tracer_provider, event_logger_provider, meter_provider
):
    if LangChainInstrumentor is None:  # pragma: no cover
        pytest.skip("opentelemetry-instrumentation-langchain not available")
    set_prompt_capture_enabled(True)
    os.environ.update(
        {
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",  # util-genai content gate
        }
    )

    # Reset singleton so new env vars are applied
    import opentelemetry.util.genai.handler as _util_handler_mod  # noqa: PLC0415

    if hasattr(_util_handler_mod.get_telemetry_handler, "_default_handler"):
        setattr(_util_handler_mod.get_telemetry_handler, "_default_handler", None)

    # Create new instrumentor for each test
    instrumentor = LangChainInstrumentor()

    # If already instrumented (from previous test), uninstrument first
    if instrumentor._is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    set_prompt_capture_enabled(True)

    # Clean up: uninstrument and reset singleton
    if instrumentor._is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    if hasattr(_util_handler_mod.get_telemetry_handler, "_default_handler"):
        setattr(_util_handler_mod.get_telemetry_handler, "_default_handler", None)


class LiteralBlockScalar(str):
    """Formats the string as a literal block scalar, preserving whitespace and
    without interpreting escape characters"""


def literal_block_scalar_presenter(dumper, data):
    """Represents a scalar string as a literal block, via '|' syntax"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


if yaml is not None:
    yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    """Pretty-prints JSON or returns long strings as a LiteralBlockScalar"""
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    """Searches the data for body strings, attempting to pretty-print JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle response body case (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])

            # Handle request body case (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)

            else:
                convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = convert_body_to_literal(choice)

    return data


class PrettyPrintJSONBody:
    """This makes request and response body recordings more readable."""

    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = convert_body_to_literal(cassette_dict)
        if yaml is None:
            return json.dumps(cassette_dict)
        return yaml.dump(cassette_dict, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def deserialize(cassette_string):
        if yaml is None:
            return json.loads(cassette_string)
        return yaml.load(cassette_string, Loader=yaml.Loader)


try:  # pragma: no cover - optional pytest-vcr dependency
    import pytest_recording  # type: ignore # noqa: F401
    import vcr as vcr_module  # type: ignore # noqa: F401

    # Register custom YAML serializer globally
    vcr_module.VCR().register_serializer("yaml", PrettyPrintJSONBody)

except ModuleNotFoundError:  # pragma: no cover - provide stub when plugin missing

    @pytest.fixture(name="vcr", scope="module")
    def _noop_vcr_fixture():
        class _VCRStub:
            def register_serializer(self, *_args, **_kwargs):
                return None

        return _VCRStub()


@pytest.fixture(scope="function", autouse=True)
def fixture_vcr(vcr):
    # When pytest-recording is installed, vcr is a Cassette and we don't need to do anything
    # The serializer is already registered on the VCR module above
    return vcr


def scrub_response_headers(response):
    """
    This scrubs sensitive response headers. Note they are case-sensitive!
    """
    response["headers"]["openai-organization"] = "test_openai_org_id"
    response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response
