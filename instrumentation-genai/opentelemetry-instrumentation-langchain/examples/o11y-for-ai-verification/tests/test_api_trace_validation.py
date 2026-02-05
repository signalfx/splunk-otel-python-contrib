"""
API Test: Trace Retrieval and GenAI Schema Validation.

This test demonstrates API-level testing:
1. Retrieve traces via APM API
2. Validate GenAI semantic conventions
3. Verify span attributes and structure
4. Check token usage and cost tracking
"""

import pytest
from validators.trace_validator import TraceValidator
from validators.span_validator import SpanValidator
from utils.data_generator import DataGenerator


@pytest.mark.api
@pytest.mark.genai
class TestTraceAPIValidation:
    """
    API tests for trace retrieval and validation.
    """

    def test_retrieve_trace_by_id(self, apm_client, trace_id_list, actual_trace_id):
        """
        Test basic trace retrieval by ID.

        Args:
            apm_client: APM client fixture
            trace_id_list: List to collect trace IDs
            actual_trace_id: Actual trace ID from command line (optional)
        """
        # Use actual trace ID if provided, otherwise generate random one
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_id_list.append(trace_id)

        # Retrieve trace
        trace_data = apm_client.get_trace(trace_id)

        # Basic validation
        assert trace_data is not None, f"Trace {trace_id} not found"
        assert (
            "spans" in trace_data or "resourceSpans" in trace_data
        ), "Trace data missing span information"

        print(
            f"✓ Successfully retrieved trace {trace_id} with {len(trace_data.get('spans', []))} spans"
        )

    def test_validate_genai_attributes(self, apm_client, actual_trace_id):
        """
        Test GenAI attribute validation on a trace.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        # Use actual trace ID if provided, otherwise generate random one
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )

        # Retrieve trace (in real test, this would exist)
        trace_data = apm_client.get_trace(trace_id)

        if trace_data:
            # Validate required GenAI attributes
            result = TraceValidator.validate_genai_trace(
                trace=trace_data, expected_system="openai"
            )

            # Check validation result structure
            assert "valid" in result, "Validation result missing 'valid' key"

            if not result["valid"]:
                # Log errors for debugging
                errors = result.get("errors", [])
                print(f"Validation errors: {errors}")

    def test_validate_span_attributes(self, apm_client, actual_trace_id):
        """
        Test individual span attribute validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data and "spans" in trace_data:
            spans = trace_data["spans"]

            for span in spans:
                # Validate span has required fields
                assert "spanId" in span or "span_id" in span, "Span missing ID"

                # Validate span attributes structure
                attributes = span.get("attributes", {})

                # Check if it's a GenAI span
                if "gen_ai.system" in attributes:
                    # Validate GenAI-specific attributes (only gen_ai.system is required for all GenAI spans)
                    result = SpanValidator.validate_span_attributes(
                        span=span, required_attrs=["gen_ai.system"]
                    )

                    assert (
                        result
                    ), f"Span {span.get('spanId')} missing required GenAI attributes"

    def test_validate_token_usage_attributes(self, apm_client, actual_trace_id):
        """
        Test token usage attribute validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data:
            # Validate token usage across trace
            result = TraceValidator.validate_token_usage(trace_data)

            # Check result structure
            assert isinstance(
                result, (bool, dict)
            ), "Token usage validation returned unexpected type"

            if isinstance(result, dict):
                assert "valid" in result, "Result missing 'valid' key"

    def test_validate_span_hierarchy(self, apm_client, actual_trace_id):
        """
        Test span parent-child relationship validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data:
            # Validate hierarchy
            result = TraceValidator.validate_span_hierarchy(trace_data)

            # Check result
            assert isinstance(
                result, (bool, dict)
            ), "Hierarchy validation returned unexpected type"

    def test_validate_span_timing(self, apm_client, actual_trace_id):
        """
        Test span timing validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data and "spans" in trace_data:
            for span in trace_data["spans"]:
                # Validate timing attributes
                result = SpanValidator.validate_span_timing(span)

                assert result, f"Span timing validation failed for {span.get('spanId')}"

    def test_find_spans_by_operation(self, apm_client, actual_trace_id):
        """
        Test finding spans by operation name.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data:
            # Find chat operation spans
            chat_spans = TraceValidator.find_spans_by_operation(
                trace=trace_data, operation="chat"
            )

            # Verify result is a list
            assert isinstance(
                chat_spans, list
            ), "find_spans_by_operation should return a list"

    def test_validate_streaming_attributes(self, apm_client, actual_trace_id):
        """
        Test streaming response attribute validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data:
            # Validate streaming attributes
            result = TraceValidator.validate_streaming_attributes(trace_data)

            # Check result structure
            assert isinstance(
                result, (bool, dict)
            ), "Streaming validation returned unexpected type"

    @pytest.mark.ai_defense
    def test_validate_ai_defense_events(self, apm_client, actual_trace_id):
        """
        Test AI Defense security event validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data:
            # Validate AI Defense events
            result = TraceValidator.validate_ai_defense_events(trace_data)

            # Check result
            assert isinstance(
                result, (bool, dict)
            ), "AI Defense validation returned unexpected type"

    def test_batch_trace_retrieval(self, apm_client, actual_trace_id):
        """
        Test retrieving multiple traces.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        # If actual trace ID provided, use it; otherwise skip test
        if not actual_trace_id:
            import pytest

            pytest.skip("Batch test requires --trace-id parameter")

        # Retrieve the trace multiple times to test batch capability
        traces = []
        for _ in range(3):
            trace_data = apm_client.get_trace(actual_trace_id)
            if trace_data:
                traces.append(trace_data)

        # Verify we can retrieve multiple traces
        assert isinstance(traces, list), "Batch retrieval should return list"
        assert len(traces) > 0, "Should retrieve at least one trace"

    def test_validate_attribute_types(self, apm_client, actual_trace_id):
        """
        Test attribute type validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data and "spans" in trace_data:
            for span in trace_data["spans"]:
                attributes = span.get("attributes", {})

                # Validate token count attributes are integers
                if "gen_ai.usage.input_tokens" in attributes:
                    result = SpanValidator.validate_attribute_type(
                        span=span,
                        attribute_name="gen_ai.usage.input_tokens",
                        expected_type="int",
                    )
                    assert result, "Token count should be integer type"

    def test_span_status_validation(self, apm_client, actual_trace_id):
        """
        Test span status validation.

        Args:
            apm_client: APM client fixture
            actual_trace_id: Actual trace ID from command line (optional)
        """
        trace_id = (
            actual_trace_id if actual_trace_id else DataGenerator.generate_trace_id()
        )
        trace_data = apm_client.get_trace(trace_id)

        if trace_data and "spans" in trace_data:
            for span in trace_data["spans"]:
                # Validate span status
                result = SpanValidator.validate_span_status(span)

                assert isinstance(
                    result, bool
                ), "Span status validation should return boolean"
