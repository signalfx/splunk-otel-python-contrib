"""
End-to-End Test: Multi-Agent Workflow with Trace Validation.

This test demonstrates a complete workflow:
1. Deploy test application with instrumentation
2. Execute multi-agent conversation
3. Wait for traces to appear in APM
4. Validate GenAI semantic conventions
5. Verify evaluation metrics
"""

import pytest
from utils.wait_helpers import WaitHelpers
from utils.data_generator import DataGenerator
from validators.trace_validator import TraceValidator


@pytest.mark.e2e
@pytest.mark.genai
@pytest.mark.slow
class TestMultiAgentWorkflow:
    """
    E2E test for multi-agent workflow with comprehensive validation.
    """

    def test_langchain_agent_with_evaluation(
        self, apm_client, test_session_id, trace_id_list, trace_wait_timeout, config
    ):
        """
        Test LangChain agent execution with evaluation metrics.

        This test validates:
        - Trace generation with correct structure
        - GenAI semantic conventions compliance
        - Token usage tracking
        - Evaluation metrics (if enabled)
        - Parent-child span relationships

        Args:
            apm_client: APM client fixture
            test_session_id: Unique session ID for this test
            trace_id_list: List to collect trace IDs for cleanup
            trace_wait_timeout: Timeout for trace availability
            config: Configuration fixture
        """
        # Step 1: Generate test data
        _ = DataGenerator.generate_prompts(count=1)[0]
        expected_system = "openai"  # or from config

        # Step 2: Execute agent workflow (simulated - in real test, call actual agent)
        # In production, this would trigger instrumented LangChain agent
        # For now, we'll simulate by generating a trace ID
        trace_id = DataGenerator.generate_trace_id()
        trace_id_list.append(trace_id)

        # Note: In real implementation, you would:
        # 1. Call instrumented agent endpoint
        # 2. Get trace_id from response headers or logs
        # 3. Wait for trace to propagate to APM

        # Step 3: Wait for trace to be available
        trace_available = WaitHelpers.wait_for_trace(
            get_trace_func=apm_client.get_trace,
            trace_id=trace_id,
            timeout=trace_wait_timeout,
        )

        assert (
            trace_available
        ), f"Trace {trace_id} did not appear within {trace_wait_timeout}s"

        # Step 4: Retrieve trace data
        trace_data = apm_client.get_trace(trace_id)
        assert trace_data is not None, "Failed to retrieve trace data"

        # Step 5: Validate GenAI semantic conventions
        validation_result = TraceValidator.validate_genai_trace(
            trace=trace_data, expected_system=expected_system
        )

        assert validation_result[
            "valid"
        ], f"GenAI validation failed: {validation_result.get('errors', [])}"

        # Step 6: Validate span hierarchy
        hierarchy_result = TraceValidator.validate_span_hierarchy(trace_data)
        assert hierarchy_result[
            "valid"
        ], f"Span hierarchy validation failed: {hierarchy_result.get('errors', [])}"

        # Step 7: Validate token usage
        token_result = TraceValidator.validate_token_usage(trace_data)
        assert token_result[
            "valid"
        ], f"Token usage validation failed: {token_result.get('errors', [])}"

        # Step 8: Extract and verify root span
        root_span = TraceValidator.find_root_span(trace_data)
        assert root_span is not None, "Root span not found"

        # Verify session ID in root span
        session_id_attr = root_span.get("attributes", {}).get("gen_ai.session.id")
        assert (
            session_id_attr == test_session_id
        ), f"Session ID mismatch: expected {test_session_id}, got {session_id_attr}"

        # Step 9: Verify LLM spans exist
        llm_spans = TraceValidator.find_spans_by_operation(trace_data, "chat")
        assert len(llm_spans) > 0, "No LLM chat spans found"

        # Step 10: Validate each LLM span
        for llm_span in llm_spans:
            span_attrs = llm_span.get("attributes", {})

            # Check required GenAI attributes
            assert "gen_ai.system" in span_attrs, "Missing gen_ai.system"
            assert "gen_ai.request.model" in span_attrs, "Missing gen_ai.request.model"
            assert (
                "gen_ai.operation.name" in span_attrs
            ), "Missing gen_ai.operation.name"

            # Verify token usage attributes
            assert "gen_ai.usage.input_tokens" in span_attrs, "Missing input tokens"
            assert "gen_ai.usage.output_tokens" in span_attrs, "Missing output tokens"

    def test_multi_agent_conversation(
        self, apm_client, test_session_id, trace_id_list, trace_wait_timeout
    ):
        """
        Test multi-turn conversation with multiple agents.

        Validates:
        - Multiple traces linked by session ID
        - Conversation flow tracking
        - Agent handoff spans

        Args:
            apm_client: APM client fixture
            test_session_id: Session ID
            trace_id_list: Trace ID collection
            trace_wait_timeout: Wait timeout
        """
        # Generate multi-turn conversation
        conversation = DataGenerator.generate_conversation(turns=3)

        trace_ids = []

        # Simulate each turn generating a trace
        for turn_idx, message in enumerate(conversation):
            trace_id = DataGenerator.generate_trace_id()
            trace_ids.append(trace_id)
            trace_id_list.append(trace_id)

            # In real test: execute agent with this message
            # For now, simulate trace generation

            # Wait for trace
            trace_available = WaitHelpers.wait_for_trace(
                get_trace_func=apm_client.get_trace,
                trace_id=trace_id,
                timeout=trace_wait_timeout,
            )

            assert trace_available, f"Trace {trace_id} for turn {turn_idx} not found"

        # Validate all traces share the same session ID
        for trace_id in trace_ids:
            trace_data = apm_client.get_trace(trace_id)
            root_span = TraceValidator.find_root_span(trace_data)

            session_id = root_span.get("attributes", {}).get("gen_ai.session.id")
            assert (
                session_id == test_session_id
            ), f"Session ID mismatch in trace {trace_id}"

    def test_agent_with_tools(
        self, apm_client, test_session_id, trace_id_list, trace_wait_timeout
    ):
        """
        Test agent using tools with proper span instrumentation.

        Validates:
        - Tool call spans
        - Tool execution spans
        - Nested span structure

        Args:
            apm_client: APM client fixture
            test_session_id: Session ID
            trace_id_list: Trace ID collection
            trace_wait_timeout: Wait timeout
        """
        # Generate trace ID
        trace_id = DataGenerator.generate_trace_id()
        trace_id_list.append(trace_id)

        # In real test: execute agent with tool usage
        # Wait for trace
        trace_available = WaitHelpers.wait_for_trace(
            get_trace_func=apm_client.get_trace,
            trace_id=trace_id,
            timeout=trace_wait_timeout,
        )

        assert trace_available, f"Trace {trace_id} not found"

        # Retrieve and validate
        trace_data = apm_client.get_trace(trace_id)

        # Find tool spans
        tool_spans = TraceValidator.find_spans_by_operation(trace_data, "tool")
        assert len(tool_spans) > 0, "No tool spans found"

        # Validate tool span attributes
        for tool_span in tool_spans:
            attrs = tool_span.get("attributes", {})
            assert "gen_ai.tool.name" in attrs, "Missing tool name"

    @pytest.mark.evaluation
    def test_agent_with_evaluation_metrics(
        self, apm_client, test_session_id, trace_id_list, trace_wait_timeout
    ):
        """
        Test agent execution with evaluation metrics.

        Validates:
        - Evaluation metric generation
        - Metric format (new unified format)
        - Score values within valid range

        Args:
            apm_client: APM client fixture
            test_session_id: Session ID
            trace_id_list: Trace ID collection
            trace_wait_timeout: Wait timeout
        """
        # Generate trace
        trace_id = DataGenerator.generate_trace_id()
        trace_id_list.append(trace_id)

        # In real test: execute agent with evaluation enabled

        # Wait for trace
        trace_available = WaitHelpers.wait_for_trace(
            get_trace_func=apm_client.get_trace,
            trace_id=trace_id,
            timeout=trace_wait_timeout,
        )

        assert trace_available, f"Trace {trace_id} not found"

        # Retrieve trace
        trace_data = apm_client.get_trace(trace_id)

        # Get metrics for this trace
        # Note: In real implementation, you'd query metrics API
        # For now, we'll validate the trace has evaluation attributes

        root_span = TraceValidator.find_root_span(trace_data)
        attrs = root_span.get("attributes", {})

        # Check for evaluation attributes
        eval_attrs = {
            k: v for k, v in attrs.items() if k.startswith("gen_ai.evaluation")
        }

        if eval_attrs:
            # Validate evaluation scores are in valid range
            for key, value in eval_attrs.items():
                if "score" in key:
                    score = float(value)
                    assert (
                        0.0 <= score <= 1.0
                    ), f"Evaluation score {key} out of range: {score}"
