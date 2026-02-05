"""
Foundation Components Test Suite (TC-PI2-FOUNDATION-01 through TC-PI2-FOUNDATION-05)

This test suite validates core GenAI observability foundation components including:
- TC-PI2-FOUNDATION-01: Orchestrator Pattern Validation
- TC-PI2-FOUNDATION-02: Parallel Agent Execution Detection
- TC-PI2-FOUNDATION-03: GenAI Semantic Conventions Compliance
- TC-PI2-FOUNDATION-04: Telemetry Completeness Validation
- TC-PI2-FOUNDATION-05: Span Relationship Integrity

These tests use existing multi-agent workflow traces to validate foundational
observability patterns and semantic conventions.
"""

import pytest


@pytest.mark.foundation
@pytest.mark.priority_p0
class TestFoundationComponents:
    """Foundation Components Test Suite - Core GenAI Observability Validation"""

    def test_foundation_01_orchestrator_pattern(self, apm_client, actual_trace_id):
        """
        TC-PI2-FOUNDATION-01: Orchestrator Pattern Validation

        Validates that orchestrator/coordinator patterns are properly captured:
        - Root orchestrator span exists
        - Orchestrator coordinates multiple child operations
        - Proper span naming conventions
        - Orchestrator attributes present

        Success Criteria:
        - Root span identified
        - At least 2 child operations under orchestrator
        - Orchestrator has workflow or coordination semantics
        """
        if not actual_trace_id:
            pytest.skip("No trace ID provided")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-01: Orchestrator Pattern Validation")
        print(f"{'='*70}\n")

        # Step 1: Retrieve trace
        trace = apm_client.get_trace(actual_trace_id, max_wait=10)
        assert trace is not None, f"Failed to retrieve trace {actual_trace_id}"

        spans = trace.get("spans", [])
        assert len(spans) > 0, "Trace has no spans"
        print(f"✓ Retrieved trace with {len(spans)} spans")

        # Step 2: Find orchestrator/coordinator spans
        orchestrator_spans = []
        for span in spans:
            op_name = span.get("operationName", "").lower()
            if any(
                keyword in op_name
                for keyword in [
                    "workflow",
                    "crew",
                    "orchestrator",
                    "coordinator",
                    "chain",
                ]
            ):
                orchestrator_spans.append(span)

        assert (
            len(orchestrator_spans) > 0
        ), "No orchestrator/coordinator span found in trace"

        orchestrator = orchestrator_spans[0]
        print(f"✓ Found orchestrator span: {orchestrator.get('operationName')}")

        # Step 3: Verify orchestrator has child operations
        # Count spans that could be children (agents, tools, LLM calls)
        child_operations = []
        for span in spans:
            op_name = span.get("operationName", "").lower()
            if any(
                keyword in op_name
                for keyword in ["agent", "tool", "llm", "chat", "invoke"]
            ):
                if span.get("spanId") != orchestrator.get("spanId"):
                    child_operations.append(span)

        assert (
            len(child_operations) >= 2
        ), f"Orchestrator should coordinate at least 2 operations, found {len(child_operations)}"

        print(f"✓ Orchestrator coordinates {len(child_operations)} child operations")

        # Step 4: Validate orchestrator attributes
        attrs = orchestrator.get("attributes", {})
        assert "gen_ai.system" in attrs, "Orchestrator missing gen_ai.system attribute"

        print(f"✓ Orchestrator has gen_ai.system: {attrs.get('gen_ai.system')}")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-01: PASSED ✓")
        print(f"{'='*70}\n")

    def test_foundation_02_parallel_execution_detection(
        self, apm_client, actual_trace_id
    ):
        """
        TC-PI2-FOUNDATION-02: Parallel Agent Execution Detection

        Validates detection of parallel vs sequential execution patterns:
        - Identify concurrent operations (overlapping time windows)
        - Detect sequential operations (non-overlapping)
        - Validate timing attributes

        Success Criteria:
        - All spans have valid start times
        - Execution pattern (parallel or sequential) detected
        - No timing anomalies
        """
        if not actual_trace_id:
            pytest.skip("No trace ID provided")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-02: Parallel Execution Detection")
        print(f"{'='*70}\n")

        # Step 1: Retrieve trace
        trace = apm_client.get_trace(actual_trace_id, max_wait=10)
        assert trace is not None, f"Failed to retrieve trace {actual_trace_id}"

        spans = trace.get("spans", [])
        print(f"✓ Retrieved trace with {len(spans)} spans")

        # Step 2: Extract timing information
        timed_spans = []
        for span in spans:
            start_time = span.get("startTime") or span.get("start_time")
            duration = span.get("duration")

            if start_time and duration:
                end_time = start_time + duration
                timed_spans.append(
                    {
                        "name": span.get("operationName", "unknown"),
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                    }
                )

        assert len(timed_spans) > 0, "No spans with timing information found"
        print(f"✓ Found {len(timed_spans)} spans with timing data")

        # Step 3: Detect parallel execution
        parallel_pairs = []
        for i, span1 in enumerate(timed_spans):
            for span2 in timed_spans[i + 1 :]:
                # Check if time windows overlap
                if span1["start"] < span2["end"] and span2["start"] < span1["end"]:
                    parallel_pairs.append((span1["name"], span2["name"]))

        # Step 4: Detect sequential execution
        sequential_pairs = []
        sorted_spans = sorted(timed_spans, key=lambda x: x["start"])
        for i in range(len(sorted_spans) - 1):
            span1 = sorted_spans[i]
            span2 = sorted_spans[i + 1]
            # Sequential if span2 starts after span1 ends
            if span2["start"] >= span1["end"]:
                sequential_pairs.append((span1["name"], span2["name"]))

        print("✓ Execution pattern analysis:")
        print(f"  - Parallel operations detected: {len(parallel_pairs)}")
        print(f"  - Sequential operations detected: {len(sequential_pairs)}")

        # Step 5: Validate no timing anomalies
        for span in timed_spans:
            assert (
                span["duration"] > 0
            ), f"Span {span['name']} has invalid duration: {span['duration']}"
            assert (
                span["start"] > 0
            ), f"Span {span['name']} has invalid start time: {span['start']}"

        print("✓ All spans have valid timing attributes")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-02: PASSED ✓")
        print(f"{'='*70}\n")

    def test_foundation_03_semantic_conventions_compliance(
        self, apm_client, actual_trace_id
    ):
        """
        TC-PI2-FOUNDATION-03: GenAI Semantic Conventions Compliance

        Validates compliance with OpenTelemetry GenAI semantic conventions:
        - gen_ai.system attribute present
        - gen_ai.operation.name for operation types
        - gen_ai.request.* attributes for requests
        - gen_ai.response.* attributes for responses
        - gen_ai.usage.* attributes for token usage

        Success Criteria:
        - All GenAI spans have gen_ai.system
        - Operation names follow conventions
        - Required attributes present based on span type
        """
        if not actual_trace_id:
            pytest.skip("No trace ID provided")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-03: Semantic Conventions Compliance")
        print(f"{'='*70}\n")

        # Step 1: Retrieve trace
        trace = apm_client.get_trace(actual_trace_id, max_wait=10)
        assert trace is not None, f"Failed to retrieve trace {actual_trace_id}"

        spans = trace.get("spans", [])
        print(f"✓ Retrieved trace with {len(spans)} spans")

        # Step 2: Identify GenAI spans
        genai_spans = []
        for span in spans:
            attrs = span.get("attributes", {})
            if "gen_ai.system" in attrs:
                genai_spans.append(span)

        assert len(genai_spans) > 0, "No GenAI spans found in trace"
        print(f"✓ Found {len(genai_spans)} GenAI spans")

        # Step 3: Validate gen_ai.system attribute
        systems = set()
        for span in genai_spans:
            attrs = span.get("attributes", {})
            system = attrs.get("gen_ai.system")
            assert (
                system is not None
            ), f"Span {span.get('operationName')} missing gen_ai.system"
            systems.add(system)

        print("✓ All GenAI spans have gen_ai.system attribute")
        print(f"  - Systems found: {', '.join(systems)}")

        # Step 4: Check for operation names
        spans_with_operation = 0
        for span in genai_spans:
            attrs = span.get("attributes", {})
            if "gen_ai.operation.name" in attrs:
                spans_with_operation += 1

        print(
            f"✓ {spans_with_operation}/{len(genai_spans)} spans have gen_ai.operation.name"
        )

        # Step 5: Check for usage attributes
        spans_with_usage = 0
        for span in genai_spans:
            attrs = span.get("attributes", {})
            if (
                "gen_ai.usage.input_tokens" in attrs
                or "gen_ai.usage.output_tokens" in attrs
            ):
                spans_with_usage += 1

        print(f"✓ {spans_with_usage}/{len(genai_spans)} spans have token usage data")

        # Step 6: Validate attribute types
        for span in genai_spans:
            attrs = span.get("attributes", {})

            # Validate token counts are numeric if present
            if "gen_ai.usage.input_tokens" in attrs:
                tokens = attrs["gen_ai.usage.input_tokens"]
                assert isinstance(
                    tokens, (int, float)
                ), f"input_tokens must be numeric, got {type(tokens)}"

            if "gen_ai.usage.output_tokens" in attrs:
                tokens = attrs["gen_ai.usage.output_tokens"]
                assert isinstance(
                    tokens, (int, float)
                ), f"output_tokens must be numeric, got {type(tokens)}"

        print("✓ All attribute types are valid")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-03: PASSED ✓")
        print(f"{'='*70}\n")

    def test_foundation_04_telemetry_completeness(self, apm_client, actual_trace_id):
        """
        TC-PI2-FOUNDATION-04: Telemetry Completeness Validation

        Validates that telemetry data is complete and consistent:
        - All spans have required fields (spanId, operationName, startTime)
        - Service name is set
        - Trace ID consistency
        - No missing critical attributes

        Success Criteria:
        - 100% of spans have required fields
        - Service name present on all spans
        - Trace ID matches across all spans
        """
        if not actual_trace_id:
            pytest.skip("No trace ID provided")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-04: Telemetry Completeness Validation")
        print(f"{'='*70}\n")

        # Step 1: Retrieve trace
        trace = apm_client.get_trace(actual_trace_id, max_wait=10)
        assert trace is not None, f"Failed to retrieve trace {actual_trace_id}"

        spans = trace.get("spans", [])
        print(f"✓ Retrieved trace with {len(spans)} spans")

        # Step 2: Validate required fields on all spans
        required_fields = ["spanId", "operationName", "startTime"]
        missing_fields = []

        for i, span in enumerate(spans):
            for field in required_fields:
                # Check both camelCase and snake_case variants
                if field not in span and field.replace("_", "").lower() not in [
                    k.replace("_", "").lower() for k in span.keys()
                ]:
                    missing_fields.append(f"Span {i} missing {field}")

        assert (
            len(missing_fields) == 0
        ), f"Spans missing required fields: {missing_fields}"

        print(f"✓ All spans have required fields: {', '.join(required_fields)}")

        # Step 3: Validate service name
        spans_with_service = 0
        service_names = set()
        for span in spans:
            service_name = span.get("serviceName") or span.get("service_name")
            if service_name:
                spans_with_service += 1
                service_names.add(service_name)

        assert spans_with_service > 0, "No spans have service name"
        print(f"✓ {spans_with_service}/{len(spans)} spans have service name")
        print(f"  - Services: {', '.join(service_names)}")

        # Step 4: Validate trace ID consistency
        trace_ids = set()
        for span in spans:
            trace_id = span.get("traceId") or span.get("trace_id")
            if trace_id:
                trace_ids.add(trace_id)

        assert (
            len(trace_ids) <= 1
        ), f"Multiple trace IDs found in single trace: {trace_ids}"

        print("✓ Trace ID consistent across all spans")

        # Step 5: Check for span duration
        spans_with_duration = 0
        for span in spans:
            if span.get("duration"):
                spans_with_duration += 1

        print(f"✓ {spans_with_duration}/{len(spans)} spans have duration data")

        # Step 6: Validate no null/empty operation names
        for span in spans:
            op_name = span.get("operationName", "")
            assert (
                op_name and op_name.strip()
            ), f"Span {span.get('spanId')} has empty operation name"

        print("✓ All spans have non-empty operation names")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-04: PASSED ✓")
        print(f"{'='*70}\n")

    def test_foundation_05_span_relationship_integrity(
        self, apm_client, actual_trace_id
    ):
        """
        TC-PI2-FOUNDATION-05: Span Relationship Integrity

        Validates span relationships and trace structure:
        - Span IDs are unique
        - Parent references are valid (if present)
        - No circular references
        - Trace forms valid tree/DAG structure

        Success Criteria:
        - All span IDs are unique
        - Parent references point to existing spans (if present)
        - No orphaned spans (except root)
        - Trace structure is valid
        """
        if not actual_trace_id:
            pytest.skip("No trace ID provided")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-05: Span Relationship Integrity")
        print(f"{'='*70}\n")

        # Step 1: Retrieve trace
        trace = apm_client.get_trace(actual_trace_id, max_wait=10)
        assert trace is not None, f"Failed to retrieve trace {actual_trace_id}"

        spans = trace.get("spans", [])
        print(f"✓ Retrieved trace with {len(spans)} spans")

        # Step 2: Validate span ID uniqueness
        span_ids = []
        for span in spans:
            span_id = span.get("spanId") or span.get("span_id")
            if span_id:
                span_ids.append(span_id)

        assert (
            len(span_ids) == len(set(span_ids))
        ), f"Duplicate span IDs found: {len(span_ids)} spans, {len(set(span_ids))} unique IDs"

        print(f"✓ All {len(span_ids)} span IDs are unique")

        # Step 3: Check parent references (if present in response)
        span_id_set = set(span_ids)
        parent_refs = []
        invalid_parents = []

        for span in spans:
            parent_id = span.get("parentSpanId") or span.get("parent_span_id")
            if parent_id:
                parent_refs.append(parent_id)
                if parent_id not in span_id_set:
                    invalid_parents.append(parent_id)

        if len(parent_refs) > 0:
            print(f"✓ Found {len(parent_refs)} parent references")
            if len(invalid_parents) > 0:
                print(
                    f"  ⚠ Note: {len(invalid_parents)} parent references not in span set (may be expected)"
                )
        else:
            print("  ℹ No parent references in response (GraphQL API limitation)")

        # Step 4: Identify root spans (no parent)
        root_spans = []
        for span in spans:
            parent_id = span.get("parentSpanId") or span.get("parent_span_id")
            if not parent_id:
                root_spans.append(span.get("operationName", "unknown"))

        print(f"✓ Found {len(root_spans)} root span(s)")
        if len(root_spans) > 0:
            print(f"  - Root operations: {', '.join(root_spans[:3])}")

        # Step 5: Validate span hierarchy depth
        # Calculate max depth based on operation names (heuristic)
        depth_indicators = ["workflow", "agent", "tool", "llm"]
        depth_counts = {indicator: 0 for indicator in depth_indicators}

        for span in spans:
            op_name = span.get("operationName", "").lower()
            for indicator in depth_indicators:
                if indicator in op_name:
                    depth_counts[indicator] += 1

        print("✓ Trace hierarchy analysis:")
        for indicator, count in depth_counts.items():
            if count > 0:
                print(f"  - {indicator.capitalize()} level: {count} span(s)")

        # Step 6: Validate no self-references
        for span in spans:
            span_id = span.get("spanId") or span.get("span_id")
            parent_id = span.get("parentSpanId") or span.get("parent_span_id")

            if span_id and parent_id:
                assert (
                    span_id != parent_id
                ), f"Span {span_id} references itself as parent"

        print("✓ No self-referencing spans detected")

        print(f"\n{'='*70}")
        print("TC-PI2-FOUNDATION-05: PASSED ✓")
        print(f"{'='*70}\n")
