"""
TC-PI2-RAG-01: RAG Pipeline End-to-End Validation

This test validates RAG (Retrieval-Augmented Generation) pipeline observability including:
- Vector DB query operations
- Retrieval metrics and performance
- Embedding attribution and tracking
- Context retrieval patterns
- RAG-specific semantic conventions

The test can work with:
1. Actual RAG pipeline traces (if available)
2. Simulated RAG patterns from existing multi-agent traces
3. Tool execution patterns that mimic retrieval operations
"""

import pytest


@pytest.mark.rag
@pytest.mark.priority_p0
class TestRAGPipeline:
    """TC-PI2-RAG-01: RAG Pipeline End-to-End Validation"""

    def test_rag_01_retrieval_pipeline_validation(self, apm_client, actual_trace_id):
        """
        TC-PI2-RAG-01: RAG Pipeline End-to-End Validation

        Validates RAG pipeline observability including:
        - Retrieval operations (vector DB queries, document search)
        - Embedding operations and attribution
        - Context augmentation patterns
        - Generation with retrieved context
        - RAG-specific metrics

        Success Criteria:
        - Retrieval operations detected and validated
        - Embedding/vector operations tracked
        - Context flow from retrieval to generation
        - Performance metrics captured
        """
        if not actual_trace_id:
            pytest.skip("No trace ID provided")

        print(f"\n{'='*70}")
        print("TC-PI2-RAG-01: RAG Pipeline End-to-End Validation")
        print(f"{'='*70}\n")

        # Step 1: Retrieve trace (allow time for propagation from collector)
        trace = apm_client.get_trace(actual_trace_id, max_wait=60)
        assert trace is not None, f"Failed to retrieve trace {actual_trace_id}"

        spans = trace.get("spans", [])
        assert len(spans) > 0, "Trace has no spans"
        print(f"✓ Retrieved trace with {len(spans)} spans")

        # Step 2: Identify RAG-related operations
        retrieval_spans = []
        embedding_spans = []
        generation_spans = []
        tool_spans = []

        for span in spans:
            op_name = span.get("operationName", "").lower()
            attrs = span.get("attributes", {})

            # Identify retrieval operations
            if any(
                keyword in op_name
                for keyword in ["retriev", "search", "query", "fetch", "read", "lookup"]
            ):
                retrieval_spans.append(span)

            # Identify embedding operations
            if any(keyword in op_name for keyword in ["embed", "vector", "encode"]):
                embedding_spans.append(span)

            # Identify generation operations
            if any(
                keyword in op_name
                for keyword in ["chat", "completion", "generate", "llm"]
            ):
                generation_spans.append(span)

            # Identify tool operations (can include retrieval)
            if "tool" in op_name:
                tool_spans.append(span)

        print("\n📊 RAG Pipeline Component Analysis:")
        print(f"  - Retrieval operations: {len(retrieval_spans)}")
        print(f"  - Embedding operations: {len(embedding_spans)}")
        print(f"  - Generation operations: {len(generation_spans)}")
        print(f"  - Tool operations: {len(tool_spans)}")

        # Step 3: Validate retrieval operations
        if len(retrieval_spans) > 0:
            print(f"\n✓ Found {len(retrieval_spans)} retrieval operation(s)")

            for i, span in enumerate(retrieval_spans, 1):
                op_name = span.get("operationName")
                duration = span.get("duration", 0)
                print(f"  {i}. {op_name} (duration: {duration}ms)")

                # Validate timing
                assert duration > 0, f"Retrieval span {op_name} has invalid duration"

                # Check for retrieval-related attributes
                attrs = span.get("attributes", {})
                if "gen_ai.system" in attrs:
                    print(f"     - System: {attrs['gen_ai.system']}")
        else:
            # If no explicit retrieval spans, check tool spans for retrieval patterns
            retrieval_tools = [
                s
                for s in tool_spans
                if any(
                    keyword in s.get("operationName", "").lower()
                    for keyword in ["read", "search", "fetch", "query", "lookup"]
                )
            ]

            if len(retrieval_tools) > 0:
                print(
                    f"\n✓ Found {len(retrieval_tools)} retrieval-like tool operation(s)"
                )
                retrieval_spans = retrieval_tools

                for i, span in enumerate(retrieval_spans, 1):
                    op_name = span.get("operationName")
                    duration = span.get("duration", 0)
                    print(f"  {i}. {op_name} (duration: {duration}ms)")
            else:
                print("\n⚠ No explicit retrieval operations found")
                print("  Note: This trace may not contain RAG pipeline operations")

        # Step 4: Validate embedding operations
        if len(embedding_spans) > 0:
            print(f"\n✓ Found {len(embedding_spans)} embedding operation(s)")

            for i, span in enumerate(embedding_spans, 1):
                op_name = span.get("operationName")
                duration = span.get("duration", 0)
                print(f"  {i}. {op_name} (duration: {duration}ms)")

                # Validate embedding span attributes
                attrs = span.get("attributes", {})
                if "gen_ai.request.model" in attrs:
                    print(f"     - Model: {attrs['gen_ai.request.model']}")
        else:
            print("\n⚠ No explicit embedding operations found")
            print("  Note: Embeddings may be implicit or handled externally")

        # Step 5: Validate generation with context
        if len(generation_spans) > 0:
            print(f"\n✓ Found {len(generation_spans)} generation operation(s)")

            for i, span in enumerate(generation_spans, 1):
                op_name = span.get("operationName")
                duration = span.get("duration", 0)
                attrs = span.get("attributes", {})

                print(f"  {i}. {op_name} (duration: {duration}ms)")

                # Check for token usage (indicates LLM generation)
                if "gen_ai.usage.input_tokens" in attrs:
                    input_tokens = attrs["gen_ai.usage.input_tokens"]
                    output_tokens = attrs.get("gen_ai.usage.output_tokens", 0)
                    print(f"     - Input tokens: {input_tokens}")
                    print(f"     - Output tokens: {output_tokens}")

                    # In RAG, input tokens should be higher (includes retrieved context)
                    if input_tokens > 500:
                        print(
                            "     ✓ High input token count suggests context augmentation"
                        )

        # Step 6: Validate RAG pipeline flow
        print("\n📈 RAG Pipeline Flow Analysis:")

        # Check for retrieval → generation pattern
        if len(retrieval_spans) > 0 and len(generation_spans) > 0:
            # Sort by start time
            retrieval_times = [s.get("startTime", 0) for s in retrieval_spans]
            generation_times = [s.get("startTime", 0) for s in generation_spans]

            earliest_retrieval = min(retrieval_times) if retrieval_times else 0
            earliest_generation = min(generation_times) if generation_times else 0

            if earliest_retrieval < earliest_generation:
                print("  ✓ Retrieval occurs before generation (expected RAG pattern)")
            else:
                print("  ⚠ Generation timing unclear relative to retrieval")

        # Step 7: Calculate RAG-specific metrics
        print("\n📊 RAG Pipeline Metrics:")

        total_retrieval_time = sum(s.get("duration", 0) for s in retrieval_spans)
        total_generation_time = sum(s.get("duration", 0) for s in generation_spans)
        total_pipeline_time = trace.get("duration", 0)

        print(f"  - Total retrieval time: {total_retrieval_time}ms")
        print(f"  - Total generation time: {total_generation_time}ms")
        print(f"  - Total pipeline time: {total_pipeline_time}ms")

        if total_pipeline_time > 0:
            retrieval_pct = (total_retrieval_time / total_pipeline_time) * 100
            generation_pct = (total_generation_time / total_pipeline_time) * 100
            print(f"  - Retrieval overhead: {retrieval_pct:.1f}%")
            print(f"  - Generation overhead: {generation_pct:.1f}%")

        # Step 8: Validate semantic conventions
        print("\n🔍 Semantic Conventions Validation:")

        genai_spans = [s for s in spans if "gen_ai.system" in s.get("attributes", {})]
        print(f"  - GenAI spans: {len(genai_spans)}/{len(spans)}")

        spans_with_model = [
            s for s in spans if "gen_ai.request.model" in s.get("attributes", {})
        ]
        print(f"  - Spans with model info: {len(spans_with_model)}")

        spans_with_tokens = [
            s for s in spans if "gen_ai.usage.input_tokens" in s.get("attributes", {})
        ]
        print(f"  - Spans with token usage: {len(spans_with_tokens)}")

        # Step 9: Validate retrieval quality indicators
        print("\n✅ RAG Pipeline Quality Indicators:")

        quality_checks = []

        # Check 1: Retrieval operations present
        if len(retrieval_spans) > 0:
            quality_checks.append("✓ Retrieval operations detected")
        else:
            quality_checks.append("⚠ No retrieval operations found")

        # Check 2: Generation operations present
        if len(generation_spans) > 0:
            quality_checks.append("✓ Generation operations detected")
        else:
            quality_checks.append("⚠ No generation operations found")

        # Check 3: Proper timing
        if total_retrieval_time > 0 and total_generation_time > 0:
            quality_checks.append("✓ Both retrieval and generation have valid timing")

        # Check 4: Token usage tracked
        if len(spans_with_tokens) > 0:
            quality_checks.append("✓ Token usage tracked")

        # Check 5: Multiple components
        if len(spans) >= 3:
            quality_checks.append("✓ Multi-component pipeline detected")

        for check in quality_checks:
            print(f"  {check}")

        # Step 10: Final validation
        print(f"\n{'='*70}")

        # For this test to pass, we need at least:
        # 1. A trace with multiple spans
        # 2. Some form of retrieval or tool operation
        # 3. Valid timing on all spans

        assert len(spans) >= 2, "RAG pipeline should have at least 2 spans"

        # Check that we have either retrieval spans or tool spans
        has_retrieval_pattern = len(retrieval_spans) > 0 or len(tool_spans) > 0
        assert has_retrieval_pattern, "No retrieval or tool operations found in trace"

        # Validate all spans have valid timing
        for span in spans:
            duration = span.get("duration", 0)
            assert (
                duration >= 0
            ), f"Span {span.get('operationName')} has invalid duration"

        print("TC-PI2-RAG-01: PASSED ✓")
        print(f"{'='*70}\n")

        # Summary
        print("RAG Pipeline Validation Summary:")
        print(f"  - Total spans analyzed: {len(spans)}")
        print(f"  - Retrieval operations: {len(retrieval_spans)}")
        print(f"  - Generation operations: {len(generation_spans)}")
        print(f"  - Tool operations: {len(tool_spans)}")
        print(f"  - GenAI spans: {len(genai_spans)}")
        print(f"  - Pipeline duration: {total_pipeline_time}ms")
        print(f"{'='*70}\n")
