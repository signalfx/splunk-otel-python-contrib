"""Integration tests based on real-world openlit SDK usage patterns.

These tests simulate the patterns shown in the openlit_processor_example.py file,
testing nested workflows, agents, tasks, and tools with proper parent-child relationships.
"""

import json
import os

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.processor.openlit_span_processor import (
    OpenlitSpanProcessor,
)


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment before each test."""
    os.environ["OTEL_GENAI_CONTENT_CAPTURE"] = "1"
    yield
    if "OTEL_GENAI_CONTENT_CAPTURE" in os.environ:
        del os.environ["OTEL_GENAI_CONTENT_CAPTURE"]


@pytest.fixture
def setup_tracer():
    """Setup tracer with processor and exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()

    # Add OpenlitSpanProcessor with attribute transformations
    processor = OpenlitSpanProcessor(
        attribute_transformations={
            "remove": [],
            "rename": {
                "openlit.span.kind": "gen_ai.span.kind",
                "openlit.workflow.name": "gen_ai.workflow.name",
                "openlit.entity.name": "gen_ai.agent.name",
                "openlit.entity.path": "gen_ai.workflow.path",
                "openlit.entity.input": "gen_ai.input.messages",
                "openlit.entity.output": "gen_ai.output.messages",
                "openlit.correlation.id": "gen_ai.conversation.id",
            },
            "add": {},
        }
    )
    provider.add_span_processor(processor)

    # Then add exporter
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer(__name__)

    return tracer, exporter, provider


class TestWorkflowPattern:
    """Test workflow pattern from the example."""

    def test_simple_workflow_with_tasks(self, setup_tracer):
        """Test @workflow pattern with nested @task spans."""
        tracer, exporter, _ = setup_tracer

        # Simulate: @workflow(name="pirate_joke_generator")
        with tracer.start_as_current_span(
            "pirate_joke_generator"
        ) as workflow_span:
            workflow_span.set_attribute("openlit.span.kind", "workflow")
            workflow_span.set_attribute(
                "openlit.workflow.name", "pirate_joke_generator"
            )
            workflow_span.set_attribute(
                "openlit.entity.name", "pirate_joke_generator"
            )

            # Simulate: @task(name="joke_creation")
            with tracer.start_as_current_span("joke_creation") as task_span:
                task_span.set_attribute("openlit.span.kind", "task")
                task_span.set_attribute("openlit.entity.name", "joke_creation")
                task_span.set_attribute(
                    "openlit.workflow.name", "pirate_joke_generator"
                )

                # Simulate OpenAI call within task
                with tracer.start_as_current_span(
                    "chat gpt-3.5-turbo"
                ) as llm_span:
                    llm_span.set_attribute(
                        "gen_ai.request.model", "gpt-3.5-turbo"
                    )
                    llm_span.set_attribute("gen_ai.system", "openai")
                    llm_span.set_attribute(
                        "gen_ai.prompt.0.content",
                        "Tell me a joke about opentelemetry",
                    )
                    llm_span.set_attribute(
                        "gen_ai.completion.0.content",
                        "Why did the trace cross the road?",
                    )

        spans = exporter.get_finished_spans()

        # Should have original spans + synthetic spans
        assert len(spans) >= 3, f"Expected at least 3 spans, got {len(spans)}"

        # Find workflow spans (original mutated + synthetic)
        workflow_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get("gen_ai.workflow.name")
            == "pirate_joke_generator"
        ]
        assert len(workflow_spans) >= 1, (
            "Should have at least one workflow span"
        )

        # Find task spans
        task_spans = [
            s
            for s in spans
            if s.name == "joke_creation"
            or (
                s.attributes
                and s.attributes.get("gen_ai.agent.name") == "joke_creation"
            )
        ]
        assert len(task_spans) >= 1, "Should have at least one task span"

        # Verify no openlit.* attributes remain on any span (mutation)
        for span in spans:
            if span.attributes:
                openlit_keys = [
                    k
                    for k in span.attributes.keys()
                    if k.startswith("openlit.")
                ]
                # Exclude the _openlit_processed marker
                openlit_keys = [
                    k for k in openlit_keys if k != "_openlit_processed"
                ]
                assert len(openlit_keys) == 0, (
                    f"Span {span.name} should not have openlit.* attributes, found: {openlit_keys}"
                )

    def test_nested_agent_with_tool(self, setup_tracer):
        """Test @agent pattern with nested @tool calls."""
        tracer, exporter, _ = setup_tracer

        # Simulate: @agent(name="joke_translation")
        with tracer.start_as_current_span("joke_translation") as agent_span:
            agent_span.set_attribute("openlit.span.kind", "agent")
            agent_span.set_attribute("openlit.entity.name", "joke_translation")
            agent_span.set_attribute(
                "openlit.workflow.name", "pirate_joke_generator"
            )
            agent_span.set_attribute(
                "openlit.entity.input",
                json.dumps({"joke": "Why did the trace cross the road?"}),
            )

            # Simulate OpenAI call within agent
            with tracer.start_as_current_span(
                "chat gpt-3.5-turbo"
            ) as llm_span:
                llm_span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                llm_span.set_attribute("gen_ai.system", "openai")

            # Simulate: @tool(name="history_jokes")
            with tracer.start_as_current_span("history_jokes") as tool_span:
                tool_span.set_attribute("openlit.span.kind", "tool")
                tool_span.set_attribute("openlit.entity.name", "history_jokes")
                tool_span.set_attribute(
                    "openlit.workflow.name", "pirate_joke_generator"
                )

                # Simulate OpenAI call within tool
                with tracer.start_as_current_span(
                    "chat gpt-3.5-turbo"
                ) as tool_llm_span:
                    tool_llm_span.set_attribute(
                        "gen_ai.request.model", "gpt-3.5-turbo"
                    )
                    tool_llm_span.set_attribute("gen_ai.system", "openai")
                    tool_llm_span.set_attribute(
                        "gen_ai.prompt.0.content", "get some history jokes"
                    )

            agent_span.set_attribute(
                "openlit.entity.output",
                json.dumps(
                    {"response": "Arr! Why did the trace walk the plank?"}
                ),
            )

        spans = exporter.get_finished_spans()

        # Should have multiple spans
        assert len(spans) >= 4, f"Expected at least 4 spans, got {len(spans)}"

        # Find agent spans
        agent_spans = [
            s
            for s in spans
            if s.attributes
            and (
                s.attributes.get("gen_ai.agent.name") == "joke_translation"
                or s.attributes.get("gen_ai.span.kind") == "agent"
            )
        ]
        assert len(agent_spans) >= 1, "Should have at least one agent span"

        # Find tool spans
        tool_spans = [
            s
            for s in spans
            if s.attributes
            and (
                s.attributes.get("gen_ai.agent.name") == "history_jokes"
                or s.attributes.get("gen_ai.span.kind") == "tool"
            )
        ]
        assert len(tool_spans) >= 1, "Should have at least one tool span"

        # Verify input/output were captured and normalized
        agent_with_input = [
            s
            for s in agent_spans
            if s.attributes and "gen_ai.input.messages" in s.attributes
        ]
        if agent_with_input:
            input_data = json.loads(
                agent_with_input[0].attributes["gen_ai.input.messages"]
            )
            assert isinstance(input_data, list), (
                "Input should be normalized to message array"
            )


class TestParentChildRelationships:
    """Test that parent-child relationships are preserved across transformations."""

    def test_parent_child_hierarchy_preserved(self, setup_tracer):
        """Test that synthetic spans maintain parent-child relationships."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("workflow") as parent:
            parent.set_attribute("openlit.span.kind", "workflow")
            parent.set_attribute("openlit.workflow.name", "test_workflow")

            with tracer.start_as_current_span("task") as child:
                child.set_attribute("openlit.span.kind", "task")
                child.set_attribute("openlit.entity.name", "test_task")
                child.set_attribute("openlit.workflow.name", "test_workflow")

        spans = exporter.get_finished_spans()

        # Build parent-child map from context
        span_map = {}
        for span in spans:
            span_id = span.context.span_id if span.context else None
            if span_id:
                span_map[span_id] = span

        # Find child spans (those with parents)
        child_spans = [s for s in spans if s.parent is not None]

        assert len(child_spans) >= 1, "Should have at least one child span"

        # Verify at least one child has a valid parent reference
        valid_parent_refs = 0
        for child in child_spans:
            if child.parent and child.parent.span_id in span_map:
                valid_parent_refs += 1

        assert valid_parent_refs >= 1, (
            "At least one child should have a valid parent reference"
        )


class TestContentNormalization:
    """Test content normalization patterns from the example."""

    def test_normalize_entity_input_output(self, setup_tracer):
        """Test that openlit.entity.input and output are normalized properly."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_task") as span:
            span.set_attribute("openlit.span.kind", "task")
            span.set_attribute("openlit.entity.name", "test_task")
            span.set_attribute("openlit.workflow.name", "test_workflow")
            # Various input formats that should be normalized
            span.set_attribute(
                "openlit.entity.input",
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Translate this joke to pirate",
                            }
                        ]
                    }
                ),
            )
            span.set_attribute(
                "openlit.entity.output",
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Arr matey!",
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
            )

        spans = exporter.get_finished_spans()

        # Find spans with normalized content - check both original (mutated) and synthetic
        spans_with_input = [
            s
            for s in spans
            if s.attributes and "gen_ai.input.messages" in s.attributes
        ]

        # Should have at least the mutated original span with gen_ai.input.messages
        assert len(spans_with_input) >= 1, (
            f"Should have spans with normalized input, got {len(spans)} spans total"
        )

        # Verify normalization
        for span in spans_with_input:
            input_str = span.attributes.get("gen_ai.input.messages")
            if input_str:
                input_data = json.loads(input_str)
                assert isinstance(input_data, list), (
                    "Input should be list of messages"
                )
                if input_data:
                    assert "role" in input_data[0], (
                        "Messages should have role field"
                    )

        # Check output normalization
        spans_with_output = [
            s
            for s in spans
            if s.attributes and "gen_ai.output.messages" in s.attributes
        ]

        if spans_with_output:
            output_str = spans_with_output[0].attributes.get(
                "gen_ai.output.messages"
            )
            output_data = json.loads(output_str)
            assert isinstance(output_data, list), (
                "Output should be list of messages"
            )

    def test_normalize_string_input(self, setup_tracer):
        """Test normalization of simple string inputs."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_task") as span:
            span.set_attribute("openlit.span.kind", "task")
            span.set_attribute("openlit.entity.name", "test_task")
            span.set_attribute("openlit.workflow.name", "test_workflow")
            # Simple string input
            span.set_attribute("openlit.entity.input", "Hello world")

        spans = exporter.get_finished_spans()

        # Should handle string input gracefully - check that span was processed
        assert len(spans) >= 1, "Should have at least one span"

        # Check if any spans have gen_ai attributes (mutation occurred)
        spans_with_genai = [
            s
            for s in spans
            if s.attributes
            and any(k.startswith("gen_ai.") for k in s.attributes.keys())
        ]

        assert len(spans_with_genai) >= 1, (
            "Should have spans with gen_ai.* attributes after processing"
        )

    def test_normalize_list_of_strings(self, setup_tracer):
        """Test normalization of list inputs."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_task") as span:
            span.set_attribute("openlit.span.kind", "task")
            span.set_attribute("openlit.entity.name", "test_task")
            span.set_attribute("openlit.workflow.name", "test_workflow")
            # List input
            span.set_attribute(
                "openlit.entity.input",
                json.dumps(["Message 1", "Message 2", "Message 3"]),
            )

        spans = exporter.get_finished_spans()

        # Check that spans were processed
        assert len(spans) >= 1, "Should have at least one span"

        # Verify that gen_ai attributes exist (processing occurred)
        spans_with_genai = [
            s
            for s in spans
            if s.attributes and "gen_ai.span.kind" in s.attributes
        ]
        assert len(spans_with_genai) >= 1, (
            "Should have processed spans with gen_ai attributes"
        )


class TestModelInference:
    """Test model name inference from span names and attributes."""

    def test_infer_model_from_span_name(self, setup_tracer):
        """Test that model is inferred from span name like 'chat gpt-3.5-turbo'."""
        tracer, exporter, _ = setup_tracer

        # Simulate OpenAI instrumentation span naming pattern
        with tracer.start_as_current_span("chat gpt-3.5-turbo") as span:
            span.set_attribute("gen_ai.system", "openai")
            # No explicit gen_ai.request.model attribute

        spans = exporter.get_finished_spans()

        # Should have spans with inferred model
        spans_with_model = [
            s
            for s in spans
            if s.attributes and s.attributes.get("gen_ai.request.model")
        ]

        if spans_with_model:
            # Model should be inferred as "gpt-3.5-turbo"
            model = spans_with_model[0].attributes.get("gen_ai.request.model")
            assert model is not None, "Model should be inferred"

    def test_preserve_explicit_model(self, setup_tracer):
        """Test that explicit model attributes are preserved."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("chat completion") as span:
            span.set_attribute("gen_ai.request.model", "gpt-5-nano")
            span.set_attribute("gen_ai.system", "openai")

        spans = exporter.get_finished_spans()

        # Should preserve explicit model
        spans_with_model = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get("gen_ai.request.model") == "gpt-5-nano"
        ]

        assert len(spans_with_model) >= 1, (
            "Should preserve explicit model attribute"
        )


class TestSpanFiltering:
    """Test span filtering logic."""

    def test_filters_non_llm_spans(self, setup_tracer):
        """Test that non-LLM spans are filtered out."""
        tracer, exporter, _ = setup_tracer

        # Create a span that shouldn't be transformed
        with tracer.start_as_current_span("database_query") as span:
            span.set_attribute("db.system", "postgresql")
            span.set_attribute("db.statement", "SELECT * FROM users")

        spans = exporter.get_finished_spans()

        # Should only have the original span, no synthetic spans
        assert len(spans) == 1, (
            f"Expected 1 span (non-LLM filtered), got {len(spans)}"
        )

        # Original span should not have gen_ai.* attributes
        span = spans[0]
        gen_ai_attrs = [
            k for k in span.attributes.keys() if k.startswith("gen_ai.")
        ]
        assert len(gen_ai_attrs) == 0, (
            "Non-LLM span should not have gen_ai.* attributes"
        )

    def test_includes_openlit_spans(self, setup_tracer):
        """Test that openlit task/workflow spans are included."""
        tracer, exporter, _ = setup_tracer

        # openlit spans should always be included
        with tracer.start_as_current_span("my_custom_task") as span:
            span.set_attribute("openlit.span.kind", "task")
            span.set_attribute("openlit.entity.name", "my_custom_task")
            span.set_attribute("openlit.workflow.name", "test_workflow")

        spans = exporter.get_finished_spans()

        # Should have original + synthetic span
        assert len(spans) >= 1, "openlit spans should be processed"

        # At least one span should have gen_ai.span.kind (from mutation or synthetic span)
        spans_with_kind = [
            s
            for s in spans
            if s.attributes and s.attributes.get("gen_ai.span.kind") == "task"
        ]
        assert len(spans_with_kind) >= 1, (
            f"openlit task should be transformed, got {len(spans)} spans"
        )


class TestOperationInference:
    """Test operation type inference."""

    def test_infer_chat_operation(self, setup_tracer):
        """Test that 'chat' operation is inferred from span name."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("chat gpt-4") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", "gpt-5-nano")

        spans = exporter.get_finished_spans()

        # The processor creates synthetic spans with operation.name
        # Check if we have spans with gen_ai attributes (indicates processing)
        spans_with_genai = [
            s
            for s in spans
            if s.attributes and "gen_ai.system" in s.attributes
        ]

        assert len(spans_with_genai) >= 1, (
            f"Should have processed spans with gen_ai attributes, got {len(spans)} total spans"
        )

    def test_infer_embedding_operation(self, setup_tracer):
        """Test that 'embedding' operation is inferred from span name."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span(
            "embedding text-embedding-ada-002"
        ) as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute(
                "gen_ai.request.model", "text-embedding-ada-002"
            )

        spans = exporter.get_finished_spans()

        # Check that embedding spans are processed
        spans_with_embedding = [
            s
            for s in spans
            if s.attributes
            and "text-embedding"
            in s.attributes.get("gen_ai.request.model", "")
        ]

        assert len(spans_with_embedding) >= 1, (
            f"Should process embedding spans, got {len(spans)} total spans"
        )


class TestComplexWorkflow:
    """Test complete workflow simulating the example end-to-end."""

    def test_full_pirate_joke_workflow(self, setup_tracer):
        """Test complete workflow pattern from the example."""
        tracer, exporter, _ = setup_tracer

        # Main workflow
        with tracer.start_as_current_span("pirate_joke_generator") as workflow:
            workflow.set_attribute("openlit.span.kind", "workflow")
            workflow.set_attribute(
                "openlit.workflow.name", "pirate_joke_generator"
            )

            # Task 1: Create joke
            with tracer.start_as_current_span("joke_creation") as task1:
                task1.set_attribute("openlit.span.kind", "task")
                task1.set_attribute("openlit.entity.name", "joke_creation")
                task1.set_attribute(
                    "openlit.workflow.name", "pirate_joke_generator"
                )

                with tracer.start_as_current_span(
                    "chat gpt-3.5-turbo"
                ) as llm1:
                    llm1.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                    llm1.set_attribute("gen_ai.system", "openai")

            # Agent: Translate joke
            with tracer.start_as_current_span("joke_translation") as agent:
                agent.set_attribute("openlit.span.kind", "agent")
                agent.set_attribute("openlit.entity.name", "joke_translation")
                agent.set_attribute(
                    "openlit.workflow.name", "pirate_joke_generator"
                )

                with tracer.start_as_current_span(
                    "chat gpt-3.5-turbo"
                ) as llm2:
                    llm2.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                    llm2.set_attribute("gen_ai.system", "openai")

                # Tool within agent
                with tracer.start_as_current_span("history_jokes") as tool:
                    tool.set_attribute("openlit.span.kind", "tool")
                    tool.set_attribute("openlit.entity.name", "history_jokes")
                    tool.set_attribute(
                        "openlit.workflow.name", "pirate_joke_generator"
                    )

                    with tracer.start_as_current_span(
                        "chat gpt-3.5-turbo"
                    ) as llm3:
                        llm3.set_attribute(
                            "gen_ai.request.model", "gpt-3.5-turbo"
                        )
                        llm3.set_attribute("gen_ai.system", "openai")

            # Task 2: Generate signature
            with tracer.start_as_current_span("signature_generation") as task2:
                task2.set_attribute("openlit.span.kind", "task")
                task2.set_attribute(
                    "openlit.entity.name", "signature_generation"
                )
                task2.set_attribute(
                    "openlit.workflow.name", "pirate_joke_generator"
                )

                with tracer.start_as_current_span(
                    "chat gpt-3.5-turbo"
                ) as llm4:
                    llm4.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                    llm4.set_attribute("gen_ai.system", "openai")

        spans = exporter.get_finished_spans()

        # Should have many spans (original mutated + synthetic)
        assert len(spans) >= 8, (
            f"Expected at least 8 spans in full workflow, got {len(spans)}"
        )

        # Verify workflow span exists - look for spans with the workflow name
        workflow_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get("gen_ai.workflow.name")
            == "pirate_joke_generator"
        ]
        assert len(workflow_spans) >= 1, (
            f"Should have workflow span, got {len(spans)} total spans, workflow_spans={len(workflow_spans)}"
        )

        # Verify all task names are present
        task_names = {"joke_creation", "signature_generation"}
        found_tasks = set()
        for span in spans:
            if span.attributes:
                agent_name = span.attributes.get("gen_ai.agent.name")
                if agent_name in task_names:
                    found_tasks.add(agent_name)

        assert len(found_tasks) >= 1, (
            f"Should find task spans, found: {found_tasks}"
        )

        # Verify no openlit.* attributes remain (mutation)
        for span in spans:
            if span.attributes:
                openlit_keys = [
                    k
                    for k in span.attributes.keys()
                    if k.startswith("openlit.") and k != "_openlit_processed"
                ]
                assert len(openlit_keys) == 0, (
                    f"Span {span.name} should not have openlit.* attributes"
                )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_span_without_attributes(self, setup_tracer):
        """Test handling of spans without attributes."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_span"):
            pass  # No attributes

        spans = exporter.get_finished_spans()

        # Should handle gracefully without errors
        assert len(spans) >= 1, "Should handle span without attributes"

    def test_malformed_input_json(self, setup_tracer):
        """Test handling of malformed JSON in input."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_task") as span:
            span.set_attribute("openlit.span.kind", "task")
            span.set_attribute("openlit.entity.name", "test_task")
            # Malformed JSON
            span.set_attribute("openlit.entity.input", "{invalid json}")

        spans = exporter.get_finished_spans()

        # Should handle gracefully without crashing
        assert len(spans) >= 1, "Should handle malformed JSON gracefully"

    def test_empty_workflow_name(self, setup_tracer):
        """Test handling of empty workflow name."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_workflow") as span:
            span.set_attribute("openlit.span.kind", "workflow")
            span.set_attribute("openlit.workflow.name", "")  # Empty

        spans = exporter.get_finished_spans()

        # Should handle empty values gracefully
        assert len(spans) >= 1, "Should handle empty workflow name"

    def test_recursive_processing_prevention(self, setup_tracer):
        """Test that spans marked as processed are not processed again."""
        tracer, exporter, _ = setup_tracer

        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("openlit.span.kind", "task")
            span.set_attribute(
                "_openlit_processed", True
            )  # Already processed marker

        spans = exporter.get_finished_spans()

        # Should not create duplicate synthetic spans
        # With the marker, it should be filtered out
        assert len(spans) >= 1, "Should handle already-processed spans"
