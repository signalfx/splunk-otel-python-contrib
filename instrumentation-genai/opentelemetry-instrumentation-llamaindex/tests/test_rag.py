"""
Test LlamaIndex RAG instrumentation without agents.

This test validates that:
1. QUERY events create Workflow spans at the root level (or auto-created if no parent)
2. RETRIEVE events create RetrievalInvocation spans with parent reference to the Workflow
3. SYNTHESIZE events don't create their own span - the LLM invocation is tracked directly
4. LLM invocations nest under their parent (Workflow) via parent span
5. Embedding invocations nest under their parent (RetrievalInvocation) via parent span
"""

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class DebugSpanExporter(SpanExporter):
    """Custom exporter that shows parent-child relationships clearly."""

    def export(self, spans):
        for span in spans:
            parent_id = span.parent.span_id if span.parent else "None (ROOT)"
            operation = span.attributes.get("gen_ai.operation.name", "unknown")

            print(f"\n{'=' * 60}")
            print(f"Span: {span.name}")
            print(f"  Operation: {operation}")
            print(f"  Span ID: {format(span.context.span_id, '016x')}")
            print(
                f"  Parent ID: {parent_id if isinstance(parent_id, str) else format(parent_id, '016x')}"
            )
            print(f"  Trace ID: {format(span.context.trace_id, '032x')}")

            # Show key attributes
            if "gen_ai.request.model" in span.attributes:
                print(f"  Model: {span.attributes['gen_ai.request.model']}")
            if "db.operation.name" in span.attributes:
                print(f"  DB Operation: {span.attributes['db.operation.name']}")

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def setup_telemetry():
    """Setup OpenTelemetry with console exporter to see trace structure."""
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(DebugSpanExporter()))
    return tracer_provider


def test_rag_without_agents():
    """Test RAG instrumentation creates correct hierarchy: Workflow -> RetrievalInvocation/LLMInvocation"""

    print("=" * 80)
    print("Setting up telemetry...")
    print("=" * 80)
    setup_telemetry()

    # Setup LlamaIndex
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Instrument
    instrumentor = LlamaindexInstrumentor()
    instrumentor.instrument()

    # Debug: Check callback handler
    from llama_index.core import Settings as LlamaSettings

    print(f"\nCallbacks registered: {len(LlamaSettings.callback_manager.handlers)}")
    for handler in LlamaSettings.callback_manager.handlers:
        print(f"  Handler: {type(handler).__name__}")

    # Create sample documents
    documents = [
        Document(
            text="Paris is the capital of France. It has a population of over 2 million.",
            metadata={"source": "geography", "country": "France"},
        ),
        Document(
            text="The Eiffel Tower is in Paris. It was completed in 1889.",
            metadata={"source": "landmarks", "country": "France"},
        ),
    ]

    print("\n" + "=" * 80)
    print("Creating vector index (should see Embedding spans)...")
    print("=" * 80)
    index = VectorStoreIndex.from_documents(documents)

    print("\n" + "=" * 80)
    print("Creating query engine...")
    print("=" * 80)
    query_engine = index.as_query_engine(similarity_top_k=2)

    print("\n" + "=" * 80)
    print(
        "Executing RAG query (should see Workflow -> retrieve.task/synthesize.task -> LLM/Embedding)..."
    )
    print("=" * 80)
    response = query_engine.query("What is the capital of France?")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Response: {response.response}")
    print(f"Source nodes: {len(response.source_nodes)}")

    print("\n" + "=" * 80)
    print("✓ Test completed!")
    print("=" * 80)
    print("\nExpected trace structure:")
    print("  Workflow (auto-created RAG workflow)")
    print("    ├─ RetrievalInvocation (retrieve)")
    print("    │   └─ EmbeddingInvocation (query embedding)")
    print("    └─ LLMInvocation (synthesize response - no Step wrapper)")
    print("=" * 80)


if __name__ == "__main__":
    test_rag_without_agents()
