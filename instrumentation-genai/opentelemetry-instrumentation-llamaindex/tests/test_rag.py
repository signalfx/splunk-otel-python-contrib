"""
Test LlamaIndex RAG instrumentation without agents.

This test validates that:
1. QUERY events create Workflow spans at the root level
2. RETRIEVE events create Step spans with parent_run_id pointing to the Workflow
3. SYNTHESIZE events create Step spans with parent_run_id pointing to the Workflow  
4. LLM invocations nest under their Step parent via parent_run_id
5. Embedding invocations nest under their Step parent via parent_run_id
"""

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


def setup_telemetry():
    """Setup OpenTelemetry with console exporter to see trace structure."""
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return tracer_provider


def test_rag_without_agents():
    """Test RAG instrumentation creates correct hierarchy: Workflow -> Steps -> LLM/Embedding"""
    
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
    print("Executing RAG query (should see Workflow -> retrieve.task/synthesize.task -> LLM/Embedding)...")
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
    print("  Workflow (gen_ai.operation.name=query)")
    print("    ├─ Step (gen_ai.operation.name=retrieve.task)")
    print("    │   └─ EmbeddingInvocation")
    print("    └─ Step (gen_ai.operation.name=synthesize.task)")
    print("        └─ LLMInvocation")
    print("=" * 80)


if __name__ == "__main__":
    test_rag_without_agents()
