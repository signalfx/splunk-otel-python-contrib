"""
Test LlamaIndex RAG instrumentation without agents.

Validates that QUERY / RETRIEVE / SYNTHESIZE callback events
produce spans when using mock LLM and embeddings.
"""

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.mock import MockLLM


def test_rag_without_agents(span_exporter, instrument):
    """Test RAG query produces spans."""
    Settings.llm = MockLLM(max_tokens=64)
    Settings.embed_model = MockEmbedding(embed_dim=8)

    documents = [
        Document(
            text="Paris is the capital of France.",
            metadata={"source": "geography"},
        ),
        Document(
            text="The Eiffel Tower is in Paris.",
            metadata={"source": "landmarks"},
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("What is the capital of France?")

    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    # Verify retrieval span has correct operation name per semconv
    retrieval_spans = [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "retrieval"
    ]
    assert len(retrieval_spans) == 1
    retrieval_span = retrieval_spans[0]
    assert retrieval_span.name == "retrieval"
    assert (
        retrieval_span.attributes.get("gen_ai.retrieval.type") == "llamaindex_retriever"
    )
