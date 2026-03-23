"""Test embedding instrumentation for LlamaIndex."""

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import MockEmbedding


def test_embedding_single_text(span_exporter, instrument):
    """Test single text embedding produces spans."""
    embed_model = MockEmbedding(embed_dim=8)
    Settings.embed_model = embed_model
    if Settings.callback_manager is None:
        Settings.callback_manager = CallbackManager()

    embedding = embed_model.get_text_embedding(
        "LlamaIndex is a data framework for LLM applications"
    )

    assert len(embedding) == 8

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1


def test_embedding_batch(span_exporter, instrument):
    """Test batch embedding produces spans."""
    embed_model = MockEmbedding(embed_dim=8)
    Settings.embed_model = embed_model
    if Settings.callback_manager is None:
        Settings.callback_manager = CallbackManager()

    texts = [
        "Paris is the capital of France",
        "Berlin is the capital of Germany",
        "Rome is the capital of Italy",
    ]
    embeddings = embed_model.get_text_embedding_batch(texts)

    assert len(embeddings) == 3
    assert all(len(e) == 8 for e in embeddings)

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
