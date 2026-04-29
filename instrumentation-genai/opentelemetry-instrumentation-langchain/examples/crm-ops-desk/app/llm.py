"""LLM and embedding factories with Azure OpenAI support.

Reads environment variables to decide between standard OpenAI and Azure OpenAI:

  Standard OpenAI (default):
    OPENAI_API_KEY          — API key
    OPENAI_MODEL            — chat model name (default: gpt-4o-mini)
    OPENAI_EMBEDDING_MODEL  — embedding model name (default: text-embedding-3-small)

  Azure via OpenAI-compatible endpoint (preferred — model name reported correctly):
    OPENAI_BASE_URL         — e.g. https://my-resource.openai.azure.com/openai/v1/
    OPENAI_API_KEY          — Azure API key
    OPENAI_MODEL            — model / deployment name (default: gpt-4o-mini)

  Azure native SDK (when AZURE_OPENAI_ENDPOINT is set, no OPENAI_BASE_URL):
    AZURE_OPENAI_ENDPOINT           — e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY            — Azure API key
    AZURE_OPENAI_API_VERSION        — e.g. 2024-12-01-preview
    AZURE_CHAT_DEPLOYMENT           — deployment name for the chat model
    AZURE_EMBEDDING_DEPLOYMENT      — deployment name for embeddings
    AZURE_EMBEDDING_DIMENSIONS      — output dimensions (default: 1536, for FAISS compat)
"""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI


def _is_azure() -> bool:
    """Return True if Azure OpenAI credentials are configured."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))


def create_chat_llm(*, temperature: float = 0.3) -> ChatOpenAI | AzureChatOpenAI:
    """Create a ChatOpenAI (or Azure variant) from environment variables.

    Preference order:
    1. OPENAI_BASE_URL set → ChatOpenAI with base_url (OpenAI-compatible Azure)
    2. AZURE_OPENAI_ENDPOINT set → AzureChatOpenAI (native Azure SDK)
    3. Neither → plain ChatOpenAI (standard OpenAI)
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        # OpenAI-compatible endpoint (e.g. Azure .openai.azure.com/openai/v1/)
        # This ensures gen_ai.request.model is reported correctly in telemetry.
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
            temperature=temperature,
        )
    if _is_azure():
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
            azure_deployment=os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            temperature=temperature,
        )
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via OpenAI or Azure OpenAI."""
    import openai

    if _is_azure():
        client = openai.AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
        )
        deployment = os.environ.get(
            "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )
        dimensions = int(os.environ.get("AZURE_EMBEDDING_DIMENSIONS", "1536"))
        resp = client.embeddings.create(
            input=texts, model=deployment, dimensions=dimensions
        )
    else:
        client = openai.OpenAI()
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(input=texts, model=model)

    return [d.embedding for d in resp.data]
