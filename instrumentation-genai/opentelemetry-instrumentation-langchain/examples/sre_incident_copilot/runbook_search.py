"""Runbook search using RAG (Retrieval Augmented Generation)."""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_loader import DataLoader


class RunbookSearch:
    """RAG-based runbook search with citations."""

    def __init__(self, data_dir: str = "data", persist_dir: Optional[str] = None):
        """Initialize runbook search.

        Args:
            data_dir: Directory containing runbook files
            persist_dir: Directory to persist vector store (None for in-memory)
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.data_loader = DataLoader(data_dir)
        
        # Azure OpenAI Embeddings configuration
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_deployment=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT"),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or load vector store."""
        if self.persist_dir and Path(self.persist_dir).exists():
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
        else:
            # Create new vector store from runbooks
            documents = self._load_runbook_documents()
            if documents:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                )
            else:
                # Create empty vector store
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir,
                )

    def _load_runbook_documents(self) -> List[Document]:
        """Load all runbooks as documents."""
        documents = []
        runbooks_dir = self.data_dir / "runbooks"

        if not runbooks_dir.exists():
            return documents

        for runbook_file in runbooks_dir.glob("*.md"):
            runbook_name = runbook_file.stem
            content = self.data_loader.load_runbook(runbook_name)
            if content:
                # Split into chunks
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": runbook_name,
                            "runbook_file": str(runbook_file),
                            "chunk_index": i,
                        },
                    )
                    documents.append(doc)

        return documents

    def search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[dict]:
        """Search runbooks for relevant sections.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with runbook sections and citations
        """
        if not self.vector_store:
            return []

        # Perform similarity search
        if score_threshold:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # Filter by threshold
            filtered_results = [
                (doc, score) for doc, score in results if score >= score_threshold
            ]
            results = filtered_results[:k]
        else:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            results = results_with_scores

        # Format results with citations
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "runbook_file": doc.metadata.get("runbook_file", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "similarity_score": float(score),
                    "citation": f"runbook:{doc.metadata.get('source', 'unknown')}#chunk-{doc.metadata.get('chunk_index', 0)}",
                }
            )

        return formatted_results

    def search_by_topic(self, topic: str, k: int = 3) -> List[dict]:
        """Search runbooks by topic (e.g., 'database', 'cache', 'deployment').

        Args:
            topic: Topic keyword
            k: Number of results

        Returns:
            List of relevant runbook sections
        """
        # Enhance query with topic context
        query = f"{topic} incident response troubleshooting mitigation"
        return self.search(query, k=k)


def create_runbook_search_tool(runbook_search: RunbookSearch):
    """Create a LangChain tool for runbook search."""
    from langchain_core.tools import tool

    @tool
    def runbook_search(query: str, k: int = 3) -> str:
        """Search runbooks for relevant incident response procedures.

        Args:
            query: Search query describing the incident or issue
            k: Number of results to return (default: 3)

        Returns:
            JSON string with runbook sections and citations
        """
        import json

        results = runbook_search.search(query, k=k)
        return json.dumps(results, indent=2)

    return runbook_search
