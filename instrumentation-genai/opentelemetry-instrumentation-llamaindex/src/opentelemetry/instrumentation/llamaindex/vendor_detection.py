"""Vendor detection for LlamaIndex embedding providers."""

from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class VendorRule:
    """Rule for detecting vendor from LlamaIndex class names."""

    exact_matches: Set[str]
    patterns: List[str]
    vendor_name: str

    def matches(self, class_name: str) -> bool:
        """Check if class name matches this vendor rule."""
        if class_name in self.exact_matches:
            return True
        class_lower = class_name.lower()
        return any(pattern in class_lower for pattern in self.patterns)


def _get_vendor_rules() -> List[VendorRule]:
    """
    Get vendor detection rules ordered by specificity (most specific first).

    Returns:
        List of VendorRule objects for detecting embedding vendors from class names
    """
    return [
        VendorRule(
            exact_matches={"AzureOpenAIEmbedding"},
            patterns=["azure"],
            vendor_name="azure",
        ),
        VendorRule(
            exact_matches={"OpenAIEmbedding"},
            patterns=["openai"],
            vendor_name="openai",
        ),
        VendorRule(
            exact_matches={"BedrockEmbedding"},
            patterns=["bedrock", "aws"],
            vendor_name="aws",
        ),
        VendorRule(
            exact_matches={
                "VertexTextEmbedding",
                "GeminiEmbedding",
                "GooglePaLMEmbedding",
            },
            patterns=["vertex", "google", "palm", "gemini"],
            vendor_name="google",
        ),
        VendorRule(
            exact_matches={"CohereEmbedding"},
            patterns=["cohere"],
            vendor_name="cohere",
        ),
        VendorRule(
            exact_matches={"HuggingFaceEmbedding", "HuggingFaceInferenceAPIEmbedding"},
            patterns=["huggingface"],
            vendor_name="huggingface",
        ),
        VendorRule(
            exact_matches={"OllamaEmbedding"},
            patterns=["ollama"],
            vendor_name="ollama",
        ),
        VendorRule(
            exact_matches={"AnthropicEmbedding"},
            patterns=["anthropic"],
            vendor_name="anthropic",
        ),
        VendorRule(
            exact_matches={"MistralAIEmbedding"},
            patterns=["mistral"],
            vendor_name="mistralai",
        ),
        VendorRule(
            exact_matches={"TogetherEmbedding"},
            patterns=["together"],
            vendor_name="together",
        ),
        VendorRule(
            exact_matches={"FireworksEmbedding"},
            patterns=["fireworks"],
            vendor_name="fireworks",
        ),
        VendorRule(
            exact_matches={"VoyageEmbedding"},
            patterns=["voyage"],
            vendor_name="voyage",
        ),
        VendorRule(
            exact_matches={"JinaEmbedding"},
            patterns=["jina"],
            vendor_name="jina",
        ),
    ]


def detect_vendor_from_class(class_name: str) -> str:
    """
    Detect vendor from LlamaIndex embedding class name.
    Uses unified detection rules combining exact matches and patterns.

    Args:
        class_name: The class name from serialized embedding information

    Returns:
        Vendor string (lowercase), defaults to None if no match found
    """
    if not class_name:
        return None

    vendor_rules = _get_vendor_rules()

    for rule in vendor_rules:
        if rule.matches(class_name):
            return rule.vendor_name

    return None
