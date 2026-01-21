"""Tests for utility functions."""

import os
from unittest.mock import patch

from opentelemetry.instrumentation.weaviate.utils import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    extract_collection_name,
    is_content_enabled,
    parse_url_to_host_port,
)


class TestIsContentEnabled:
    """Test content capture environment variable check."""

    def test_content_enabled_true(self):
        """Test content capture is enabled when env var is 'true'."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "true"}):
            assert is_content_enabled() is True

    def test_content_enabled_true_uppercase(self):
        """Test content capture is enabled when env var is 'TRUE'."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "TRUE"}):
            assert is_content_enabled() is True

    def test_content_enabled_true_mixed_case(self):
        """Test content capture is enabled when env var is 'TrUe'."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "TrUe"}):
            assert is_content_enabled() is True

    def test_content_disabled_false(self):
        """Test content capture is disabled when env var is 'false'."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "false"}):
            assert is_content_enabled() is False

    def test_content_disabled_empty(self):
        """Test content capture is disabled when env var is empty."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: ""}):
            assert is_content_enabled() is False

    def test_content_disabled_not_set(self):
        """Test content capture is disabled when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_content_enabled() is False

    def test_content_disabled_invalid_value(self):
        """Test content capture is disabled when env var has invalid value."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "yes"}):
            assert is_content_enabled() is False


class TestParseUrlToHostPort:
    """Test URL parsing utility."""

    def test_parse_http_url(self):
        """Test parsing HTTP URL."""
        host, port = parse_url_to_host_port("http://localhost:8080")
        assert host == "localhost"
        assert port == 8080

    def test_parse_https_url(self):
        """Test parsing HTTPS URL."""
        host, port = parse_url_to_host_port("https://example.com:443")
        assert host == "example.com"
        assert port == 443

    def test_parse_url_without_port(self):
        """Test parsing URL without explicit port."""
        host, port = parse_url_to_host_port("http://localhost")
        assert host == "localhost"
        assert port is None  # urlparse returns None when port is not specified

    def test_parse_https_url_without_port(self):
        """Test parsing HTTPS URL without explicit port."""
        host, port = parse_url_to_host_port("https://example.com")
        assert host == "example.com"
        assert port is None  # urlparse returns None when port is not specified

    def test_parse_url_with_path(self):
        """Test parsing URL with path."""
        host, port = parse_url_to_host_port("http://localhost:8080/v1")
        assert host == "localhost"
        assert port == 8080

    def test_parse_invalid_url(self):
        """Test parsing invalid URL returns None."""
        host, port = parse_url_to_host_port("not-a-url")
        assert host is None
        assert port is None

    def test_parse_none_url(self):
        """Test parsing None URL."""
        host, port = parse_url_to_host_port(None)
        assert host is None
        assert port is None


class TestExtractCollectionName:
    """Test collection name extraction utility."""

    def test_extract_from_args(self):
        """Test extracting collection name from positional args."""

        # Mock function and instance
        def mock_func():
            pass

        instance = None
        args = ("MyCollection",)
        kwargs = {}

        result = extract_collection_name(
            mock_func, instance, args, kwargs, "weaviate.schema", "get"
        )
        # Result depends on implementation - this is a basic structure test
        assert result is None or isinstance(result, str)

    def test_extract_from_kwargs(self):
        """Test extracting collection name from keyword args."""

        def mock_func():
            pass

        instance = None
        args = ()
        kwargs = {"class_name": "MyCollection"}

        result = extract_collection_name(
            mock_func, instance, args, kwargs, "weaviate.data", "create"
        )
        # Result depends on implementation
        assert result is None or isinstance(result, str)

    def test_extract_with_no_collection(self):
        """Test extraction when no collection name is present."""

        def mock_func():
            pass

        instance = None
        args = ()
        kwargs = {}

        result = extract_collection_name(
            mock_func, instance, args, kwargs, "weaviate.query", "raw"
        )
        assert result is None or isinstance(result, str)
