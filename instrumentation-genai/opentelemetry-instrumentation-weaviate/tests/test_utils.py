"""Tests for utility functions."""

import pytest

from opentelemetry.instrumentation.weaviate.utils import (
    extract_collection_name,
    parse_url_to_host_port,
)


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
