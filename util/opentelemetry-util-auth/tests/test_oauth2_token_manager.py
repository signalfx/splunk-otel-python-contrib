# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for OAuth2TokenManager."""

import os
import time
from unittest.mock import patch

import pytest
import responses

from opentelemetry.util.auth import OAuth2TokenManager


@pytest.fixture
def oauth2_env_vars(monkeypatch):
    """Set OAuth2 environment variables for testing."""
    monkeypatch.setenv("LLM_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("LLM_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("LLM_TOKEN_URL", "https://auth.example.com/oauth2/token")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")


class TestOAuth2TokenManager:
    """Test OAuth2TokenManager functionality."""

    def test_initialization_with_env_vars(self, oauth2_env_vars):
        """Test initialization using environment variables."""
        manager = OAuth2TokenManager()
        assert manager.client_id == "test-client-id"
        assert manager.client_secret == "test-client-secret"
        assert manager.token_url == "https://auth.example.com/oauth2/token"
        assert manager.auth_method == "basic"

    def test_initialization_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        manager = OAuth2TokenManager(
            client_id="explicit-client-id",
            client_secret="explicit-secret",
            token_url="https://explicit.example.com/token",
            auth_method="post",
        )
        assert manager.client_id == "explicit-client-id"
        assert manager.auth_method == "post"

    def test_missing_credentials_raises_error(self):
        """Test that missing credentials raise ValueError."""
        with pytest.raises(ValueError, match="OAuth2 credentials required"):
            OAuth2TokenManager(client_id="only-id")

    def test_missing_token_url_raises_error(self):
        """Test that missing token URL raises ValueError."""
        with pytest.raises(ValueError, match="OAuth2 token URL required"):
            OAuth2TokenManager(
                client_id="test-id",
                client_secret="test-secret",
            )

    def test_invalid_auth_method_raises_error(self):
        """Test that invalid auth method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid auth_method"):
            OAuth2TokenManager(
                client_id="test-id",
                client_secret="test-secret",
                token_url="https://example.com/token",
                auth_method="invalid",  # type: ignore
            )

    @responses.activate
    def test_get_token_basic_auth(self, oauth2_env_vars):
        """Test token retrieval using Basic Auth."""
        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"access_token": "test-token-123", "expires_in": 3600},
            status=200,
        )

        manager = OAuth2TokenManager()
        token = manager.get_token()

        assert token == "test-token-123"
        assert manager.is_token_valid()
        assert len(responses.calls) == 1

    @responses.activate
    def test_get_token_post_body(self, oauth2_env_vars, monkeypatch):
        """Test token retrieval using POST body."""
        monkeypatch.setenv("LLM_AUTH_METHOD", "post")

        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"access_token": "test-token-456", "expires_in": 1800},
            status=200,
        )

        manager = OAuth2TokenManager()
        token = manager.get_token()

        assert token == "test-token-456"
        assert len(responses.calls) == 1

    @responses.activate
    def test_token_caching(self, oauth2_env_vars):
        """Test that tokens are cached and reused."""
        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"access_token": "cached-token", "expires_in": 3600},
            status=200,
        )

        manager = OAuth2TokenManager()

        # First call should hit the API
        token1 = manager.get_token()
        assert token1 == "cached-token"
        assert len(responses.calls) == 1

        # Second call should use cached token
        token2 = manager.get_token()
        assert token2 == "cached-token"
        assert len(responses.calls) == 1  # Still only 1 API call

    @responses.activate
    def test_token_refresh_before_expiry(self, oauth2_env_vars):
        """Test that token refreshes before expiry buffer."""
        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"access_token": "token-1", "expires_in": 600},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"access_token": "token-2", "expires_in": 600},
            status=200,
        )

        # Use 600 second buffer (same as token expiry)
        manager = OAuth2TokenManager(token_refresh_buffer_seconds=600)

        token1 = manager.get_token()
        assert token1 == "token-1"

        # Token should be considered expired due to buffer
        token2 = manager.get_token()
        assert token2 == "token-2"
        assert len(responses.calls) == 2

    def test_invalidate_token(self, oauth2_env_vars):
        """Test token invalidation."""
        manager = OAuth2TokenManager()
        manager._token = "some-token"
        manager._token_expiry = time.time() + 3600

        assert manager.is_token_valid()

        manager.invalidate()

        assert not manager.is_token_valid()
        assert manager._token is None

    def test_time_until_expiry(self, oauth2_env_vars):
        """Test time_until_expiry property."""
        manager = OAuth2TokenManager()
        manager._token = "test-token"
        manager._token_expiry = time.time() + 1000

        remaining = manager.time_until_expiry
        assert 995 < remaining < 1005  # Allow for execution time

    def test_get_llm_base_url_with_model(self, oauth2_env_vars):
        """Test get_llm_base_url with model."""
        url = OAuth2TokenManager.get_llm_base_url("gpt-4o-mini")
        assert url == "https://api.example.com/v1/gpt-4o-mini"

    def test_get_llm_base_url_without_model(self, oauth2_env_vars):
        """Test get_llm_base_url without model."""
        url = OAuth2TokenManager.get_llm_base_url()
        assert url == "https://api.example.com/v1"

    def test_get_llm_base_url_missing_env_var(self, monkeypatch):
        """Test get_llm_base_url raises error when env var missing."""
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="LLM_BASE_URL environment variable is required"):
            OAuth2TokenManager.get_llm_base_url()

    @responses.activate
    def test_oauth2_error_handling(self, oauth2_env_vars):
        """Test error handling for OAuth2 failures."""
        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"error": "invalid_client"},
            status=401,
        )

        manager = OAuth2TokenManager()

        with pytest.raises(Exception):
            manager.get_token()

    @responses.activate
    def test_missing_access_token_in_response(self, oauth2_env_vars):
        """Test handling of invalid OAuth2 response."""
        responses.add(
            responses.POST,
            "https://auth.example.com/oauth2/token",
            json={"token_type": "Bearer", "expires_in": 3600},  # Missing access_token
            status=200,
        )

        manager = OAuth2TokenManager()

        with pytest.raises(ValueError, match="missing 'access_token' field"):
            manager.get_token()

