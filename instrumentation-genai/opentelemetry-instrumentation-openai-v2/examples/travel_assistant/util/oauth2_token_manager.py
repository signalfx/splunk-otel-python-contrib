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

"""OAuth2 Token Manager for OpenAI-compatible LLM endpoints."""

import base64
import os
import time
from typing import Optional

import requests


class OAuth2TokenManager:
    """
    Manages OAuth2 tokens for LLM endpoints.

    Uses client credentials flow to obtain and refresh access tokens
    for use with OpenAI-compatible APIs.

    Usage:
        from util import OAuth2TokenManager

        token_manager = OAuth2TokenManager()  # Uses env vars
        token = token_manager.get_token()

    Environment Variables:
        LLM_CLIENT_ID: OAuth2 client ID (required)
        LLM_CLIENT_SECRET: OAuth2 client secret (required)
        LLM_TOKEN_URL: Token endpoint (default: https://id.cisco.com/oauth2/default/v1/token)
        LLM_BASE_URL: LLM endpoint base (default: https://chat-ai.cisco.com/openai/deployments)
    """

    DEFAULT_TOKEN_URL = "https://id.cisco.com/oauth2/default/v1/token"
    DEFAULT_LLM_BASE_URL = "https://chat-ai.cisco.com/openai/deployments"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: Optional[str] = None,
        token_refresh_buffer_seconds: int = 300,
    ):
        """
        Initialize the token manager.

        Args:
            client_id: OAuth2 client ID (or use LLM_CLIENT_ID env var)
            client_secret: OAuth2 client secret (or use LLM_CLIENT_SECRET env var)
            token_url: Token endpoint URL (or use LLM_TOKEN_URL env var)
            token_refresh_buffer_seconds: Refresh token this many seconds before expiry
        """
        self.client_id = client_id or os.environ.get("LLM_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("LLM_CLIENT_SECRET")
        self.token_url = (
            token_url or os.environ.get("LLM_TOKEN_URL") or self.DEFAULT_TOKEN_URL
        )
        self.token_refresh_buffer = token_refresh_buffer_seconds

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "OAuth2 credentials required. "
                "Set client_id/client_secret or LLM_CLIENT_ID/LLM_CLIENT_SECRET env vars."
            )

        self._token: Optional[str] = None
        self._token_expiry: float = 0

    def get_token(self) -> str:
        """
        Get a valid access token, refreshing if needed.

        Returns:
            Valid OAuth2 access token (JWT)

        Raises:
            requests.RequestException: If token request fails
        """
        if self._token and time.time() < (
            self._token_expiry - self.token_refresh_buffer
        ):
            return self._token

        return self._refresh_token()

    def _refresh_token(self) -> str:
        """Request a new token from the OAuth2 endpoint."""
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        response = requests.post(
            self.token_url,
            headers={
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            data="grant_type=client_credentials",
            timeout=30,
        )
        response.raise_for_status()

        token_data = response.json()
        self._token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.time() + expires_in

        return self._token

    def invalidate(self) -> None:
        """Force token refresh on next get_token() call."""
        self._token = None
        self._token_expiry = 0

    def is_token_valid(self) -> bool:
        """Check if current token is still valid."""
        return bool(
            self._token
            and time.time() < (self._token_expiry - self.token_refresh_buffer)
        )

    @property
    def token_expires_at(self) -> float:
        """Unix timestamp when token expires."""
        return self._token_expiry

    @classmethod
    def get_llm_base_url(cls, model: str = "gpt-4o-mini") -> str:
        """
        Get the LLM base URL for a given model.

        Args:
            model: Model name (e.g., "gpt-4o-mini")

        Returns:
            Full base URL for the model endpoint
        """
        base = os.environ.get("LLM_BASE_URL") or cls.DEFAULT_LLM_BASE_URL
        return f"{base}/{model}"
