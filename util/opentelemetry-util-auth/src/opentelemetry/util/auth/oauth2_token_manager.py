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

"""OAuth2 Token Manager for LLM application authentication."""

import base64
import logging
import os
import time
from typing import Literal, Optional

import requests

_LOGGER = logging.getLogger(__name__)

AuthMethod = Literal["basic", "post"]


class OAuth2TokenManager:
    """
    Manages OAuth2 tokens for LLM endpoints using client credentials flow.

    Automatically refreshes tokens before expiry and supports multiple
    authentication methods (Basic Auth and POST body).

    Usage:
        >>> from opentelemetry.util.auth import OAuth2TokenManager
        >>>
        >>> # Using environment variables
        >>> token_manager = OAuth2TokenManager()
        >>> token = token_manager.get_token()
        >>>
        >>> # Passing credentials explicitly
        >>> token_manager = OAuth2TokenManager(
        ...     client_id="your-client-id",
        ...     client_secret="your-client-secret",
        ...     token_url="https://your-idp.com/oauth2/token"
        ... )

    Environment Variables:
        LLM_CLIENT_ID: OAuth2 client ID (required)
        LLM_CLIENT_SECRET: OAuth2 client secret (required)
        LLM_TOKEN_URL: OAuth2 token endpoint URL (required)
        LLM_AUTH_METHOD: Authentication method - "basic" or "post" (default: "basic")
        LLM_SCOPE: OAuth2 scope (optional)
        LLM_BASE_URL: LLM endpoint base URL (optional, for get_llm_base_url())

    Example with OAuth2 POST method (Azure AD):
        >>> os.environ["LLM_AUTH_METHOD"] = "post"
        >>> os.environ["LLM_SCOPE"] = "api://your-resource/.default"
        >>> token_manager = OAuth2TokenManager()
        >>> token = token_manager.get_token()
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: Optional[str] = None,
        auth_method: Optional[AuthMethod] = None,
        scope: Optional[str] = None,
        token_refresh_buffer_seconds: int = 300,
    ):
        """
        Initialize the OAuth2 token manager.

        Args:
            client_id: OAuth2 client ID (or use LLM_CLIENT_ID env var)
            client_secret: OAuth2 client secret (or use LLM_CLIENT_SECRET env var)
            token_url: Token endpoint URL (or use LLM_TOKEN_URL env var)
            auth_method: "basic" for Basic Auth header, "post" for POST body
                        (or use LLM_AUTH_METHOD env var, default: "basic")
            scope: OAuth2 scope (or use LLM_SCOPE env var, optional)
            token_refresh_buffer_seconds: Refresh token this many seconds before expiry
        """
        self.client_id = client_id or os.environ.get("LLM_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("LLM_CLIENT_SECRET")
        self.token_url = token_url or os.environ.get("LLM_TOKEN_URL")
        self.auth_method: AuthMethod = (
            auth_method
            or os.environ.get("LLM_AUTH_METHOD", "basic").lower()  # type: ignore
        )
        self.scope = scope or os.environ.get("LLM_SCOPE")
        self.token_refresh_buffer = token_refresh_buffer_seconds

        # Validate required parameters
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "OAuth2 credentials required. "
                "Set client_id/client_secret or LLM_CLIENT_ID/LLM_CLIENT_SECRET env vars."
            )

        if not self.token_url:
            raise ValueError(
                "OAuth2 token URL required. "
                "Set token_url or LLM_TOKEN_URL env var."
            )

        if self.auth_method not in ("basic", "post"):
            raise ValueError(
                f"Invalid auth_method '{self.auth_method}'. "
                "Must be 'basic' or 'post'."
            )

        self._token: Optional[str] = None
        self._token_expiry: float = 0

        _LOGGER.debug(
            "OAuth2TokenManager initialized with auth_method=%s, token_url=%s",
            self.auth_method,
            self.token_url,
        )

    def get_token(self) -> str:
        """
        Get a valid access token, refreshing if needed.

        Returns:
            Valid OAuth2 access token (JWT)

        Raises:
            requests.RequestException: If token request fails
            ValueError: If token response is invalid
        """
        if self._token and time.time() < (
            self._token_expiry - self.token_refresh_buffer
        ):
            _LOGGER.debug("Using cached token (expires in %.0f seconds)",
                         self._token_expiry - time.time())
            return self._token

        _LOGGER.debug("Token expired or not cached, refreshing...")
        return self._refresh_token()

    def _refresh_token(self) -> str:
        """Request a new token from the OAuth2 endpoint."""
        try:
            if self.auth_method == "basic":
                return self._refresh_token_basic_auth()
            else:
                return self._refresh_token_post_body()
        except requests.RequestException as e:
            _LOGGER.error("OAuth2 token refresh failed: %s", e)
            raise
        except (KeyError, ValueError) as e:
            _LOGGER.error("Invalid OAuth2 token response: %s", e)
            raise ValueError(f"Invalid OAuth2 token response: {e}") from e

    def _refresh_token_basic_auth(self) -> str:
        """Request token using Basic Authentication (most common)."""
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        data = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope

        _LOGGER.debug("Requesting OAuth2 token (Basic Auth) from %s", self.token_url)

        response = requests.post(
            self.token_url,
            headers={
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            data=data,
            timeout=30,
        )
        response.raise_for_status()

        return self._extract_token_from_response(response)

    def _refresh_token_post_body(self) -> str:
        """Request token using POST body (Azure AD, some enterprise IdPs)."""
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope

        _LOGGER.debug("Requesting OAuth2 token (POST body) from %s", self.token_url)

        response = requests.post(
            self.token_url,
            headers={
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=data,
            timeout=30,
        )
        response.raise_for_status()

        return self._extract_token_from_response(response)

    def _extract_token_from_response(self, response: requests.Response) -> str:
        """Extract and cache token from OAuth2 response."""
        token_data = response.json()

        if "access_token" not in token_data:
            raise ValueError("OAuth2 response missing 'access_token' field")

        self._token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.time() + expires_in

        _LOGGER.info("OAuth2 token refreshed successfully (expires in %ds)", expires_in)
        _LOGGER.debug("New token expiry timestamp: %.0f", self._token_expiry)

        return self._token

    def invalidate(self) -> None:
        """Force token refresh on next get_token() call."""
        _LOGGER.debug("Invalidating cached token")
        self._token = None
        self._token_expiry = 0

    def is_token_valid(self) -> bool:
        """
        Check if current token is still valid (not expired).

        Returns:
            True if token is cached and not expired, False otherwise
        """
        valid = bool(
            self._token
            and time.time() < (self._token_expiry - self.token_refresh_buffer)
        )
        _LOGGER.debug("Token valid: %s", valid)
        return valid

    @property
    def token_expires_at(self) -> float:
        """
        Unix timestamp when the current token expires.

        Returns:
            Token expiry timestamp (0 if no token cached)
        """
        return self._token_expiry

    @property
    def time_until_expiry(self) -> float:
        """
        Seconds until the current token expires.

        Returns:
            Seconds until expiry (0 if token is expired or not cached)
        """
        if not self._token:
            return 0
        remaining = self._token_expiry - time.time()
        return max(0, remaining)

    @classmethod
    def get_llm_base_url(cls, model: Optional[str] = None) -> str:
        """
        Get the LLM base URL, optionally with model appended.

        Args:
            model: Model name to append (e.g., "gpt-4o-mini"). If None,
                   returns base URL as-is.

        Returns:
            Full base URL for the model endpoint

        Raises:
            ValueError: If LLM_BASE_URL environment variable is not set

        Example:
            >>> OAuth2TokenManager.get_llm_base_url("gpt-4o-mini")
            'https://api.example.com/v1/gpt-4o-mini'
        """
        base = os.environ.get("LLM_BASE_URL")
        if not base:
            raise ValueError("LLM_BASE_URL environment variable is required")

        if model:
            return f"{base.rstrip('/')}/{model}"
        return base


__all__ = ["OAuth2TokenManager"]

