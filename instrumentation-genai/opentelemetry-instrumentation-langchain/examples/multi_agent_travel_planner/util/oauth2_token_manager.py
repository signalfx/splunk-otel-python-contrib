"""OAuth2 Token Manager for LLM integration."""

import base64
import os
import time
from typing import Optional

import requests


class OAuth2TokenManager:
    """
    Manages OAuth2 tokens for custom LLM endpoints.

    This class handles OAuth2 client credentials flow authentication,
    including automatic token refresh when tokens expire.

    Environment Variables:
        LLM_CLIENT_ID: OAuth2 client ID
        LLM_CLIENT_SECRET: OAuth2 client secret
        LLM_TOKEN_URL: OAuth2 token endpoint URL
        LLM_BASE_URL: Base URL for the LLM API
        LLM_APP_KEY: Optional application key for request tracking

    Example:
        >>> token_manager = OAuth2TokenManager()
        >>> token = token_manager.get_token()
        >>> base_url = OAuth2TokenManager.get_llm_base_url("gpt-4o-mini")
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: Optional[str] = None,
    ):
        """
        Initialize the OAuth2 token manager.

        Args:
            client_id: OAuth2 client ID. Defaults to LLM_CLIENT_ID env var.
            client_secret: OAuth2 client secret. Defaults to LLM_CLIENT_SECRET env var.
            token_url: OAuth2 token endpoint URL. Defaults to LLM_TOKEN_URL env var.

        Raises:
            ValueError: If required credentials are not provided.
        """
        self.client_id = client_id or os.environ.get("LLM_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("LLM_CLIENT_SECRET")
        self.token_url = token_url or os.environ.get("LLM_TOKEN_URL")

        if not all([self.client_id, self.client_secret, self.token_url]):
            raise ValueError(
                "Missing required OAuth2 credentials. "
                "Set LLM_CLIENT_ID, LLM_CLIENT_SECRET, and LLM_TOKEN_URL environment variables."
            )

        self._token: Optional[str] = None
        self._token_expiry: float = 0

    @staticmethod
    def get_llm_base_url(model_name: str) -> str:
        """
        Construct the LLM base URL for a given model.

        Args:
            model_name: Name of the model (e.g., "gpt-4o-mini")

        Returns:
            Full base URL for the LLM API endpoint.

        Raises:
            ValueError: If LLM_BASE_URL environment variable is not set.
        """
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise ValueError("LLM_BASE_URL environment variable is not set")
        return f"{base_url}/{model_name}"

    def _refresh_token(self) -> None:
        """Refresh the OAuth2 token using client credentials flow."""
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}

        # token_url is validated in __init__, assert for type checker
        assert self.token_url is not None
        response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()

        token_data = response.json()
        self._token = token_data["access_token"]
        # Set expiry with 60-second buffer
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.time() + expires_in - 60

    def get_token(self) -> str:
        """
        Get a valid OAuth2 token, refreshing if necessary.

        Returns:
            Valid OAuth2 access token.
        """
        if self._token is None or time.time() >= self._token_expiry:
            self._refresh_token()
        return str(self._token)

    def is_token_valid(self) -> bool:
        """
        Check if the current token is valid and not expired.

        Returns:
            True if token is valid, False otherwise.
        """
        return self._token is not None and time.time() < self._token_expiry
