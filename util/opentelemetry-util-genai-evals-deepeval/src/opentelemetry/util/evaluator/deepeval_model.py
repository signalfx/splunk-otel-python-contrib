# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""OAuth2-authenticated LLM model for DeepEval via LiteLLM."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Optional

import requests

_logger = logging.getLogger(__name__)


class OAuth2TokenManager:
    """
    Lightweight OAuth2 token manager for custom LLM providers.

    Supports client_credentials grant type with Basic Auth or form POST.

    Environment Variables:
        DEEPEVAL_LLM_CLIENT_ID: OAuth2 client ID
        DEEPEVAL_LLM_CLIENT_SECRET: OAuth2 client secret
        DEEPEVAL_LLM_TOKEN_URL: OAuth2 token endpoint
        DEEPEVAL_LLM_GRANT_TYPE: Grant type (default: client_credentials)
        DEEPEVAL_LLM_SCOPE: OAuth2 scope (optional)
        DEEPEVAL_LLM_AUTH_METHOD: Token auth method - "basic" or "post" (default: basic)
    """

    def __init__(self) -> None:
        self.client_id = os.environ.get("DEEPEVAL_LLM_CLIENT_ID")
        self.client_secret = os.environ.get("DEEPEVAL_LLM_CLIENT_SECRET")
        self.token_url = os.environ.get("DEEPEVAL_LLM_TOKEN_URL")
        self.grant_type = os.environ.get(
            "DEEPEVAL_LLM_GRANT_TYPE", "client_credentials"
        )
        self.scope = os.environ.get("DEEPEVAL_LLM_SCOPE")
        self.auth_method = os.environ.get(
            "DEEPEVAL_LLM_AUTH_METHOD", "basic"
        ).lower()
        self._token: Optional[str] = None
        self._token_expiry: float = 0
        self._refresh_buffer: int = 300  # Refresh 5 min before expiry

    def get_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if self._token and time.time() < (
            self._token_expiry - self._refresh_buffer
        ):
            return self._token
        return self._refresh_token()

    def _refresh_token(self) -> str:
        """Request a new token from the OAuth2 endpoint."""
        if not self.token_url or not self.client_id or not self.client_secret:
            raise ValueError(
                "OAuth2 configuration incomplete. Required: "
                "DEEPEVAL_LLM_TOKEN_URL, DEEPEVAL_LLM_CLIENT_ID, DEEPEVAL_LLM_CLIENT_SECRET"
            )

        if self.auth_method == "post":
            return self._refresh_token_form_post()
        return self._refresh_token_basic_auth()

    def _refresh_token_basic_auth(self) -> str:
        """Token refresh using Basic Auth (Cisco-style)."""
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        data = f"grant_type={self.grant_type}"
        if self.scope:
            data += f"&scope={self.scope}"

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
        return self._parse_token_response(response.json())

    def _refresh_token_form_post(self) -> str:
        """Token refresh using form POST (Azure AD / GAIT style)."""
        data = {
            "grant_type": self.grant_type,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope

        response = requests.post(
            self.token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
            timeout=30,
        )
        response.raise_for_status()
        return self._parse_token_response(response.json())

    def _parse_token_response(self, token_data: dict) -> str:
        """Parse token response and update internal state."""
        self._token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.time() + expires_in
        return self._token

    def invalidate(self) -> None:
        """Force token refresh on next get_token() call."""
        self._token = None
        self._token_expiry = 0


def create_eval_model() -> Any | None:
    """
    Create LiteLLM-based eval model from environment variables.

    If DEEPEVAL_LLM_BASE_URL is set, creates a LiteLLMModel configured
    for the custom endpoint. Supports both static API keys and OAuth2
    token-based authentication.

    Environment Variables:
        DEEPEVAL_LLM_BASE_URL: Custom LLM endpoint (required for custom model)
        DEEPEVAL_LLM_MODEL: Model name (default: gpt-4o-mini)
        DEEPEVAL_LLM_PROVIDER: Provider identifier for model prefix (default: openai)
        DEEPEVAL_LLM_API_KEY: Static API key (if not using OAuth2)
        DEEPEVAL_LLM_CLIENT_APP_NAME: App key/name (passed in headers for some providers)
        DEEPEVAL_LLM_AUTH_HEADER: Auth header name (default: api-key)
        DEEPEVAL_LLM_EXTRA_HEADERS: JSON string of additional HTTP headers (optional).
            Note: LiteLLM does not support extra_headers via env vars natively,
            so we provide this for API gateways requiring custom headers.

        OAuth2 Configuration (optional):
        DEEPEVAL_LLM_TOKEN_URL: OAuth2 token endpoint
        DEEPEVAL_LLM_CLIENT_ID: OAuth2 client ID
        DEEPEVAL_LLM_CLIENT_SECRET: OAuth2 client secret
        DEEPEVAL_LLM_GRANT_TYPE: OAuth2 grant type (default: client_credentials)
        DEEPEVAL_LLM_SCOPE: OAuth2 scope
        DEEPEVAL_LLM_AUTH_METHOD: Token auth method - "basic" or "post"

    Returns:
        LiteLLMModel instance if configured, None otherwise (uses default OpenAI)

    Example - OAuth2 with Basic Auth (Okta-style):
        DEEPEVAL_LLM_BASE_URL=https://llm-gateway.example.com/openai/deployments/gpt-4o-mini
        DEEPEVAL_LLM_MODEL=gpt-4o-mini
        DEEPEVAL_LLM_CLIENT_ID=<client-id>
        DEEPEVAL_LLM_CLIENT_SECRET=<client-secret>
        DEEPEVAL_LLM_TOKEN_URL=https://identity.example.com/oauth2/default/v1/token
        DEEPEVAL_LLM_CLIENT_APP_NAME=<app-key>

    Example - Azure Active Directory:
        DEEPEVAL_LLM_BASE_URL=https://your-api.example.com/v1
        DEEPEVAL_LLM_MODEL=gpt-4o-mini
        DEEPEVAL_LLM_PROVIDER=openai
        DEEPEVAL_LLM_CLIENT_ID=<azure-client-id>
        DEEPEVAL_LLM_CLIENT_SECRET=<azure-client-secret>
        DEEPEVAL_LLM_TOKEN_URL=https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token
        DEEPEVAL_LLM_SCOPE=api://<resource>/.default
        DEEPEVAL_LLM_AUTH_METHOD=post

    Example - Static API Key (no OAuth2):
        DEEPEVAL_LLM_BASE_URL=https://api.your-provider.com/v1
        DEEPEVAL_LLM_MODEL=gpt-4o-mini
        DEEPEVAL_LLM_PROVIDER=openai
        DEEPEVAL_LLM_API_KEY=<your-api-key>
        # Note: Do NOT set DEEPEVAL_LLM_TOKEN_URL when using static API key

    Example - Azure OpenAI with Custom Headers (API Gateway):
        DEEPEVAL_LLM_BASE_URL=https://your-gateway.example.com/openai/deployments
        DEEPEVAL_LLM_MODEL=gpt-4o
        DEEPEVAL_LLM_PROVIDER=azure
        DEEPEVAL_LLM_API_KEY=<your-api-key>
        DEEPEVAL_LLM_EXTRA_HEADERS={"system-code": "APP-123", "x-tenant-id": "tenant-abc"}
    """
    base_url = os.environ.get("DEEPEVAL_LLM_BASE_URL")
    if not base_url:
        return None  # Use default OpenAI

    try:
        from deepeval.models import LiteLLMModel
    except ImportError:
        # LiteLLM not installed, fall back to default
        return None

    model = os.environ.get("DEEPEVAL_LLM_MODEL", "gpt-4o-mini")
    provider = os.environ.get("DEEPEVAL_LLM_PROVIDER", "openai")
    app_name = os.environ.get("DEEPEVAL_LLM_CLIENT_APP_NAME")
    auth_header = os.environ.get("DEEPEVAL_LLM_AUTH_HEADER", "api-key")

    # Determine API key - OAuth2 or static
    token_url = os.environ.get("DEEPEVAL_LLM_TOKEN_URL")
    if token_url:
        # OAuth2 authentication
        token_manager = OAuth2TokenManager()
        api_key = token_manager.get_token()
    else:
        # Static API key
        api_key = os.environ.get("DEEPEVAL_LLM_API_KEY", "")

    # Build generation kwargs with extra headers
    generation_kwargs: dict[str, Any] = {}
    extra_headers: dict[str, str] = {}

    # Parse custom headers from JSON environment variable.
    # Note: LiteLLM does not natively support extra_headers via environment variables
    # (see https://docs.litellm.ai/docs/completion/input#optional-fields).
    # We provide DEEPEVAL_LLM_EXTRA_HEADERS to enable custom headers for API gateways
    # that require additional headers (e.g., system-code for Azure OpenAI proxies).
    # Example: DEEPEVAL_LLM_EXTRA_HEADERS='{"system-code": "APP-123"}'
    extra_headers_json = os.environ.get("DEEPEVAL_LLM_EXTRA_HEADERS")
    if extra_headers_json:
        try:
            custom_headers = json.loads(extra_headers_json)
            if isinstance(custom_headers, dict):
                extra_headers.update(custom_headers)
            else:
                _logger.warning(
                    "DEEPEVAL_LLM_EXTRA_HEADERS must be a JSON object (dict), "
                    "got %s. Custom headers will be ignored.",
                    type(custom_headers).__name__,
                )
        except json.JSONDecodeError as e:
            # Log warning without exposing potentially sensitive header values
            _logger.warning(
                "Failed to parse DEEPEVAL_LLM_EXTRA_HEADERS as JSON: %s. "
                "Custom headers will be ignored. Expected format: "
                '\'{"header-name": "header-value"}\'',
                e.msg,
            )

    # Add auth header
    if api_key:
        extra_headers[auth_header] = api_key

    # Add app name if provided (Cisco-style)
    if app_name:
        generation_kwargs["user"] = json.dumps({"appkey": app_name})

    if extra_headers:
        generation_kwargs["extra_headers"] = extra_headers

    return LiteLLMModel(
        model=f"{provider}/{model}",
        base_url=base_url,
        api_key=api_key or "placeholder",
        temperature=0,
        generation_kwargs=generation_kwargs if generation_kwargs else None,
    )


__all__ = ["OAuth2TokenManager", "create_eval_model"]
