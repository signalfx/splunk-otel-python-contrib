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

"""OAuth2 token helpers (stdlib-only).

These utilities are intentionally dependency-light so examples and instrumentations
can obtain OAuth2 access tokens without requiring third-party HTTP clients.

Environment Variables (no aliases):
  - `OAUTH2_CLIENT_ID`
  - `OAUTH2_CLIENT_SECRET`
  - `OAUTH2_TOKEN_URL`
  - `OAUTH2_LLM_BASE_URL`
  - `OAUTH2_APP_KEY` (provider-specific, used by examples)
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Optional


class OAuth2TokenError(RuntimeError):
    pass


@dataclass
class OAuth2TokenResponse:
    access_token: str
    expires_in: int | None = None
    token_type: str | None = None
    raw: Mapping[str, object] | None = None


class OAuth2ClientCredentialsTokenManager:
    """Caches and refreshes OAuth2 access tokens using client-credentials flow."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        token_url: str,
        grant_type: str = "client_credentials",
        scope: str | None = None,
        timeout_seconds: float = 30.0,
        refresh_buffer_seconds: int = 300,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError("OAuth2 client_id and client_secret are required")
        if not token_url:
            raise ValueError("OAuth2 token_url is required")
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._grant_type = grant_type or "client_credentials"
        self._scope = scope
        self._timeout_seconds = float(timeout_seconds)
        self._refresh_buffer_seconds = int(refresh_buffer_seconds)
        self._extra_headers = dict(extra_headers or {})

        self._token: str | None = None
        self._token_expiry: float = 0.0

    def get_token(self) -> str:
        if self._token and time.time() < (
            self._token_expiry - self._refresh_buffer_seconds
        ):
            return self._token
        return self._refresh_token()

    def invalidate(self) -> None:
        self._token = None
        self._token_expiry = 0.0

    def is_token_valid(self) -> bool:
        return bool(
            self._token
            and time.time()
            < (self._token_expiry - self._refresh_buffer_seconds)
        )

    @property
    def token_expires_at(self) -> float:
        return self._token_expiry

    def _refresh_token(self) -> str:
        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode("utf-8")
        ).decode("ascii")

        payload: dict[str, str] = {"grant_type": self._grant_type}
        if self._scope:
            payload["scope"] = self._scope
        body = urllib.parse.urlencode(payload).encode("utf-8")

        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}",
            **self._extra_headers,
        }

        request = urllib.request.Request(
            self._token_url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise OAuth2TokenError(
                f"OAuth2 token request failed: HTTP {exc.code} {exc.reason} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OAuth2TokenError(
                f"OAuth2 token request failed: {exc}"
            ) from exc

        try:
            token_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OAuth2TokenError(
                f"OAuth2 token response is not valid JSON: {raw[:200]}"
            ) from exc

        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise OAuth2TokenError(
                f"OAuth2 token response missing access_token: keys={list(token_data.keys())}"
            )
        expires_in_val = token_data.get("expires_in")
        expires_in: int | None = None
        if isinstance(expires_in_val, (int, float)) and expires_in_val > 0:
            expires_in = int(expires_in_val)

        self._token = access_token
        self._token_expiry = time.time() + (expires_in or 3600)
        return self._token


class OAuth2TokenManager:
    """Convenience wrapper around OAuth2 client-credentials token retrieval.

    Uses a default Cisco Chat AI token URL and base URL, but is fully configurable
    via environment variables.

    Environment Variables:
        OAUTH2_CLIENT_ID: OAuth2 client ID (required)
        OAUTH2_CLIENT_SECRET: OAuth2 client secret (required)
        OAUTH2_TOKEN_URL: Token endpoint (default: https://id.cisco.com/oauth2/default/v1/token)
        OAUTH2_LLM_BASE_URL: LLM endpoint base (default: https://chat-ai.cisco.com/openai/deployments)
    """

    DEFAULT_TOKEN_URL = "https://id.cisco.com/oauth2/default/v1/token"
    DEFAULT_LLM_BASE_URL = "https://chat-ai.cisco.com/openai/deployments"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: Optional[str] = None,
        token_refresh_buffer_seconds: int = 300,
    ) -> None:
        resolved_client_id = client_id or os.environ.get("OAUTH2_CLIENT_ID")
        resolved_client_secret = client_secret or os.environ.get(
            "OAUTH2_CLIENT_SECRET"
        )
        resolved_token_url = (
            token_url
            or os.environ.get("OAUTH2_TOKEN_URL")
            or self.DEFAULT_TOKEN_URL
        )
        if not resolved_client_id or not resolved_client_secret:
            raise ValueError(
                "OAuth2 credentials required. "
                "Set client_id/client_secret or OAUTH2_CLIENT_ID/OAUTH2_CLIENT_SECRET."
            )
        self._delegate = OAuth2ClientCredentialsTokenManager(
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            token_url=resolved_token_url,
            refresh_buffer_seconds=token_refresh_buffer_seconds,
        )

    def get_token(self) -> str:
        return self._delegate.get_token()

    def invalidate(self) -> None:
        self._delegate.invalidate()

    def is_token_valid(self) -> bool:
        return self._delegate.is_token_valid()

    @property
    def token_expires_at(self) -> float:
        return self._delegate.token_expires_at

    @classmethod
    def get_llm_base_url(cls, model: str = "gpt-4o-mini") -> str:
        base = (
            os.environ.get("OAUTH2_LLM_BASE_URL") or cls.DEFAULT_LLM_BASE_URL
        )
        return f"{base}/{model}"


__all__ = [
    "OAuth2ClientCredentialsTokenManager",
    "OAuth2TokenError",
    "OAuth2TokenManager",
    "OAuth2TokenResponse",
]
