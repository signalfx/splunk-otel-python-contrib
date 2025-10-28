import base64
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import requests
from langchain_openai import ChatOpenAI

DEFAULT_CACHE_FILE = "/tmp/.token.json"


class TokenManager:
    """Cache-aware client credentials manager for Cisco CircuIT tokens."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        app_key: Optional[str],
        cache_file: str = DEFAULT_CACHE_FILE,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.app_key = app_key
        self.cache_file = cache_file
        self.token_url = "https://id.cisco.com/oauth2/default/v1/token"

    def _get_cached_token(self) -> Optional[str]:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                cache_data = json.load(handle)

            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() < expires_at - timedelta(minutes=5):
                return cache_data["access_token"]
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        return None

    def _fetch_new_token(self) -> str:
        payload = "grant_type=client_credentials"
        value = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("utf-8")
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}",
        }
        response = requests.post(
            self.token_url,
            headers=headers,
            data=payload,
            timeout=30,
        )
        response.raise_for_status()

        token_data = response.json()
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        cache_data = {
            "access_token": token_data["access_token"],
            "expires_at": expires_at.isoformat(),
        }

        if self.cache_file:
            with open(self.cache_file, "w", encoding="utf-8") as handle:
                json.dump(cache_data, handle, indent=2)
            os.chmod(self.cache_file, 0o600)
        return token_data["access_token"]

    def get_token(self) -> str:
        token = self._get_cached_token()
        if token:
            return token
        return self._fetch_new_token()

    def cleanup_token_cache(self) -> None:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return
        with open(self.cache_file, "r+b") as handle:
            length = handle.seek(0, 2)
            handle.seek(0)
            handle.write(b"\0" * length)
        os.remove(self.cache_file)


def _assert_credentials(client_id: Optional[str], client_secret: Optional[str]) -> Tuple[str, str]:
    if not client_id or not client_secret:
        raise RuntimeError(
            "CISCO_CLIENT_ID and CISCO_CLIENT_SECRET must be set in the environment."
        )
    return client_id, client_secret


def build_token_manager(cache_file: str = DEFAULT_CACHE_FILE) -> TokenManager:
    """Return a TokenManager using Cisco CircuIT credentials from env."""
    client_id, client_secret = _assert_credentials(
        os.getenv("CISCO_CLIENT_ID"),
        os.getenv("CISCO_CLIENT_SECRET"),
    )
    app_key = os.getenv("CISCO_APP_KEY")
    return TokenManager(client_id, client_secret, app_key, cache_file)


def create_cisco_chat_llm(
    *,
    cache_file: str = DEFAULT_CACHE_FILE,
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    top_p: float = 0.9,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.5,
    stop_sequences: Optional[list[str]] = None,
    seed: int = 100,
    extra_model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ChatOpenAI, TokenManager, str]:
    """Create a ChatOpenAI client pre-configured for Cisco CircuIT endpoints.

    Returns the LLM instance, the underlying TokenManager, and the API key used.
    """
    token_manager = build_token_manager(cache_file=cache_file)
    api_key = token_manager.get_token()

    app_key = token_manager.app_key
    user_md = {"appkey": app_key} if app_key else {}

    model_kwargs: Dict[str, Any] = {}
    if user_md:
        model_kwargs["user"] = json.dumps(user_md)
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop_sequences=stop_sequences or ["\n", "Human:", "AI:"],
        seed=seed,
        api_key=api_key,
        base_url=f"https://chat-ai.cisco.com/openai/deployments/{model}",
        default_headers={"api-key": api_key},
        model_kwargs=model_kwargs,
    )
    return llm, token_manager, api_key


__all__ = [
    "TokenManager",
    "build_token_manager",
    "create_cisco_chat_llm",
    "DEFAULT_CACHE_FILE",
]
