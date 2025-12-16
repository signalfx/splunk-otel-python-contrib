"""DeepEval evaluation model for Cisco CircuIT."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import requests
from deepeval.models.base_model import DeepEvalBaseLLM
from requests.auth import HTTPBasicAuth

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CircuitConfig:
    base_url: str
    deployment: str
    app_key: str
    client_id: str | None
    client_secret: str | None
    token_url: str
    token_cache: Path | None
    static_token: str | None
    session_id: str | None
    user_id: str | None
    timeout: float
    temperature: float | None
    max_tokens: int | None
    system_prompt: str | None


class _CiscoCircuitTokenManager:
    """Cache-aware OAuth client credentials manager for CircuIT."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        cache_path: Path | None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._cache_path = cache_path
        self._lock = Lock()

    def _read_cache(self) -> str | None:
        if not self._cache_path or not self._cache_path.exists():
            return None
        try:
            payload = json.loads(self._cache_path.read_text("utf-8"))
            expires_at = datetime.fromisoformat(payload["expires_at"]).replace(
                tzinfo=timezone.utc
            )
            if datetime.now(timezone.utc) < expires_at - timedelta(minutes=5):
                return payload["access_token"]
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    def _write_cache(self, token: str, expires_in: int) -> None:
        if not self._cache_path:
            return
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(expires_in, 0)
        )
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(
                    {
                        "access_token": token,
                        "expires_at": expires_at.isoformat(),
                    }
                ),
                encoding="utf-8",
            )
            os.chmod(self._cache_path, 0o600)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug(
                "Unable to persist CircuIT token cache", exc_info=True
            )

    def _fetch_new_token(self) -> tuple[str, int]:
        auth = HTTPBasicAuth(self._client_id, self._client_secret)
        response = requests.post(
            self._token_url,
            headers={"Accept": "application/json"},
            data={"grant_type": "client_credentials"},
            auth=auth,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(
                "CircuIT token endpoint did not return an access_token"
            )
        expires_in = int(data.get("expires_in", 3600))
        self._write_cache(token, expires_in)
        return token, expires_in

    def get_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if not force_refresh:
                cached = self._read_cache()
                if cached:
                    return cached
            token, _ = self._fetch_new_token()
            return token


def _load_config() -> _CircuitConfig:
    base_url = os.getenv(
        "CIRCUIT_UPSTREAM_BASE", "https://chat-ai.cisco.com"
    ).rstrip("/")
    deployment = (
        os.getenv("DEEPEVAL_CIRCUIT_DEPLOYMENT")
        or os.getenv("CIRCUIT_DEFAULT_DEPLOYMENT")
        or "gpt-4o-mini"
    )
    app_key = os.getenv("CISCO_APP_KEY") or os.getenv("CIRCUIT_APP_KEY")
    if not app_key:
        raise RuntimeError(
            "CISCO_APP_KEY environment variable is required for CircuIT evaluation"
        )

    client_id = os.getenv("CISCO_CLIENT_ID")
    client_secret = os.getenv("CISCO_CLIENT_SECRET")
    static_token = os.getenv("CISCO_CIRCUIT_TOKEN") or os.getenv(
        "CIRCUIT_ACCESS_TOKEN"
    )
    token_url = os.getenv(
        "CISCO_TOKEN_URL", "https://id.cisco.com/oauth2/default/v1/token"
    )
    token_cache_env = os.getenv("CIRCUIT_TOKEN_CACHE")
    if token_cache_env:
        token_cache = Path(token_cache_env).expanduser()
    else:
        token_cache = Path(tempfile.gettempdir()) / "circuit_eval_token.json"

    if not static_token and (not client_id or not client_secret):
        raise RuntimeError(
            "CircuIT evaluation requires either CISCO_CIRCUIT_TOKEN or both CISCO_CLIENT_ID and CISCO_CLIENT_SECRET"
        )

    session_id = os.getenv("CIRCUIT_SESSION_ID")
    user_id = os.getenv("CIRCUIT_USER_ID")
    timeout = float(os.getenv("CIRCUIT_TIMEOUT", "60"))
    temperature_env = os.getenv("CIRCUIT_TEMPERATURE")
    temperature = float(temperature_env) if temperature_env else None
    max_tokens_env = os.getenv("CIRCUIT_MAX_TOKENS")
    max_tokens = int(max_tokens_env) if max_tokens_env else None
    system_prompt = os.getenv("CIRCUIT_SYSTEM_PROMPT")

    return _CircuitConfig(
        base_url=base_url,
        deployment=deployment,
        app_key=app_key,
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url,
        token_cache=token_cache,
        static_token=static_token,
        session_id=session_id,
        user_id=user_id,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


class CiscoCircuitEvaluationLLM(DeepEvalBaseLLM):
    """Deepeval model that forwards prompts to the Cisco CircuIT API."""

    def __init__(
        self, config: _CircuitConfig, session: requests.Session | None = None
    ) -> None:
        self._config = config
        self._session = session or requests.Session()
        self._endpoint = f"{config.base_url}/openai/deployments/{config.deployment}/chat/completions"
        self._user_payload = {
            "appkey": config.app_key,
            "session_id": config.session_id or "",
            "user": config.user_id or "",
        }
        self._token_manager: _CiscoCircuitTokenManager | None = None
        self._static_token: str | None
        if config.static_token:
            self._static_token = config.static_token
        elif config.client_id and config.client_secret:
            self._token_manager = _CiscoCircuitTokenManager(
                config.client_id,
                config.client_secret,
                config.token_url,
                config.token_cache,
            )
            self._static_token = None
        else:
            self._static_token = None
        self._model_name = f"circuit://{config.deployment}"

    def load_model(self) -> "CiscoCircuitEvaluationLLM":
        return self

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: Any | None = None) -> Any:
        text = self._invoke(prompt)
        if schema is None:
            return text
        try:
            return self._apply_schema(schema, text)
        except Exception as exc:
            LOGGER.debug(
                "CircuIT schema parsing failed; falling back to raw text",
                exc_info=True,
            )
            raise TypeError("schema parsing failed") from exc

    async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def _invoke(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        payload = self._build_payload(prompt)
        response = self._post(payload)
        raw_body = ""
        try:
            raw_body = response.text  # type: ignore[attr-defined]
        except Exception:
            raw_body = ""
        if not raw_body:
            try:
                raw_body = json.dumps(response.json())
            except Exception:
                raw_body = ""
        if os.getenv("CIRCUIT_DEBUG_RAW") == "1":
            print(
                f"[circuit] status={getattr(response, 'status_code', '?')} body={raw_body}"
            )
        if raw_body:
            LOGGER.debug("CircuIT raw response: %s", raw_body[:2000])
        try:
            data = json.loads(raw_body) if raw_body else response.json()
        except (
            json.JSONDecodeError,
            TypeError,
            AttributeError,
        ) as exc:  # pragma: no cover - defensive
            raise RuntimeError("CircuIT response was not valid JSON") from exc
        content = self._extract_content(data)
        if content is None:
            raise RuntimeError(
                "CircuIT response did not include message content"
            )
        return content

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self._config.system_prompt:
            messages.append(
                {"role": "system", "content": self._config.system_prompt}
            )
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
        }
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
        payload["user"] = json.dumps(self._user_payload)
        return payload

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-key": token,
        }

    def _resolve_token(self, force_refresh: bool = False) -> str:
        if self._static_token:
            return self._static_token
        if not self._token_manager:
            raise RuntimeError("CircuIT OAuth credentials are not configured")
        return self._token_manager.get_token(force_refresh=force_refresh)

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        token = self._resolve_token()
        response = self._session.post(
            self._endpoint,
            headers=self._headers(token),
            json=payload,
            timeout=self._config.timeout,
        )
        if (
            response.status_code == 401
            and not self._static_token
            and self._token_manager
        ):
            LOGGER.info("CircuIT returned 401; refreshing token and retrying")
            fresh_token = self._resolve_token(force_refresh=True)
            response = self._session.post(
                self._endpoint,
                headers=self._headers(fresh_token),
                json=payload,
                timeout=self._config.timeout,
            )
        response.raise_for_status()
        return response

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str | None:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for entry in content:
                if isinstance(entry, dict):
                    if entry.get("type") == "text" and isinstance(
                        entry.get("text"), str
                    ):
                        return entry["text"].strip()
                    if entry.get("type") == "output_text" and isinstance(
                        entry.get("text"), str
                    ):
                        return entry["text"].strip()
                elif isinstance(entry, str) and entry.strip():
                    return entry.strip()
        return None

    @staticmethod
    def _apply_schema(schema: Any, payload: str) -> Any:
        if hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(payload)
        if hasattr(schema, "model_validate"):
            import json as _json

            data = _json.loads(payload)
            return schema.model_validate(data)
        if hasattr(schema, "parse_raw"):
            return schema.parse_raw(payload)
        if callable(schema):
            import json as _json

            data = _json.loads(payload)
            return schema(**data)
        raise TypeError("unsupported schema type")


def create_circuit_llm() -> CiscoCircuitEvaluationLLM:
    """Factory used by the entry point registry."""

    config = _load_config()
    return CiscoCircuitEvaluationLLM(config)


__all__ = [
    "CiscoCircuitEvaluationLLM",
    "create_circuit_llm",
]
