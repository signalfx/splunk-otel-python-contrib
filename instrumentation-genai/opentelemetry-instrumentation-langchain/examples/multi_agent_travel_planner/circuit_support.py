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

"""Reusable helpers for connecting LangChain apps to Cisco CircuIT."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Optional

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no"}
_FORCE_OAUTH_ENV_VARS: tuple[str, ...] = (
    "CIRCUIT_FORCE_OAUTH",
    "TRAVEL_FORCE_CIRCUIT_OAUTH",
)
_DEBUG_ENV_VARS: tuple[str, ...] = (
    "CIRCUIT_DEBUG_CONNECTIONS",
    "TRAVEL_DEBUG_CONNECTIONS",
)
_CONNECTION_DEBUG_CACHE: set[str] = set()
_CONNECTION_DEBUG_LOCK = Lock()


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _debug_connections_enabled() -> bool:
    # Respect the first explicit override; default to enabled for visibility.
    for env in _DEBUG_ENV_VARS:
        value = os.getenv(env)
        if value is not None:
            return value.strip().lower() not in _FALSEY
    return True


_DEBUG_CONNECTIONS_ENABLED = _debug_connections_enabled()


def _token_preview(token: Optional[str]) -> str:
    if not token:
        return "<empty>"
    length = len(token)
    if length <= 8:
        return f"{token[:2]}...{token[-2:]} (len={length})"
    return f"{token[:4]}...{token[-4:]} (len={length})"


def _debug_token_message(message: str, preview: Optional[str] = None) -> None:
    if not _DEBUG_CONNECTIONS_ENABLED:
        return
    if preview:
        print(f"[circuit-debug] {message} token={preview}")
    else:
        print(f"[circuit-debug] {message}")


def circuit_provider_enabled(provider_setting: Optional[str] = None) -> bool:
    if provider_setting:
        return provider_setting.strip().lower() in {"circuit", "splunk-circuit"}
    return bool(os.getenv("CISCO_APP_KEY") or os.getenv("CIRCUIT_APP_KEY"))


def resolve_model_name(
    provider_setting: Optional[str] = None,
    *,
    default_openai_model: str = "gpt-5-nano",
) -> str:
    if circuit_provider_enabled(provider_setting):
        return (
            os.getenv("CIRCUIT_DEPLOYMENT")
            or os.getenv("CIRCUIT_DEFAULT_DEPLOYMENT")
            or os.getenv("OPENAI_MODEL")
            or default_openai_model
        )
    return os.getenv("OPENAI_MODEL", default_openai_model)


def _circuit_token_cache_path(default_filename: str) -> Path | None:
    raw = os.getenv("CIRCUIT_TOKEN_CACHE")
    if raw is None:
        return Path(tempfile.gettempdir()) / default_filename
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _read_cached_token(cache_path: Path | None) -> Optional[str]:
    if not cache_path or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text("utf-8"))
        token = payload.get("access_token")
        expires_raw = payload.get("expires_at")
        if not token or not expires_raw:
            return None
        expires_at = datetime.fromisoformat(expires_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) < expires_at - timedelta(minutes=5):
            return token
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _write_cached_token(cache_path: Path | None, token: str, expires_in: int) -> None:
    if not cache_path:
        return
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(expires_in, 0))
    payload = {
        "access_token": token,
        "expires_at": expires_at.isoformat(),
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            os.chmod(cache_path, 0o600)
        except Exception:
            pass
    except Exception:  # pragma: no cover - defensive
        _debug_token_message("Unable to persist CircuIT token cache")


def _fetch_circuit_token(client_id: str, client_secret: str, token_url: str) -> tuple[str, int]:
    try:
        requests = importlib.import_module("requests")
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("requests is required to mint Cisco CircuIT access tokens") from exc

    response = requests.post(
        token_url,
        data={"grant_type": "client_credentials"},
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        auth=(client_id, client_secret),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("CircuIT token endpoint did not return access_token")
    expires_in = int(payload.get("expires_in", 3600))
    return token, expires_in


def _force_oauth() -> bool:
    return any(_truthy_env(name) for name in _FORCE_OAUTH_ENV_VARS)


def _mint_circuit_token(default_cache_filename: str) -> tuple[str, str, Optional[Path]]:
    cache_path = _circuit_token_cache_path(default_cache_filename)
    cached = _read_cached_token(cache_path)
    if cached:
        return cached, "oauth-cache", cache_path

    client_id = os.getenv("CISCO_CLIENT_ID") or os.getenv("CIRCUIT_CLIENT_ID")
    client_secret = os.getenv("CISCO_CLIENT_SECRET") or os.getenv("CIRCUIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set CISCO_CIRCUIT_TOKEN or provide Cisco OAuth client credentials")
    token_url = os.getenv("CISCO_TOKEN_URL", "https://id.cisco.com/oauth2/default/v1/token")
    token, expires_in = _fetch_circuit_token(client_id, client_secret, token_url)
    _write_cached_token(cache_path, token, expires_in)
    return token, "oauth-fetch", cache_path


def _augment_circuit_kwargs(
    model: str,
    base_kwargs: Dict[str, Any],
    *,
    default_cache_filename: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs: Dict[str, Any] = dict(base_kwargs)
    debug: Dict[str, Any] = {"provider": "circuit", "deployment": model}

    app_key = os.getenv("CISCO_APP_KEY") or os.getenv("CIRCUIT_APP_KEY")
    if not app_key:
        raise RuntimeError("CISCO_APP_KEY (or CIRCUIT_APP_KEY) must be set when using CircuIT")

    raw_base = kwargs.pop("base_url", None) or os.getenv("CIRCUIT_API_BASE")
    if raw_base:
        sanitized = raw_base.rstrip("/")
        if sanitized.endswith("/v1"):
            sanitized = sanitized[: -len("/v1")]
        if "/openai/deployments/" not in sanitized:
            sanitized = f"{sanitized}/openai/deployments/{model}"
    else:
        upstream = os.getenv("CIRCUIT_UPSTREAM_BASE", "https://chat-ai.cisco.com").rstrip("/")
        sanitized = f"{upstream}/openai/deployments/{model}"
    if sanitized.endswith("/chat/completions"):
        sanitized = sanitized[: -len("/chat/completions")]
    base_url = sanitized
    kwargs["base_url"] = base_url
    debug["base_url"] = base_url

    force_oauth = _force_oauth()
    debug["force_oauth"] = force_oauth

    api_key = kwargs.get("api_key")
    token_source_label: Optional[str] = None
    ignored_sources: list[str] = []

    if api_key:
        token_source_label = "kwargs"
    else:
        static_token: Optional[str] = None
        static_name: Optional[str] = None
        for candidate in ("CISCO_CIRCUIT_TOKEN", "CIRCUIT_ACCESS_TOKEN"):
            value = os.getenv(candidate)
            if value:
                static_token = value
                static_name = candidate
                break
        if static_token and not force_oauth:
            api_key = static_token
            token_source_label = f"static-env:{static_name}"
        else:
            if static_name and force_oauth:
                ignored_sources.append(static_name)

    cache_path_str = ""
    preview = ""

    if not api_key:
        api_key, issued_from, cache_path = _mint_circuit_token(default_cache_filename)
        token_source_label = issued_from
        cache_path_str = str(cache_path) if cache_path else ""
        debug["token_cache_path"] = cache_path_str
        preview = _token_preview(api_key)
        _debug_token_message(
            f"minted CircuIT token ({issued_from}) cache={cache_path_str or '<none>'}",
            preview,
        )
    else:
        preview = _token_preview(api_key)
        if token_source_label == "kwargs":
            _debug_token_message("using CircuIT token supplied via kwargs", preview)
        else:
            env_name = token_source_label.split(":", 1)[-1] if token_source_label else "env"
            _debug_token_message(f"using CircuIT token from {env_name}", preview)

    if ignored_sources:
        joined = ",".join(sorted(set(ignored_sources)))
        debug["ignored_token_sources"] = joined
        _debug_token_message(f"force OAuth enabled so ignoring static token from {joined}")

    debug["token_source"] = token_source_label or "unknown"
    debug["token_hint"] = preview
    if cache_path_str:
        debug["token_cache_path"] = cache_path_str

    kwargs["api_key"] = api_key

    default_headers = dict(kwargs.get("default_headers", {}))
    default_headers.setdefault("api-key", api_key)
    kwargs["default_headers"] = default_headers

    model_kwargs = dict(kwargs.get("model_kwargs", {}))
    user_payload = {"appkey": app_key}
    session_id = os.getenv("CIRCUIT_SESSION_ID")
    user_id = os.getenv("CIRCUIT_USER_ID")
    if session_id:
        user_payload["session_id"] = session_id
        debug["session_id_present"] = True
    else:
        debug["session_id_present"] = False
    if user_id:
        user_payload["user"] = user_id
        debug["user_id_present"] = True
    else:
        debug["user_id_present"] = False
    model_kwargs["user"] = json.dumps(user_payload)
    kwargs["model_kwargs"] = model_kwargs

    timeout_env = os.getenv("CIRCUIT_TIMEOUT")
    if timeout_env and "timeout" not in kwargs:
        try:
            kwargs["timeout"] = float(timeout_env)
            debug["timeout"] = kwargs["timeout"]
        except ValueError:
            pass

    retries_env = os.getenv("CIRCUIT_MAX_RETRIES")
    if retries_env and "max_retries" not in kwargs:
        try:
            kwargs["max_retries"] = int(retries_env)
            debug["max_retries"] = kwargs["max_retries"]
        except ValueError:
            pass

    return kwargs, debug


def resolve_openai_kwargs(
    model: str,
    *,
    provider_setting: Optional[str] = None,
    default_cache_filename: str = "circuit_llm_token.json",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs: Dict[str, Any] = {}
    debug: Dict[str, Any] = {"provider": "openai", "model": model}

    base_url_envs: Iterable[str] = (
        "TRAVEL_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )
    for env in base_url_envs:
        base_url = os.getenv(env)
        if base_url:
            base_url = base_url.rstrip("/")
            kwargs["base_url"] = base_url
            debug["base_url"] = base_url
            break

    api_key = os.getenv("TRAVEL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
        debug["api_key_present"] = True

    organization = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    if organization:
        kwargs["organization"] = organization
        debug["organization_present"] = True

    if circuit_provider_enabled(provider_setting):
        kwargs, circuit_debug = _augment_circuit_kwargs(
            model,
            kwargs,
            default_cache_filename=default_cache_filename,
        )
        debug.update(circuit_debug)
    else:
        # Remove OpenAI API key when CircuIT is not selected to avoid leaking hints.
        if circuit_provider_enabled(None):
            kwargs.pop("api_key", None)
            debug.pop("api_key_present", None)
            debug["ignored_openai_api_key"] = True
        debug["provider"] = "openai"

    return kwargs, debug


def log_connection_target(agent_name: str, model: str, debug: Dict[str, Any]) -> None:
    if not _DEBUG_CONNECTIONS_ENABLED:
        return
    provider = debug.get("provider", "openai")
    base_url = debug.get("base_url") or "<openai-default>"
    token_source = debug.get("token_source") or (
        "api-key" if debug.get("api_key_present") else "env-default"
    )
    key = f"{provider}|{base_url}|{model}"
    with _CONNECTION_DEBUG_LOCK:
        if key in _CONNECTION_DEBUG_CACHE:
            return
        _CONNECTION_DEBUG_CACHE.add(key)
    parts = [
        f"provider={provider}",
        f"model={model}",
        f"base_url={base_url}",
        f"token_source={token_source}",
        f"agent={agent_name}",
    ]
    if debug.get("session_id_present") is not None:
        parts.append(f"session_flag={'1' if debug.get('session_id_present') else '0'}")
    if debug.get("user_id_present") is not None:
        parts.append(f"user_flag={'1' if debug.get('user_id_present') else '0'}")
    token_hint = debug.get("token_hint")
    if token_hint:
        parts.append(f"token_hint={token_hint}")
    cache_path = debug.get("token_cache_path")
    if cache_path:
        parts.append(f"cache={cache_path}")
    if debug.get("force_oauth"):
        parts.append("force_oauth=1")
    ignored = debug.get("ignored_token_sources")
    if ignored:
        parts.append(f"ignored={ignored}")
    print("[circuit-debug] " + " ".join(parts))


def create_chat_openai(
    agent_name: str,
    *,
    session_id: Optional[str] = None,
    temperature: float = 0.0,
    provider_setting: Optional[str] = None,
    default_openai_model: str = "gpt-5-nano",
    default_cache_filename: str = "circuit_llm_token.json",
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
) -> Any:
    try:
        module = importlib.import_module("langchain_openai")
        ChatOpenAI = getattr(module, "ChatOpenAI")
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "langchain_openai.ChatOpenAI is required for CircuIT integration"
        ) from exc

    model = model_override or resolve_model_name(
        provider_setting, default_openai_model=default_openai_model
    )
    kwargs, debug = resolve_openai_kwargs(
        model,
        provider_setting=provider_setting,
        default_cache_filename=default_cache_filename,
    )
    log_connection_target(agent_name, model, debug)

    final_tags = list(tags) if tags is not None else []
    if f"agent:{agent_name}" not in final_tags:
        final_tags.append(f"agent:{agent_name}")

    llm_metadata: Dict[str, Any] = dict(metadata or {})
    llm_metadata.setdefault("agent_name", agent_name)
    if session_id is not None:
        llm_metadata.setdefault("session_id", session_id)
        llm_metadata.setdefault("thread_id", session_id)

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        tags=final_tags,
        metadata=llm_metadata,
        **kwargs,
    )


__all__ = [
    "circuit_provider_enabled",
    "resolve_model_name",
    "resolve_openai_kwargs",
    "log_connection_target",
    "create_chat_openai",
]
