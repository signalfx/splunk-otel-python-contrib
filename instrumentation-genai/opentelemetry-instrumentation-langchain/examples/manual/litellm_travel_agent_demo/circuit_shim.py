#!/usr/bin/env python3
"""Small shim proxy for Cisco CircuIT OpenAI-compatible endpoint.

Purpose:
- Converts Authorization: Bearer <token> -> api-key: <token>
- Injects a top-level `user` JSON object containing the configured appkey
  (and preserves any provided session_id/user fields)
- Forwards the request to the real CircuIT API and returns the response

Usage:
1. Install deps: pip install flask requests
2. Set environment variables:
    - CISCO_CLIENT_ID / CISCO_CLIENT_SECRET (required for auto token refresh)
    - CISCO_APP_KEY (required)
    - CIRCUIT_UPSTREAM_BASE (optional, default: https://chat-ai.cisco.com)
    - CISCO_TOKEN_URL, CIRCUIT_TOKEN_CACHE (optional overrides)
3. Run: python circuit_shim.py
4. Point LiteLLM's `api_base` to the shim from inside Docker, for example:
   http://host.docker.internal:5001

Note: This is intentionally lightweight for local development only.
"""

import base64
import json
import logging
import os
from datetime import datetime, timedelta
from urllib.parse import urljoin

from flask import Flask, Response, jsonify, request
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("circuit_shim")

app = Flask(__name__)

# Upstream Circuit base (where we forward requests)
CIRCUIT_UPSTREAM_BASE = os.getenv("CIRCUIT_UPSTREAM_BASE", "https://chat-ai.cisco.com")

# Cisco OAuth credentials used to mint/refresh CircuIT tokens
CISCO_CLIENT_ID = os.getenv("CISCO_CLIENT_ID")
CISCO_CLIENT_SECRET = os.getenv("CISCO_CLIENT_SECRET")
CISCO_APP_KEY = os.getenv("CISCO_APP_KEY")
CISCO_TOKEN_URL = os.getenv("CISCO_TOKEN_URL", "https://id.cisco.com/oauth2/default/v1/token")
CIRCUIT_TOKEN_CACHE = os.getenv("CIRCUIT_TOKEN_CACHE", "/tmp/circuit_shim_token.json")


class TokenManager:
    """Cache-aware client credentials manager for Cisco CircuIT tokens."""

    def __init__(self, client_id: str, client_secret: str, cache_file: str, token_url: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache_file = cache_file
        self.token_url = token_url

    def _get_cached_token(self) -> str | None:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                cache_data = json.load(handle)
            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() < expires_at - timedelta(minutes=5):
                return cache_data["access_token"]
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None
        return None

    def _fetch_new_token(self) -> str:
        payload = "grant_type=client_credentials"
        basic_value = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("utf-8")
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic_value}",
        }
        response = requests.post(self.token_url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        token_data = response.json()
        expires_in = int(token_data.get("expires_in", 3600))
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

    def get_token(self, force_refresh: bool = False) -> str:
        if force_refresh:
            return self._fetch_new_token()
        token = self._get_cached_token()
        if token:
            return token
        return self._fetch_new_token()

    def invalidate_cache(self) -> None:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return
        try:
            os.remove(self.cache_file)
        except OSError:
            pass


TOKEN_MANAGER: TokenManager | None = None
if CISCO_CLIENT_ID and CISCO_CLIENT_SECRET:
    TOKEN_MANAGER = TokenManager(
        client_id=CISCO_CLIENT_ID,
        client_secret=CISCO_CLIENT_SECRET,
        cache_file=CIRCUIT_TOKEN_CACHE,
        token_url=CISCO_TOKEN_URL,
    )
else:
    log.warning(
        "CISCO_CLIENT_ID/SECRET not set; shim will rely on Authorization headers for CircuIT tokens"
    )


def extract_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    # fallback: return the header verbatim
    return auth_header


def build_upstream_headers(incoming_headers, token: str | None):
    headers = {}
    # Forward some incoming headers that make sense
    if "Content-Type" in incoming_headers:
        headers["Content-Type"] = incoming_headers["Content-Type"]
    else:
        headers["Content-Type"] = "application/json"

    # Circuit expects the OAuth access token in `api-key` header
    if token:
        headers["api-key"] = token

    # Copy any other headers you deem safe (optional)
    for k in ("Accept", "X-Request-ID", "User-Agent"):
        v = incoming_headers.get(k)
        if v:
            headers[k] = v
    return headers


def resolve_circuit_token(auth_header: str | None) -> str | None:
    if TOKEN_MANAGER:
        try:
            return TOKEN_MANAGER.get_token()
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Failed to mint CircuIT token via client credentials: %s", exc)
    return extract_token(auth_header)


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]) 
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]) 
def proxy(path):
    auth_header = request.headers.get("Authorization")
    token = resolve_circuit_token(auth_header)
    if not token:
        msg = {
            "error": "missing_token",
            "message": "Shim could not determine a CircuIT token. Set CISCO_CLIENT_ID/CISCO_CLIENT_SECRET (preferred) or include an Authorization header with a CircuIT bearer token.",
        }
        log.error("No CircuIT token available for request %s", request.path)
        return jsonify(msg), 500

    # Build headers for upstream
    headers = build_upstream_headers(request.headers, token)

    # Prepare body early so it's available for path remapping
    body = None
    if request.data:
        try:
            body = request.get_json(force=True)
        except Exception:
            # Not JSON - forward raw
            body = request.data

    # Determine upstream URL
    # If the incoming request is the OpenAI-compatible chat completion path,
    # remap it to Circuit's deployment-style path:
    #   incoming: /v1/chat/completions  ->  /openai/deployments/<model>/chat/completions
    upstream_url = None
    normalized_path = path.strip("/")
    # Remap several common OpenAI-style completion paths to Circuit's deployment path.
    # LiteLLM/OpenAI wrappers sometimes call '/v1/chat/completions' or '/chat/completions'
    # so accept both. Try to extract the deployment/model name from multiple possible keys.
    log.debug("normalized_path for remap check: '%s'", normalized_path)
    if "chat/completions" in normalized_path:
        # CircuIT does not currently support OpenAI's logprob params - drop them early.
        if isinstance(body, dict):
            removed = {}
            for unsupported_key in ("logprobs", "top_logprobs"):
                if unsupported_key in body:
                    removed[unsupported_key] = body.pop(unsupported_key)
            if removed:
                log.info("Removing unsupported params for CircuIT chat/completions: %s", list(removed))
        # Prefer common keys: 'model', 'deployment', or 'model_name'
        model_name = None
        if isinstance(body, dict):
            model_name = body.get("model") or body.get("deployment") or body.get("model_name")
        if not model_name:
            model_name = os.getenv("CIRCUIT_DEFAULT_DEPLOYMENT")
        if model_name:
            upstream_url = f"{CIRCUIT_UPSTREAM_BASE.rstrip('/')}/openai/deployments/{model_name}/chat/completions"
            log.info("Mapped incoming completions path to deployment '%s' -> %s", model_name, upstream_url)
        else:
            # fallback to default behavior
            upstream_url = urljoin(CIRCUIT_UPSTREAM_BASE.rstrip("/") + "/", path)
    else:
        upstream_url = urljoin(CIRCUIT_UPSTREAM_BASE.rstrip("/") + "/", path)

    log.info("Proxying %s %s -> %s", request.method, request.path, upstream_url)

    # If JSON and missing `user`, inject it per CircuIT API requirements.
    # Circuit examples pass `user` as a JSON string, so stringify it here.
    if isinstance(body, dict):
        # Extract existing user if present
        existing_user = body.get("user")
        user_obj = None
        if isinstance(existing_user, str):
            try:
                user_obj = json.loads(existing_user)
            except Exception:
                user_obj = None
        elif isinstance(existing_user, dict):
            user_obj = existing_user

        if user_obj is None:
            effective_appkey = CISCO_APP_KEY or ""
            if not effective_appkey:
                log.warning("No CISCO_APP_KEY set; injecting empty appkey into user object")
            user_obj = {"appkey": effective_appkey, "session_id": "", "user": ""}
        else:
            if "appkey" not in user_obj or not user_obj.get("appkey"):
                # Fill from CISCO_APP_KEY if available
                user_obj["appkey"] = user_obj.get("appkey") or CISCO_APP_KEY or ""

        # Circuit expects the `user` field as a JSON string
        try:
            body["user"] = json.dumps(user_obj)
        except Exception:
            body["user"] = str(user_obj)

        # Fail fast if appkey is empty — helps local debugging
        if not user_obj.get("appkey"):
            msg = {
                "error": "missing_appkey",
                "message": "No appkey provided. Set CISCO_APP_KEY env var or pass 'user' with '{\"appkey\":\"<your_appkey>\"}' in the JSON body.",
                "help": "Example body: -d '{\"model\":\"gpt-4o-mini\", \"messages\":[{\"role\":\"user\",\"content\":\"hi\"}], \"user\": \"{\\\"appkey\\\":\\\"<your_appkey>\\\"}\" }'"
            }
            log.error("Missing appkey in request and CISCO_APP_KEY not set; aborting and not forwarding to upstream")
            return jsonify(msg), 400

    # Helper to mask sensitive tokens for logs
    def _mask_value(v: str) -> str:
        try:
            if not v:
                return v
            if len(v) <= 10:
                return "<masked>"
            return v[:6] + "..." + v[-4:]
        except Exception:
            return "<masked>"

    # Forward request to upstream
    try:
        # Debugging: log headers/body that we'll send upstream (mask sensitive values)
        debug_headers = dict(headers)
        # Mask api-key (and Authorization if accidentally present) in debug headers
        if debug_headers.get("api-key"):
            debug_headers["api-key"] = _mask_value(debug_headers.get("api-key"))
        if debug_headers.get("Authorization"):
            debug_headers["Authorization"] = _mask_value(debug_headers.get("Authorization"))

        # Mask appkey if in body.user
        debug_body = None
        if isinstance(body, dict):
            debug_body = dict(body)
            try:
                # body['user'] may be a JSON string; parse for masking
                user_val = debug_body.get("user")
                if isinstance(user_val, str):
                    try:
                        user_parsed = json.loads(user_val)
                        # Mask appkey if present
                        if isinstance(user_parsed, dict) and user_parsed.get("appkey"):
                            user_parsed["appkey"] = _mask_value(user_parsed.get("appkey"))
                        debug_body["user"] = json.dumps(user_parsed)
                    except Exception:
                        # leave as-is
                        pass
                elif isinstance(user_val, dict):
                    # Mask inline dict user
                    if user_val.get("appkey"):
                        u = dict(user_val)
                        u["appkey"] = _mask_value(u.get("appkey"))
                        debug_body["user"] = u
            except Exception:
                pass
        else:
            debug_body = "<non-json body>"

        log.info("Forwarding to upstream URL=%s headers=%s body=%s", upstream_url, debug_headers, debug_body)

        json_body = body if isinstance(body, dict) else None
        data_body = None if isinstance(body, dict) else body
        resp = requests.request(
            request.method,
            upstream_url,
            headers=headers,
            json=json_body,
            data=data_body,
            timeout=60,
        )
        # If the cached token expired between refreshes, retry once with a forced refresh
        if resp.status_code == 401 and TOKEN_MANAGER:
            log.warning("CircuIT returned 401; forcing token refresh and retrying once")
            try:
                fresh_token = TOKEN_MANAGER.get_token(force_refresh=True)
            except Exception as exc:  # pragma: no cover - defensive
                log.error("Failed to refresh CircuIT token after 401: %s", exc)
            else:
                headers = build_upstream_headers(request.headers, fresh_token)
                resp = requests.request(
                    request.method,
                    upstream_url,
                    headers=headers,
                    json=json_body,
                    data=data_body,
                    timeout=60,
                )
    except requests.RequestException as exc:
        log.exception("Upstream request failed: %s", exc)
        return jsonify({"error": "upstream request failed", "detail": str(exc)}), 502

    # Return upstream response
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers] if resp.raw and getattr(resp.raw, 'headers', None) else [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded_headers]

    return Response(resp.content, status=resp.status_code, headers=dict(response_headers))


if __name__ == "__main__":
    port = int(os.getenv("CIRCUIT_SHIM_PORT", "5001"))
    host = os.getenv("CIRCUIT_SHIM_HOST", "0.0.0.0")
    log.info("Starting circuit_shim on %s:%s forwarding to %s", host, port, CIRCUIT_UPSTREAM_BASE)
    app.run(host=host, port=port)
