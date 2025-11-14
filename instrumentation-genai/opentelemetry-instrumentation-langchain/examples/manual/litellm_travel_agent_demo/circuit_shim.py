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
    - CISCO_APP_KEY (required)
    - CIRCUIT_UPSTREAM_BASE (optional, default: https://chat-ai.cisco.com)
3. Run: python circuit_shim.py
4. Point LiteLLM's `api_base` to the shim from inside Docker, for example:
   http://host.docker.internal:5001

Note: This is intentionally lightweight for local development only.
"""

import os
import logging
from urllib.parse import urljoin

from flask import Flask, request, Response, jsonify
import requests
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("circuit_shim")

app = Flask(__name__)

# Upstream Circuit base (where we forward requests)
CIRCUIT_UPSTREAM_BASE = os.getenv("CIRCUIT_UPSTREAM_BASE", "https://chat-ai.cisco.com")
# Local appkey to include in the `user` JSON
# Local appkey to include in the `user` JSON (CISCO_APP_KEY per project examples)
CISCO_APP_KEY = os.getenv("CISCO_APP_KEY")


def extract_token(auth_header: str) -> str | None:
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


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]) 
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]) 
def proxy(path):
    auth_header = request.headers.get("Authorization")
    token = extract_token(auth_header)

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

        resp = requests.request(request.method, upstream_url, headers=headers, json=body if isinstance(body, dict) else None, data=None if isinstance(body, dict) else body, timeout=60)
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
