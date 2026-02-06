"""
LlamaIndex Zero-Code Server using CircuIT or OpenAI LLM.

This server exposes an HTTP endpoint for chat requests and uses
zero-code OpenTelemetry instrumentation via opentelemetry-instrument.

Run with: opentelemetry-instrument python server.py
"""

import json
import os
import base64
import time
from typing import Any
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.llms.openai import OpenAI


class OAuth2TokenManager:
    """Simple OAuth2 client-credentials token manager for custom LLM gateways."""

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str | None = None,
        token_refresh_buffer_seconds: int = 300,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.token_refresh_buffer = token_refresh_buffer_seconds
        self._token: str | None = None
        self._token_expiry = 0.0

    def get_token(self) -> str:
        if self._token and time.time() < (
            self._token_expiry - self.token_refresh_buffer
        ):
            return self._token
        return self._refresh_token()

    def _refresh_token(self) -> str:
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        data = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope
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
        token_data = response.json()
        self._token = str(token_data["access_token"])
        expires_in = int(token_data.get("expires_in", 3600))
        self._token_expiry = time.time() + expires_in
        return self._token


# =============================================================================
# LLM Configuration - OAuth2 Provider
# =============================================================================

# Optional app key for request tracking
LLM_APP_KEY = os.environ.get("LLM_APP_KEY")

# Check if we should use OAuth2 or standard OpenAI
USE_OAUTH2 = bool(os.environ.get("LLM_CLIENT_ID"))

# Initialize token manager if OAuth2 credentials are present
token_manager: OAuth2TokenManager | None = None
if USE_OAUTH2:
    token_manager = OAuth2TokenManager(
        token_url=os.environ.get("LLM_TOKEN_URL", ""),
        client_id=os.environ.get("LLM_CLIENT_ID", ""),
        client_secret=os.environ.get("LLM_CLIENT_SECRET", ""),
        scope=os.environ.get("LLM_SCOPE"),
    )


# Custom LLM for Cisco CircuIT
class CircuITLLM(CustomLLM):
    """Custom LLM implementation for Cisco CircuIT gateway."""

    api_url: str
    token_manager: OAuth2TokenManager
    app_key: str | None = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=128000,
            num_output=4096,
        )

    @llm_chat_callback()
    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Make a chat completion request to CircuIT."""
        # Get fresh token
        access_token = self.token_manager.get_token()

        # Convert LlamaIndex messages to OpenAI format
        api_messages = []
        for msg in messages:
            api_messages.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
            )

        payload: dict[str, Any] = {
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if self.app_key:
            payload["user"] = json.dumps({"appkey": self.app_key})

        # Make request to CircuIT
        response = requests.post(
            self.api_url,
            headers={
                "api-key": access_token,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=result,
        )

    @llm_chat_callback()
    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat - for now, just call sync version."""
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages: list[ChatMessage], **kwargs: Any):
        """Stream chat - not supported, just return single response."""
        response = self.chat(messages, **kwargs)
        yield response

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete is not used by ReActAgent, but required by interface."""
        raise NotImplementedError("Use chat() instead")

    def stream_complete(self, prompt: str, **kwargs: Any):
        """Streaming not implemented."""
        raise NotImplementedError("Streaming not supported")


# Initialize LLM
def initialize_llm():
    """Initialize OAuth2 gateway LLM or fall back to standard OpenAI API key."""
    openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    llm_base_url = os.getenv("LLM_BASE_URL")
    # Backward-compatible fallback names
    llm_base_url = llm_base_url or os.getenv("CIRCUIT_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if USE_OAUTH2 and token_manager:
        if not llm_base_url:
            raise RuntimeError(
                "LLM_BASE_URL is required when using OAuth2 gateway credentials."
            )

        return CircuITLLM(
            api_url=str(llm_base_url),
            token_manager=token_manager,
            app_key=LLM_APP_KEY,
            model_name=openai_model_name,
            temperature=0.0,
        )

    if openai_api_key:
        return OpenAI(model=openai_model_name, temperature=0.0)

    raise RuntimeError(
        "No LLM credentials configured. Set either OAuth2 gateway credentials "
        "(LLM_BASE_URL/LLM_TOKEN_URL/LLM_CLIENT_ID/LLM_CLIENT_SECRET) "
        "or OPENAI_API_KEY."
    )


# HTTP Request Handler
class ChatRequestHandler(BaseHTTPRequestHandler):
    llm = None

    def do_GET(self):
        """Handle GET requests for health check."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests for chat."""
        if self.path == "/chat":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                request_data = json.loads(post_data.decode("utf-8"))
                message = request_data.get("message", "")
                system_prompt = request_data.get(
                    "system_prompt", "You are a helpful assistant!"
                )

                if not message:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "message is required"}).encode()
                    )
                    return

                # Call LLM
                messages = [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=message),
                ]

                result = self.llm.chat(messages)

                # Send response
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"response": result.message.content}
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Log requests to stdout."""
        print(f"{self.address_string()} - {format % args}")


def run_server(port=8080):
    """Run the HTTP server."""
    # Initialize LLM
    ChatRequestHandler.llm = initialize_llm()
    provider = "CircuIT" if isinstance(ChatRequestHandler.llm, CircuITLLM) else "OpenAI"
    print(f"✓ LlamaIndex {provider} LLM initialized")

    server_address = ("", port)
    httpd = HTTPServer(server_address, ChatRequestHandler)
    print(f"Server running on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    run_server(port)
