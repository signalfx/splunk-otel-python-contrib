"""
LlamaIndex Zero-Code Server using CircuIT LLM.

This server exposes an HTTP endpoint for chat requests and uses
zero-code OpenTelemetry instrumentation via opentelemetry-instrument.

Run with: opentelemetry-instrument python main_server.py
"""

import json
import os
from typing import Any, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_chat_callback


# OAuth2 Token Manager for Cisco CircuIT
class OAuth2TokenManager:
    """Manages OAuth2 token lifecycle for Cisco CircuIT gateway."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: Optional[str] = None,
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

    def get_token(self) -> str:
        """Get valid access token, refreshing if needed."""
        import time
        import base64

        # Check if token is still valid (with 5-minute buffer)
        if self._access_token and self._token_expiry:
            if time.time() < self._token_expiry - 300:
                return self._access_token

        # Request new token using Basic Auth (Cisco CircuIT method)
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope

        response = requests.post(self.token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]

        # Calculate expiry time (default to 3600 seconds if not provided)
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.time() + expires_in

        print(f"✓ OAuth2 token obtained, expires in {expires_in}s")
        return self._access_token


# Custom LLM for Cisco CircuIT
class CircuITLLM(CustomLLM):
    """Custom LLM implementation for Cisco CircuIT gateway."""

    api_url: str
    token_manager: OAuth2TokenManager
    app_key: str
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

        # CircuIT requires appkey as JSON string in user field
        user_field = json.dumps({"appkey": self.app_key})

        # Make request to CircuIT
        response = requests.post(
            self.api_url,
            headers={
                "api-key": access_token,
                "Content-Type": "application/json",
            },
            json={
                "messages": api_messages,
                "temperature": self.temperature,
                "user": user_field,
            },
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
    """Initialize CircuIT LLM from environment variables."""
    circuit_base_url = os.getenv("CIRCUIT_BASE_URL")
    circuit_token_url = os.getenv("CIRCUIT_TOKEN_URL")
    circuit_client_id = os.getenv("CIRCUIT_CLIENT_ID")
    circuit_client_secret = os.getenv("CIRCUIT_CLIENT_SECRET")
    circuit_app_key = os.getenv("CIRCUIT_APP_KEY", "llamaindex-zero-code-demo")
    circuit_scope = os.getenv("CIRCUIT_SCOPE")

    if not all(
        [circuit_base_url, circuit_token_url, circuit_client_id, circuit_client_secret]
    ):
        raise RuntimeError("Missing required CircuIT environment variables")

    token_manager = OAuth2TokenManager(
        token_url=circuit_token_url,
        client_id=circuit_client_id,
        client_secret=circuit_client_secret,
        scope=circuit_scope,
    )

    return CircuITLLM(
        api_url=circuit_base_url,
        token_manager=token_manager,
        app_key=circuit_app_key,
        model_name="gpt-4o-mini",
        temperature=0.0,
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
    print("✓ LlamaIndex CircuIT LLM initialized")

    server_address = ("", port)
    httpd = HTTPServer(server_address, ChatRequestHandler)
    print(f"Server running on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    run_server(port)
