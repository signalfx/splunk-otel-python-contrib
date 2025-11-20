import json
from pathlib import Path
from typing import Any

import pytest

from opentelemetry.util.evaluator.circuit_deepeval import (
    CiscoCircuitEvaluationLLM,
    create_circuit_llm,
)


class _DummyResponse:
    def __init__(
        self, *, status_code: int = 200, payload: dict[str, Any] | None = None
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self._text = json.dumps(self._payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")


def test_create_circuit_llm_requires_app_key(monkeypatch):
    monkeypatch.delenv("CISCO_APP_KEY", raising=False)
    monkeypatch.delenv("CIRCUIT_APP_KEY", raising=False)
    monkeypatch.setenv("DEEPEVAL_CIRCUIT_DEPLOYMENT", "demo")
    with pytest.raises(RuntimeError, match="CISCO_APP_KEY"):
        create_circuit_llm()


def test_static_token_invocation(monkeypatch):
    captured = {}

    def fake_post(self, url, headers=None, json=None, timeout=None):  # pylint: disable=unused-argument
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return _DummyResponse(
            payload={
                "choices": [
                    {
                        "message": {"content": "hello"},
                    }
                ]
            }
        )

    monkeypatch.setenv("CISCO_APP_KEY", "appkey")
    monkeypatch.setenv("CISCO_CIRCUIT_TOKEN", "token-123")
    monkeypatch.setenv("DEEPEVAL_CIRCUIT_DEPLOYMENT", "deployment")

    monkeypatch.setattr("requests.Session.post", fake_post, raising=False)

    model = create_circuit_llm()
    assert isinstance(model, CiscoCircuitEvaluationLLM)
    result = model.generate("Hi")
    assert result == "hello"
    assert "/openai/deployments/deployment/chat/completions" in captured["url"]
    user_payload = json.loads(captured["payload"]["user"])
    assert user_payload["appkey"] == "appkey"
    assert captured["headers"]["api-key"] == "token-123"


def test_token_refresh_flow(monkeypatch, tmp_path):
    token_cache = tmp_path / "token.json"
    tokens = ["initial-token", "refreshed-token"]

    def fake_token_post(url, headers=None, data=None, auth=None, timeout=None):  # pylint: disable=unused-argument
        token = tokens.pop(0)
        return _DummyResponse(
            payload={"access_token": token, "expires_in": 3600}
        )

    responses = [
        _DummyResponse(status_code=401),
        _DummyResponse(
            payload={
                "choices": [
                    {
                        "message": {"content": "refreshed"},
                    }
                ]
            }
        ),
    ]

    def fake_session_post(self, url, headers=None, json=None, timeout=None):  # pylint: disable=unused-argument
        response = responses.pop(0)
        if response.status_code >= 400:
            return response
        return response

    monkeypatch.setenv("CISCO_APP_KEY", "appkey")
    monkeypatch.setenv("CISCO_CLIENT_ID", "client")
    monkeypatch.setenv("CISCO_CLIENT_SECRET", "secret")
    monkeypatch.setenv("CIRCUIT_TOKEN_CACHE", str(token_cache))
    monkeypatch.setenv("DEEPEVAL_CIRCUIT_DEPLOYMENT", "deployment")

    monkeypatch.setattr("requests.post", fake_token_post)
    monkeypatch.setattr(
        "requests.Session.post", fake_session_post, raising=False
    )

    model = create_circuit_llm()
    assert model.generate("refresh please") == "refreshed"
    cache_content = json.loads(Path(token_cache).read_text("utf-8"))
    assert cache_content["access_token"] in {
        "initial-token",
        "refreshed-token",
    }


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("CISCO_CIRCUIT_TOKEN", raising=False)
    monkeypatch.delenv("CISCO_CLIENT_ID", raising=False)
    monkeypatch.delenv("CISCO_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("CISCO_APP_KEY", "appkey")
    monkeypatch.setenv("DEEPEVAL_CIRCUIT_DEPLOYMENT", "deployment")
    with pytest.raises(RuntimeError, match="CISCO_CIRCUIT_TOKEN"):
        create_circuit_llm()
