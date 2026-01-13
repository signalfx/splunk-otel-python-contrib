from unittest import mock

import pytest

from opentelemetry.util.oauth2_token_manager import OAuth2TokenManager


def test_token_manager_requires_credentials(monkeypatch):
    monkeypatch.delenv("OAUTH2_CLIENT_ID", raising=False)
    monkeypatch.delenv("OAUTH2_CLIENT_SECRET", raising=False)

    with pytest.raises(
        ValueError, match=r"OAUTH2_CLIENT_ID/OAUTH2_CLIENT_SECRET"
    ):
        OAuth2TokenManager()


def test_oauth_env_vars_take_precedence(monkeypatch):
    monkeypatch.setenv("OAUTH2_CLIENT_ID", "oauth2-id")
    monkeypatch.setenv("OAUTH2_CLIENT_SECRET", "oauth2-secret")
    monkeypatch.setenv("OAUTH2_TOKEN_URL", "https://oauth2.example/token")

    with mock.patch(
        "opentelemetry.util.oauth2_token_manager.OAuth2ClientCredentialsTokenManager"
    ) as mocked_delegate:
        OAuth2TokenManager()

    mocked_delegate.assert_called_once()
    kwargs = mocked_delegate.call_args.kwargs
    assert kwargs["client_id"] == "oauth2-id"
    assert kwargs["client_secret"] == "oauth2-secret"
    assert kwargs["token_url"] == "https://oauth2.example/token"


def test_llm_base_url_oauth_env_takes_precedence(monkeypatch):
    monkeypatch.setenv(
        "OAUTH2_LLM_BASE_URL", "https://oauth2-llm.example/deployments"
    )

    assert (
        OAuth2TokenManager.get_llm_base_url("model-x")
        == "https://oauth2-llm.example/deployments/model-x"
    )
