# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""Tests for deepeval_model.py OAuth2 and custom header support."""

import importlib
import os
from unittest import mock


class TestExtraHeadersParsing:
    """Tests for DEEPEVAL_LLM_EXTRA_HEADERS environment variable parsing."""

    def test_extra_headers_parsed_from_json(self):
        """Test that extra headers are correctly parsed from JSON env var."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://example.com/v1",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_API_KEY": "test-key",
            "DEEPEVAL_LLM_EXTRA_HEADERS": '{"system-code": "APP-123", "x-tenant-id": "tenant-abc"}',
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)
                dm.create_eval_model()

                # Check that LiteLLMModel was called with extra_headers
                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]
                assert "generation_kwargs" in call_kwargs
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]

                # Verify custom headers are present
                assert extra_headers["system-code"] == "APP-123"
                assert extra_headers["x-tenant-id"] == "tenant-abc"
                # Auth header should also be present
                assert extra_headers["api-key"] == "test-key"

    def test_extra_headers_invalid_json_logs_warning(self):
        """Test that invalid JSON in extra headers logs a warning and is ignored."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://example.com/v1",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_API_KEY": "test-key",
            "DEEPEVAL_LLM_EXTRA_HEADERS": "not-valid-json",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)

                # Capture warning log
                with mock.patch.object(dm._logger, "warning") as mock_warn:
                    dm.create_eval_model()

                    # Verify warning was logged (without sensitive data)
                    mock_warn.assert_called_once()
                    call_args = mock_warn.call_args[0]
                    assert (
                        "Failed to parse DEEPEVAL_LLM_EXTRA_HEADERS"
                        in call_args[0]
                    )
                    assert "not-valid-json" not in str(
                        call_args
                    )  # No sensitive data

                # Should still work, just without custom headers
                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]

                # Only auth header present, no custom headers
                assert extra_headers == {"api-key": "test-key"}

    def test_extra_headers_empty_string_ignored(self):
        """Test that empty string for extra headers is handled."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://example.com/v1",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_API_KEY": "test-key",
            "DEEPEVAL_LLM_EXTRA_HEADERS": "",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)
                dm.create_eval_model()

                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]
                assert extra_headers == {"api-key": "test-key"}

    def test_extra_headers_non_dict_json_logs_warning(self):
        """Test that non-dict JSON (e.g., array) logs a warning and is ignored."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://example.com/v1",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_API_KEY": "test-key",
            "DEEPEVAL_LLM_EXTRA_HEADERS": '["not", "a", "dict"]',
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)

                # Capture warning log
                with mock.patch.object(dm._logger, "warning") as mock_warn:
                    dm.create_eval_model()

                    # Verify warning was logged about wrong type
                    mock_warn.assert_called_once()
                    call_args = mock_warn.call_args[0]
                    assert "must be a JSON object" in call_args[0]
                    assert "list" in str(call_args)  # Type name is logged

                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]
                # Non-dict JSON should be ignored
                assert extra_headers == {"api-key": "test-key"}

    def test_extra_headers_without_base_url_returns_none(self):
        """Test that create_eval_model returns None without base URL."""
        env = {
            "DEEPEVAL_LLM_EXTRA_HEADERS": '{"system-code": "APP-123"}',
        }

        with mock.patch.dict(os.environ, env, clear=True):
            import opentelemetry.util.evaluator.deepeval_model as dm

            importlib.reload(dm)
            result = dm.create_eval_model()

            # Should return None when no base URL is set
            assert result is None

    def test_custom_auth_header_name(self):
        """Test that custom auth header name is used."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://example.com/v1",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_API_KEY": "test-key",
            "DEEPEVAL_LLM_AUTH_HEADER": "Authorization",
            "DEEPEVAL_LLM_EXTRA_HEADERS": '{"system-code": "APP-123"}',
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)
                dm.create_eval_model()

                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]

                # Custom auth header name should be used
                assert "Authorization" in extra_headers
                assert extra_headers["Authorization"] == "test-key"
                assert extra_headers["system-code"] == "APP-123"
                # Default api-key should not be present
                assert "api-key" not in extra_headers

    def test_static_api_key_without_oauth2(self):
        """Test static API key works without OAuth2 credentials (no token_url, client_id, client_secret)."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://example.com/v1",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_PROVIDER": "azure",
            "DEEPEVAL_LLM_API_KEY": "static-api-key-12345",
            # No DEEPEVAL_LLM_TOKEN_URL, CLIENT_ID, or CLIENT_SECRET
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)
                dm.create_eval_model()

                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]

                # Verify model and base_url are set correctly
                assert call_kwargs["model"] == "azure/gpt-4o"
                assert call_kwargs["base_url"] == "https://example.com/v1"
                assert call_kwargs["api_key"] == "static-api-key-12345"

                # Verify auth header is set with static API key
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]
                assert extra_headers["api-key"] == "static-api-key-12345"

    def test_static_api_key_with_extra_headers(self):
        """Test static API key combined with extra headers (Softbank use case)."""
        env = {
            "DEEPEVAL_LLM_BASE_URL": "https://azure-gateway.example.com/openai/deployments",
            "DEEPEVAL_LLM_MODEL": "gpt-4o",
            "DEEPEVAL_LLM_PROVIDER": "azure",
            "DEEPEVAL_LLM_API_KEY": "azure-api-key",
            "DEEPEVAL_LLM_EXTRA_HEADERS": '{"system-code": "SOFTBANK-APP-123"}',
            # No OAuth2 credentials - using static API key
        }

        with mock.patch.dict(os.environ, env, clear=True):
            mock_litellm_model = mock.MagicMock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "deepeval.models": mock.MagicMock(
                        LiteLLMModel=mock_litellm_model
                    )
                },
            ):
                import opentelemetry.util.evaluator.deepeval_model as dm

                importlib.reload(dm)
                dm.create_eval_model()

                mock_litellm_model.assert_called_once()
                call_kwargs = mock_litellm_model.call_args[1]

                # Verify extra headers contain both system-code and api-key
                extra_headers = call_kwargs["generation_kwargs"][
                    "extra_headers"
                ]
                assert extra_headers["system-code"] == "SOFTBANK-APP-123"
                assert extra_headers["api-key"] == "azure-api-key"

    def test_default_openai_with_only_openai_api_key(self):
        """Test that only OPENAI_API_KEY (no DEEPEVAL_LLM_* vars) returns None for default OpenAI."""
        env = {
            "OPENAI_API_KEY": "sk-test-openai-key-12345",
            # No DEEPEVAL_LLM_BASE_URL or other custom vars
        }

        with mock.patch.dict(os.environ, env, clear=True):
            import opentelemetry.util.evaluator.deepeval_model as dm

            importlib.reload(dm)
            result = dm.create_eval_model()

            # Should return None - DeepEval will use default OpenAI with OPENAI_API_KEY
            assert result is None

    def test_default_openai_not_affected_by_extra_headers_env(self):
        """Test that DEEPEVAL_LLM_EXTRA_HEADERS without base URL doesn't break default OpenAI."""
        env = {
            "OPENAI_API_KEY": "sk-test-openai-key-12345",
            "DEEPEVAL_LLM_EXTRA_HEADERS": '{"system-code": "APP-123"}',
            # No DEEPEVAL_LLM_BASE_URL - should still use default OpenAI
        }

        with mock.patch.dict(os.environ, env, clear=True):
            import opentelemetry.util.evaluator.deepeval_model as dm

            importlib.reload(dm)
            result = dm.create_eval_model()

            # Should return None - extra headers are ignored without base URL
            # DeepEval will use default OpenAI with OPENAI_API_KEY
            assert result is None
