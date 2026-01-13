splunk-otel-util-genai-test
===========================

This package contains lightweight utilities used by GenAI examples and test/demo
code. It intentionally does **not** live in ``splunk-otel-util-genai`` to keep the
core runtime utility package minimal.

Currently provided modules:

- ``opentelemetry.util.oauth2_token_manager`` (stdlib-only OAuth2 client-credentials helper)

