Tokenator - Rate Limit Predictor for GenAI
==========================================

*"I'll be back... before you hit your rate limit"*

Predictive rate limit monitoring for agentic AI applications.
Transforms existing OpenTelemetry token telemetry into actionable
warnings before rate limits are breached.

Installation
------------

::

    pip install splunk-util-genai-tokenator

Configuration
-------------

::

    export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"
    export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_WARNING_THRESHOLD=0.8
    export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_DB_PATH=~/.opentelemetry_genai_rate_limit.db
