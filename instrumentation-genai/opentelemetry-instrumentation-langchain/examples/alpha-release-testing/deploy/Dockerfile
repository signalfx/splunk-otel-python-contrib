# Alpha Release Testing - Multi-App Container Image
# Supports: LangChain Evaluation, LangGraph Travel Planner, Traceloop, Direct Azure OpenAI
#
# Build from the repository root:
#   docker build -f instrumentation-genai/opentelemetry-instrumentation-langchain/examples/alpha-release-testing/deploy/Dockerfile \
#       -t alpha-test-apps:latest .
#
# Run examples:
#   # LangChain Evaluation
#   docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY alpha-test-apps:latest python tests/apps/langchain_evaluation_app.py
#
#   # LangGraph (Zero-Code)
#   docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY alpha-test-apps:latest \
#       opentelemetry-instrument python tests/apps/langgraph_travel_planner_app.py
#
#   # LangGraph (Manual)
#   docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY alpha-test-apps:latest \
#       python tests/apps/langgraph_travel_planner_app.py

FROM python:3.13-slim

ENV APP_HOME=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR ${APP_HOME}

# System tooling for curl/health checks and timezone awareness
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tzdata \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the directories needed for editable installs
COPY instrumentation-genai ${APP_HOME}/instrumentation-genai
COPY util ${APP_HOME}/util

WORKDIR ${APP_HOME}/instrumentation-genai/opentelemetry-instrumentation-langchain/examples/alpha-release-testing

# Drop any developer .env that might be present to avoid baking secrets into the image
RUN rm -f config/.env

# Install local packages in the same order as the documented steps
# Using .venv-langchain for consistency with local development
RUN python -m venv .venv-langchain \
    && . .venv-langchain/bin/activate \
    && pip install --upgrade pip \
    && pip install --no-deps -e ../../../../util/opentelemetry-util-genai \
    && pip install --no-deps -e ../../../../util/opentelemetry-util-genai-emitters-splunk \
    && pip install --no-deps -e ../../../../util/opentelemetry-util-genai-evals \
    && pip install -e ../../../../util/opentelemetry-util-genai-evals-deepeval \
    && pip install -e ../.. \
    && pip install langchain langchain-openai langchain-core langgraph python-dotenv openai

# Default environment can be overridden at runtime
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    OTEL_SERVICE_NAME=alpha-release-test \
    OTEL_RESOURCE_ATTRIBUTES=deployment.environment=alpha,test.phase=validation

# Activate venv for all commands
ENV PATH="${APP_HOME}/instrumentation-genai/opentelemetry-instrumentation-langchain/examples/alpha-release-testing/.venv-langchain/bin:$PATH"

# Health check (optional - can be customized per deployment)
HEALTHCHECK --interval=5m --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default entrypoint runs the test runner
# Can be overridden at runtime for specific apps
ENTRYPOINT ["./run_tests.sh"]
CMD ["all"]
