FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2



COPY requirements.txt requirements.txt
# Install from requirements file
RUN uv pip install -r requirements.txt




RUN uv pip install aws-opentelemetry-distro==0.19.0
# Fail the build rather than ship an image whose GenAI instrumentation cannot import.
RUN python -c "\
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as g; \
assert hasattr(g, 'GEN_AI_CONVERSATION_ID'), 'semconv too old: aws-opentelemetry-distro==0.19.0 pin regressed'; \
from opentelemetry.instrumentation.openai import OpenAIInstrumentor; \
print('instrumentation import OK')"


# Signal that this is running in Docker for host binding logic
ENV DOCKER_CONTAINER=1

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

# Copy entire project (respecting .dockerignore)
COPY . .

# Use the full module path

CMD ["opentelemetry-instrument", "python", "-m", "main"]
