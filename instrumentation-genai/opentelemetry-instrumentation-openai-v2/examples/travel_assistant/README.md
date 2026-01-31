# Travel Assistant - OpenAI SDK Example

A travel assistant demonstrating OpenAI SDK **chat completions with tool calling** and **embeddings** with OpenTelemetry instrumentation.

## Features

- **Embeddings Search**: Semantic destination search using cosine similarity
- **Tool Calling Loop**: Uses OpenAI function calling to search flights, hotels, activities, and weather
- **Multi-turn Conversation**: Automatic tool execution and result integration
- **Full Observability**: OpenTelemetry traces for all LLM calls (chat + embeddings)
- **Dual Auth Support**: Works with Circuit (internal) or OpenAI API directly

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt

# Install the instrumentation package (from repo root)
pip install -e ../../
```

## Usage

### Quick Start with .env File

```bash
cp .env.example .env
# Edit .env with your credentials
python main.py
```

### Option 1: Circuit (Internal LLM Gateway)

```bash
export LLM_CLIENT_ID="your-client-id"
export LLM_CLIENT_SECRET="your-client-secret"
export LLM_APP_KEY="your-app-key"  # optional
export LLM_MODEL="gpt-4o-mini"     # optional, default: gpt-4o-mini

python main.py
```

### Option 2: OpenAI API Directly

```bash
export USE_OPENAI_DIRECT=true
export OPENAI_API_KEY="your-openai-api-key"
export LLM_MODEL="gpt-4o-mini"     # optional

python main.py
```

## Environment Variables

### Chat (Circuit or OpenAI)

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_CLIENT_ID` | OAuth2 client ID for Circuit | Required (Circuit) |
| `LLM_CLIENT_SECRET` | OAuth2 client secret | Required (Circuit) |
| `LLM_APP_KEY` | Application key for Circuit | Optional |
| `LLM_MODEL` | Chat model to use | `gpt-4o-mini` |
| `USE_OPENAI_DIRECT` | Use OpenAI API directly | `false` |
| `OPENAI_API_KEY` | OpenAI API key | Required if direct |

### Embeddings (Azure OpenAI - required)

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_OPENAI_API_VERSION` | API version (e.g., `2024-02-01`) | Required |
| `AZURE_EMBEDDING_DEPLOYMENT` | Azure deployment name for embeddings | `text-embedding-ada-002` |

**Note:** Circuit chat endpoint doesn't support embeddings, so Azure is required for the embedding demo. If Azure is not configured, Part 1 (semantic search) will be skipped.

### Observability

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |

## Available Tools

The assistant has access to these mock tools:

- **search_flights**: Find flights between cities
- **search_hotels**: Find hotels in a city
- **search_activities**: Discover activities and attractions
- **get_weather**: Get weather forecast

## Example Output

```
======================================================================
✈️  Travel Assistant (OpenAI SDK + Tool Calling + Embeddings)
======================================================================
📡 Exporting to Console + OTLP (http://localhost:4317)
✅ OpenAI instrumentation enabled

======================================================================
📍 PART 1: Semantic Destination Search (Embeddings)
======================================================================

🔍 Semantic Search: 'I want a relaxing beach vacation with temples'
--------------------------------------------------
   📊 Generating query embedding...
   ✅ Query embedding: 1536 dimensions
   📊 Generating embeddings for 6 destinations...
   ✅ Destination embeddings generated

   🎯 Top 3 matches:
      1. Bali, Indonesia (similarity: 0.847)
         Tropical paradise with stunning beaches, rice terraces...
      2. Tokyo, Japan (similarity: 0.721)
         Vibrant metropolis blending ancient temples...
      3. Marrakech, Morocco (similarity: 0.698)
         Exotic bazaars, vibrant souks...

======================================================================
🗺️  PART 2: Travel Planning (Chat + Tool Calling)
======================================================================

💬 User: I want to plan a trip to Tokyo from San Francisco...
--------------------------------------------------

🔄 Iteration 1
   🛠️  4 tool call(s)
   🔧 Calling search_flights({'origin': 'San Francisco', 'destination': 'Tokyo'...})
   📋 Result: Flights from San Francisco to Tokyo...
   ...

🔄 Iteration 2

✅ Assistant: Here's your Tokyo trip plan...

======================================================================
📊 Flushing spans to Console + OTLP...
✅ Traces exported!
======================================================================
```

## Observability

All OpenAI API calls are automatically instrumented, generating spans with:

- `gen_ai.system`: `openai`
- `gen_ai.operation.name`: `chat`
- `gen_ai.request.model`: Model name
- `gen_ai.response.model`: Actual model used
- `gen_ai.usage.input_tokens`: Input token count
- `gen_ai.usage.output_tokens`: Output token count

Tool calls are captured with function names and arguments.
