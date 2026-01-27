#!/usr/bin/env python3

from __future__ import annotations

import os

from dotenv import load_dotenv

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.util.genai.handler import get_telemetry_handler

# Load .env first
load_dotenv()

try:
    from openai import OpenAI
    from traceloop.sdk import Traceloop
    from traceloop.sdk.decorators import agent, task, tool, workflow

    # Initialize Traceloop - this will also trigger TraceloopSpanProcessor registration
    Traceloop.init(disable_batch=True, api_endpoint="http://localhost:4318")
except ImportError:
    raise RuntimeError("Install traceloop-sdk: pip install traceloop-sdk")
except (TypeError, ValueError) as config_error:
    # Configuration errors should fail-fast during startup
    raise RuntimeError(f"Traceloop configuration error: {config_error}")
except Exception as runtime_error:
    print(f"Warning: Traceloop initialization issue: {runtime_error}")
    raise

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# After Traceloop.init(), add:
logger_provider = LoggerProvider(
    resource=Resource({"service.name": "traceloop-example"})
)
log_processor = BatchLogRecordProcessor(
    OTLPLogExporter(endpoint="http://localhost:4318/v1/logs")
)
logger_provider.add_log_record_processor(log_processor)
set_logger_provider(logger_provider)


@agent(name="joke_translation")
def translate_joke_to_pirate(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Translate the below joke to pirate-like english:\n\n{joke}",
            }
        ],
    )

    history_jokes_tool()

    return completion.choices[0].message.content


@tool(name="history_jokes")
def history_jokes_tool():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "get some history jokes"}],
    )

    return completion.choices[0].message.content


@task(name="joke_creation")
def create_joke():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"}
        ],
    )

    return completion.choices[0].message.content


@task(name="signature_generation")
def generate_signature(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Also tell me a joke about yourself!"}
        ],
    )

    return completion.choices[0].message.content


@workflow(name="pirate_joke_generator")
def joke_workflow():
    eng_joke = create_joke()
    # pirate_joke = translate_joke_to_pirate(eng_joke)
    # Use keyword arguments to ensure Traceloop captures the input correctly
    print(translate_joke_to_pirate(joke=eng_joke))
    signature = generate_signature(joke=eng_joke)
    print(eng_joke + "\n\n" + signature)


if __name__ == "__main__":
    import time

    # run_example()
    joke_workflow()

    # Deepeval runs evaluations in a background thread - if we exit too early,
    # evaluation results won't be emitted
    print("\nWaiting for evaluations to complete...")
    try:
        handler = get_telemetry_handler()
        handler.wait_for_evaluations(timeout=60.0)  # Wait up to 60 seconds
        print("Evaluations completed.")
    except Exception as e:
        print(f"[WARN] Could not wait for evaluations: {e}")

    # Give time for log records to be flushed
    print("Flushing log records...")
    time.sleep(2)
    try:
        logger_provider.force_flush()
    except Exception:
        pass
    print(" Done.")
