#!/usr/bin/env python3

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load .env first
load_dotenv()

try:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.decorators import task, workflow, agent, tool
    from openai import OpenAI

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
        messages=[{"role": "user", "content": f"get some history jokes"}],
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
    print(translate_joke_to_pirate(eng_joke))
    signature = generate_signature(eng_joke)
    print(eng_joke + "\n\n" + signature)


if __name__ == "__main__":
    # run_example()
    joke_workflow()
