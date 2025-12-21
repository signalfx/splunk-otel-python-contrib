#!/usr/bin/env python3

import os
import traceback

import openlit
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

try:
    openlit.init(otlp_endpoint="http://0.0.0.0:4318")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is LLM Observability?",
            }
        ],
        model="gpt-3.5-turbo",
    )
    print("response:", chat_completion.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
