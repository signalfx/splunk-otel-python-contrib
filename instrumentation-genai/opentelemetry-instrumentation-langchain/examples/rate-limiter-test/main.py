from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import time
import re
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Reduce noisy debug logging during local testing
logging.basicConfig(level=logging.INFO)
for noisy_logger in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Print rate limiting config to verify
print("=" * 80)
print("Rate Limiting Configuration:")
print(f"  RPS: {os.getenv('OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS')}")
print(f"  BURST: {os.getenv('OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST')}")
print(f"  EVALUATORS: {os.getenv('OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS')}")
print("=" * 80)
print()


def main():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "What is the tallest mountain?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "What is Python?",
        "What is the largest ocean?",
        "What year did WWII end?",
    ]

    print("Making 8 LLM calls to trigger rate limiting...\n")
    for i, question in enumerate(questions, 1):
        messages = [
            SystemMessage(content="You are a helpful assistant!"),
            HumanMessage(content=question),
        ]
        result = llm.invoke(messages).content
        print(f"Call {i}: {question}")
        print(f"Answer: {result}\n")

    # Estimate expected evaluation count from evaluator config
    eval_spec = os.getenv("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", "")
    metrics_count = 0
    match = re.search(r"LLMInvocation\(([^)]*)\)", eval_spec)
    if match:
        metrics = [m.strip() for m in match.group(1).split(",") if m.strip()]
        metrics_count = len(metrics)
    expected_evals = len(questions) * metrics_count if metrics_count else None

    # Wait for background evaluations to complete based on rate limit settings
    rps_raw = os.getenv("OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS", "1")
    burst_raw = os.getenv("OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST", "4")
    try:
        rps = float(rps_raw)
    except ValueError:
        rps = 1.0
    try:
        burst = int(float(burst_raw))
    except ValueError:
        burst = 4

    # Approximate time for queued evaluations to be admitted by the rate limiter.
    queued = max(0, len(questions) - max(burst, 0))
    if rps <= 0:
        wait_seconds = 120
    else:
        wait_seconds = int(queued / rps + 10)
    wait_seconds = max(10, min(wait_seconds, 300))

    limiter_state = "disabled" if rps <= 0 else f"rps={rps}, burst={burst}"
    if expected_evals is not None:
        print(f"Expected evaluations: {expected_evals}")
    print(f"Rate limiter: {limiter_state}")
    print(f"Waiting {wait_seconds}s for evaluations to complete...")
    time.sleep(wait_seconds)
    print("Done!")


if __name__ == "__main__":
    main()
