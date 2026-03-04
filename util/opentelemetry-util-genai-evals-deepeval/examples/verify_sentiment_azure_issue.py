#!/usr/bin/env python3
# ruff: noqa: E402
"""Verify sentiment evaluation consistency across LLM providers.

Runs two test cases (flight booking responses) through the sentiment metric
to check whether the score, label, and explanation are internally consistent
across different LLM judge backends.

Usage
-----
Configure one of the provider blocks below via environment variables, then run:

    python verify_sentiment_azure_issue.py

Environment variables (pick ONE provider):

  # --- Azure OpenAI ---
  export DEEPEVAL_LLM_BASE_URL="https://<your-azure-endpoint>.openai.azure.com/"
  export DEEPEVAL_LLM_MODEL="gpt-4o"
  export DEEPEVAL_LLM_PROVIDER="azure"
  export DEEPEVAL_LLM_API_KEY="<your-azure-api-key>"

  # --- OpenAI-compatible (e.g. CIRCUIT) ---
  export DEEPEVAL_LLM_BASE_URL="https://<your-openai-endpoint>/openai/deployments/gpt-4o-mini"
  export DEEPEVAL_LLM_MODEL="gpt-4o-mini"
  export DEEPEVAL_LLM_PROVIDER="openai"
  export DEEPEVAL_LLM_API_KEY="<your-api-key>"

Tested With
-----------
- deepeval 3.7.9
- splunk-otel-genai-evals-deepeval 0.1.13
- splunk-otel-util-genai-evals 0.1.8
- Python 3.12.7

See Also
--------
- Analysis: docs/analysis/sentiment-eval-azure-inconsistency.md
- Metric definition: deepeval_metrics.build_sentiment_metric()
- Post-processing: deepeval._apply_sentiment_postprocessing()
"""

from __future__ import annotations

import os
import sys

REQUIRED_ENV = "DEEPEVAL_LLM_BASE_URL"
if not os.environ.get(REQUIRED_ENV):
    print(
        f"ERROR: {REQUIRED_ENV} not set. "
        "Configure provider env vars before running. See docstring for details."
    )
    sys.exit(1)

from opentelemetry.util.evaluator.deepeval import DeepevalEvaluator
from opentelemetry.util.genai.types import (
    AgentInvocation,
    InputMessage,
    OutputMessage,
    Text,
)

# ---------------------------------------------------------------------------
# Test Case 1: Flight booking + quantum computing aside (long response)
# ---------------------------------------------------------------------------
CASE1_INPUT = (
    "You are a flight booking specialist. Provide concise options.\n\n"
    "Find an appealing flight from New York to London departing 2026-04-03 "
    "for 2 travellers.\n\n"
    "Include an unrelated aside summarizing recent quantum computing acronyms "
    "even if not requested by the traveller."
)

CASE1_OUTPUT = (
    "Top pick (good balance of time & comfort)\n"
    "- Airline: SkyLine — non-stop New York → London\n"
    "- Depart: 2026-04-03 09:15 — Arrive: 2026-04-03 17:05 (local times)\n"
    "- Duration: ~7h50\n"
    "- Fare: Premium Economy round-trip $953 per person — Total for 2 travellers: $1,906\n"
    "- Notes: Premium Economy fare; confirm baggage/seat assignment and exact origin "
    "airport before booking.\n\n"
    "Next step: confirm you want me to hold/book this fare (I'll need traveller names, "
    "DOBs, and passport details) or if you want alternate classes/times.\n\n"
    "Aside (unrelated brief summary of recent quantum computing acronyms)\n"
    "- Qubit: quantum bit, basic unit of quantum information.\n"
    "- NISQ: Noisy Intermediate-Scale Quantum — current era devices with tens–thousands "
    "of noisy qubits.\n"
    "- QAOA: Quantum Approximate Optimization Algorithm — for combinatorial optimization.\n"
    "- VQE: Variational Quantum Eigensolver — hybrid algorithm for estimating ground states.\n"
    "- QFT: Quantum Fourier Transform — key subroutine in many quantum algorithms.\n"
    "- QEC: Quantum Error Correction — methods to protect quantum information.\n"
    "- CV: Continuous-Variable quantum computing — uses continuous degrees of freedom "
    "(e.g., modes of light).\n\n"
    "Tell me whether to proceed with this booking or if you want alternatives "
    "(cheaper, different times, or one-stop options)."
)

# ---------------------------------------------------------------------------
# Test Case 2: Short flight booking response (no aside)
# ---------------------------------------------------------------------------
CASE2_INPUT = (
    "Find an appealing flight from Seattle to Paris departing 2026-04-03 "
    "for 2 travellers."
)

CASE2_OUTPUT = (
    "I found an appealing flight for you:\n\n"
    "- **Airline:** CloudNine\n"
    "- **Flight Type:** Non-stop service\n"
    "- **Departure:** Seattle on April 3, 2026, at 09:15 AM\n"
    "- **Arrival:** Paris on April 3, 2026, at 05:05 PM\n"
    "- **Fare:** Premium economy fare of $745 return for 2 travelers.\n\n"
    "Would you like more information or assistance with anything else?"
)


TEST_CASES = [
    ("Case 1: NY→London + quantum aside", CASE1_INPUT, CASE1_OUTPUT),
    ("Case 2: Seattle→Paris (short)", CASE2_INPUT, CASE2_OUTPUT),
]


def run_eval(label: str, invocation, metrics: list[str]) -> None:
    """Run evaluation and print results."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    evaluator = DeepevalEvaluator(
        metrics=metrics, invocation_type=type(invocation).__name__
    )
    try:
        results = evaluator.evaluate(invocation)
        for r in results:
            print(f"\n  Metric: {r.metric_name}")
            print(f"  Score:  {r.score}")
            print(f"  Label:  {r.label}")
            passed = r.attributes.get("gen_ai.evaluation.passed")
            print(f"  Passed: {passed}")
            if r.explanation:
                print(f"  Explanation: {r.explanation[:300]}...")
            if r.error:
                print(f"  Error: {r.error}")
    except Exception as e:
        import traceback

        print(f"\n  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


def build_agent(input_text: str, output_text: str) -> AgentInvocation:
    inv = AgentInvocation(name="flight_specialist")
    inv.input_messages.append(
        InputMessage(role="user", parts=[Text(content=input_text)])
    )
    inv.output_messages.append(
        OutputMessage(
            role="assistant",
            parts=[Text(content=output_text)],
            finish_reason="stop",
        )
    )
    return inv


def main() -> None:
    provider = os.environ.get("DEEPEVAL_LLM_PROVIDER", "unknown")
    model = os.environ.get("DEEPEVAL_LLM_MODEL", "unknown")
    base_url = os.environ.get("DEEPEVAL_LLM_BASE_URL", "")

    print(f"Provider: {provider}")
    print(f"Model:    {model}")
    print(f"Endpoint: {base_url[:40]}...")

    all_metrics = [
        "bias",
        "toxicity",
        "answer_relevancy",
        "hallucination",
        "sentiment",
    ]

    for case_label, input_text, output_text in TEST_CASES:
        agent_inv = build_agent(input_text, output_text)
        run_eval(
            f"{case_label} — AgentInvocation (full suite)",
            agent_inv,
            all_metrics,
        )

    print(f"\n{'=' * 70}")
    print(
        "  Expected: neutral/positive text should score >= 0.35"
        " (label='Neutral' or 'Positive', passed=True)"
    )
    print("  If score < 0.35 and label='Negative': LLM judge miscalibration")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
