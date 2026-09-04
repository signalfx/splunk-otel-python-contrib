"""Fraud-detection agent on Bedrock AgentCore, instrumented with AWS's own ADOT distro.

See ``claims_intake_coding/agentcore-adot/main.py`` for the instrumentation rationale --
this agent is the same ADOT-only pattern applied to a different business task, deliberately
kept as a second, independently deployed AgentCore runtime rather than a second code path in
the same one: the onboarding wizard's discovery step is meant to find multiple distinct
agents reporting into the same Agent Observability project, not one agent doing two things.
"""

from __future__ import annotations

import json
import os
import sys
import time
from uuid import uuid4

from bedrock_agentcore import BedrockAgentCoreApp
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.trace import Status, StatusCode

OpenAIInstrumentor().instrument()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
AGENTCORE_RUNTIME_ID = os.environ.get("AGENTCORE_RUNTIME_ID", "")
FEATURE_ID = os.environ.get("GENAI_LENS_FEATURE_ID", "Fraud Detection")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "fraud-detection")

_client = OpenAI()

app = BedrockAgentCoreApp()

_FRAUD_SYSTEM = """You are an insurance fraud-detection assistant. Given structured claim
facts, assess a fraud risk score from 0 (no indicators) to 100 (strong indicators) and list
the specific rationale. Consider timing inconsistencies, prior claim frequency, mismatched
damage descriptions, and any explicit red flags in the facts. Never accuse the claimant of a
crime; report indicators and a recommendation only (clear / flag_for_review / escalate).
Respond with compact JSON only, keys: fraud_score, recommendation, rationale (list of
strings)."""


def _tracer():
    return trace.get_tracer("fraud.detection")


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Score one claim for fraud risk. Root span carries the genai_lens feature attribution."""
    claim_facts = payload.get("claim_facts") or {
        "claim_id": "CLM-90001",
        "days_since_policy_start": 9,
        "claim_count_5y": 4,
        "reported_loss_usd": 48000,
        "description": "Total loss of newly acquired jewelry, no receipts available, "
        "claimant requests expedited cash settlement.",
    }
    session_id = payload.get("session_id") or str(uuid4())
    claim_id = str(claim_facts.get("claim_id") or f"CLM-{time.time_ns() % 10_000_000}")

    started = time.time()
    with _tracer().start_as_current_span("AgentInvocation.fraud_detection") as root:
        root.set_attribute("gen_ai.operation.name", "chain")
        root.set_attribute("gen_ai.agent.name", "fraud-detection")
        root.set_attribute("session.id", session_id)
        root.set_attribute("fraud.claim_id", claim_id)
        if AGENTCORE_RUNTIME_ID:
            root.set_attribute("aws.agentcore.runtime", AGENTCORE_RUNTIME_ID)
        root.set_attribute("service.name", SERVICE_NAME)
        root.set_attribute("genai_lens.feature_id", FEATURE_ID)
        try:
            response = _client.chat.completions.create(
                model=MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": _FRAUD_SYSTEM},
                    {"role": "user", "content": json.dumps(claim_facts, separators=(",", ":"))},
                ],
            )
            text = response.choices[0].message.content or "{}"
            try:
                assessment = json.loads(text)
            except json.JSONDecodeError:
                assessment = {"raw": text}

            score = assessment.get("fraud_score")
            root.set_attribute("fraud.score", score if isinstance(score, (int, float)) else -1)
            root.set_attribute(
                "fraud.recommendation", str(assessment.get("recommendation") or "unknown")
            )
            root.set_attribute("fraud.duration_ms", round((time.time() - started) * 1000, 1))
            root.set_status(Status(StatusCode.OK))
            return {
                "claim_id": claim_id,
                "session_id": session_id,
                "assessment": assessment,
            }
        except Exception as exc:
            root.set_status(Status(StatusCode.ERROR, str(exc)))
            root.record_exception(exc)
            raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for i in range(n):
            out = invoke({"claim_facts": {"claim_id": f"CLM-{70000 + i}"}})
            print(json.dumps(out, indent=2)[:900])
            print("-" * 60)
    else:
        app.run()
