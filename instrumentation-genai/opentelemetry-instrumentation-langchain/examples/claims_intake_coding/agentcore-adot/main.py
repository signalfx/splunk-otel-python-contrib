"""Claims intake coding agent on Bedrock AgentCore, instrumented with AWS's own ADOT distro.

Deliberately the AWS-standardized instrumentation path rather than the Splunk OTel stack the
sibling ``ai_underwriting_pipeline`` example uses: this process is started under
``opentelemetry-instrument`` (see ``deploy.sh`` / ``patch_dockerfile.py``), and the only
instrumentation registered explicitly here is the community ``opentelemetry-instrumentation-
openai`` package for the raw OpenAI client. No LangChain, no Splunk GenAI emitters -- those
pin ``opentelemetry-instrumentation``/``opentelemetry-semantic-conventions`` to the ``~0.57b0``
generation, which conflicts with ADOT's own pin (``0.19.0`` -> ``0.65b0``); this agent avoids
that conflict by simply not depending on either Splunk package.

Business logic is intentionally thin: one call extracts structured intake fields from a
free-text claim description. The point of this example is the instrumentation path, not the
coding logic.
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

# Explicit instrument() call rather than relying solely on opentelemetry-instrument's
# entry-point auto-discovery: it is idempotent (BaseInstrumentor no-ops on a second call) and
# means this still works if the process is ever run without the wrapper for local smoke tests.
OpenAIInstrumentor().instrument()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# Same coverage-join constraint as ai_underwriting_pipeline: AgentCore only hands the
# container a runtime id if deploy.sh already knew it, which means a first-ever deploy runs
# without it.
AGENTCORE_RUNTIME_ID = os.environ.get("AGENTCORE_RUNTIME_ID", "")
FEATURE_ID = os.environ.get("GENAI_LENS_FEATURE_ID", "Claims Intake Coding")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "claims-intake-coding")

_client = OpenAI()

app = BedrockAgentCoreApp()

_CODING_SYSTEM = """You are an insurance claims intake coding assistant. Given a free-text
claim description, extract the structured fields a claims examiner needs next:
claim_type, loss_description (one sentence), estimated_severity (low/medium/high),
suggested_coverage_line, missing_information (list of strings, empty if nothing is missing).
Respond with compact JSON only, no prose."""


def _tracer():
    return trace.get_tracer("claims.intake.coding")


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Code one claim intake. Root span carries the genai_lens feature attribution."""
    claim_text = payload.get("claim_text") or (
        "Water heater burst in the basement overnight, flooding the finished basement and "
        "damaging drywall, flooring, and stored furniture. Homeowner noticed it this morning."
    )
    session_id = payload.get("session_id") or str(uuid4())
    claim_id = payload.get("claim_id") or f"CLM-{time.time_ns() % 10_000_000}"

    started = time.time()
    with _tracer().start_as_current_span("AgentInvocation.claims_intake_coding") as root:
        root.set_attribute("gen_ai.operation.name", "chain")
        root.set_attribute("gen_ai.agent.name", "claims-intake-coding")
        root.set_attribute("session.id", session_id)
        root.set_attribute("claims.claim_id", claim_id)
        # The same coverage-join key ai_underwriting_pipeline sets, and for the same reason:
        # it has to be a span attribute because AO drops resource attributes.
        if AGENTCORE_RUNTIME_ID:
            root.set_attribute("aws.agentcore.runtime", AGENTCORE_RUNTIME_ID)
        root.set_attribute("service.name", SERVICE_NAME)
        root.set_attribute("genai_lens.feature_id", FEATURE_ID)
        try:
            response = _client.chat.completions.create(
                model=MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": _CODING_SYSTEM},
                    {"role": "user", "content": claim_text},
                ],
            )
            text = response.choices[0].message.content or "{}"
            try:
                coded = json.loads(text)
            except json.JSONDecodeError:
                coded = {"raw": text}

            root.set_attribute("claims.coding_produced", bool(coded))
            root.set_attribute("claims.duration_ms", round((time.time() - started) * 1000, 1))
            root.set_status(Status(StatusCode.OK))
            return {
                "claim_id": claim_id,
                "session_id": session_id,
                "coded": coded,
            }
        except Exception as exc:
            root.set_status(Status(StatusCode.ERROR, str(exc)))
            root.record_exception(exc)
            raise


if __name__ == "__main__":
    # Same rule as ai_underwriting_pipeline: app.run() is the default because AgentCore
    # starts the container as `python -m main` and health-checks the HTTP server it brings
    # up. Local smoke runs are opt-in.
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for i in range(n):
            out = invoke({"claim_id": f"CLM-{80000 + i}"})
            print(json.dumps(out, indent=2)[:900])
            print("-" * 60)
    else:
        app.run()
