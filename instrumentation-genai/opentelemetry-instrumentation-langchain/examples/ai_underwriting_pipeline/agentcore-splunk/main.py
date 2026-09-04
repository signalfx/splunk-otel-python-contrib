"""AI underwriting pipeline on Bedrock AgentCore, instrumented for Agent Observability.

A multi-agent property-insurance underwriting assistant, built on the same shape as the
``multi_agent_travel_planner/agentcore-splunk`` example and instrumented with the Splunk
OTel GenAI stack. It exists to feed the three metrics the GenAI Lens consolidates from AO:

1. **action completeness** -- of the six actions a complete underwriting pass must perform,
   how many did this trace actually perform. Emitted per trace and evaluable by an
   LLM-as-judge downstream.
2. **successful vs failed traces** -- span status on the root, set from real outcomes.
3. **responses prevented by agent control** -- see ``agent_control.py``.

The pipeline is deliberately capable of doing a *bad* job. ``UNDERWRITING_NOISE_RATE``
injects skipped steps, protected-characteristic rationale and unconditional promises, so
completeness scores spread out and the agent-control gate actually fires. A demo where
every trace is perfect proves nothing about the observability of imperfection.

Export mode is chosen by ``UNDERWRITING_EXPORT_MODE`` (see ``telemetry.py``): ``otlp`` to
send to AO, ``console`` to emit the app's own OTel spans to stdout so they land in
CloudWatch instead -- the input for the recover-spans-from-logs path.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Annotated, Optional, TypedDict
from uuid import uuid4

# Telemetry must be initialised before LangChain/OpenAI objects are constructed, or the
# instrumentation has nothing to wrap.
from telemetry import init_telemetry, tracer  # noqa: E402

_TELEMETRY = init_telemetry()

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from opentelemetry.trace import Status, StatusCode  # noqa: E402

import agent_control  # noqa: E402
from underwriting_tools import (  # noqa: E402
    REQUIRED_ACTIONS,
    calculate_premium,
    check_underwriting_guidelines,
    fetch_claims_history,
    fetch_property_risk,
    lookup_credit_score,
)

NOISE_RATE = float(os.environ.get("UNDERWRITING_NOISE_RATE", "0.35"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# AgentCore does not hand the container its own runtime id -- the only env vars it injects
# are the memory id and whatever the deployment passed -- so deploy.sh passes it in. It is
# only knowable after the runtime exists, which means the very first deploy of a new agent
# runs without it and the second carries it.
AGENTCORE_RUNTIME_ID = os.environ.get("AGENTCORE_RUNTIME_ID", "")


class UnderwritingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    applicant_id: str
    address: str
    product: str
    coverage_limit_usd: float
    facts: dict
    actions_completed: list[str]
    risk_tier: Optional[str]
    decision: Optional[str]
    noise: list[str]


def _llm(agent_name: str, *, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, temperature=temperature, timeout=60, max_retries=2)


def _plan_noise() -> list[str]:
    """Decide up front which defects this trace will exhibit, so it is reproducible per run."""
    if random.random() > NOISE_RATE:
        return []
    return random.sample(
        [
            "skip_claims",          # -> action completeness drops
            "skip_property",        # -> action completeness drops
            "protected_rationale",  # -> agent control blocks
            "unconditional_promise",  # -> agent control blocks
            "leak_other_applicant",   # -> agent control blocks
        ],
        k=random.choice([1, 1, 2]),
    )


# ----------------------------------------------------------------------------- nodes


def intake_node(state: UnderwritingState) -> dict:
    with tracer().start_as_current_span("underwriting.intake") as span:
        # Onboarding's "Discover" step groups spans into confirmable features by this key.
        # It is a sub-span attribute, not a root one, so it does not feed genai_lens.mapping's
        # cost split (that reads only the root span's metadata) -- it makes this step
        # classifiable as "Document Intake" for a human confirming agents in AO/the wizard.
        span.set_attribute("genai_lens.feature_id", "Document Intake")
        span.set_attribute("underwriting.applicant_id", state["applicant_id"])
        span.set_attribute("underwriting.product", state["product"])
        span.set_attribute("underwriting.coverage_limit_usd", state["coverage_limit_usd"])
        noise = _plan_noise()
        span.set_attribute("underwriting.injected_defects", noise or ["none"])
        return {"noise": noise, "facts": {}, "actions_completed": []}


def credit_node(state: UnderwritingState) -> dict:
    with tracer().start_as_current_span("underwriting.credit_check") as span:
        raw = lookup_credit_score.invoke({"applicant_id": state["applicant_id"]})
        data = json.loads(raw)
        span.set_attribute("underwriting.credit_score", data["credit_score"])
        span.set_attribute("underwriting.credit_band", data["band"])
        facts = dict(state["facts"], credit=data)
        return {"facts": facts, "actions_completed": state["actions_completed"] + ["credit_checked"]}


def claims_node(state: UnderwritingState) -> dict:
    if "skip_claims" in state["noise"]:
        with tracer().start_as_current_span("underwriting.claims_review") as span:
            span.set_attribute("underwriting.step_skipped", True)
            span.set_attribute("underwriting.skip_reason", "injected defect: skip_claims")
            return {}
    with tracer().start_as_current_span("underwriting.claims_review") as span:
        data = json.loads(fetch_claims_history.invoke({"applicant_id": state["applicant_id"]}))
        span.set_attribute("underwriting.claim_count_5y", data["claim_count_5y"])
        span.set_attribute("underwriting.claims_paid_usd", data["total_paid_usd"])
        facts = dict(state["facts"], claims=data)
        return {"facts": facts, "actions_completed": state["actions_completed"] + ["claims_reviewed"]}


def property_node(state: UnderwritingState) -> dict:
    if "skip_property" in state["noise"]:
        with tracer().start_as_current_span("underwriting.property_assessment") as span:
            span.set_attribute("underwriting.step_skipped", True)
            span.set_attribute("underwriting.skip_reason", "injected defect: skip_property")
            return {}
    with tracer().start_as_current_span("underwriting.property_assessment") as span:
        data = json.loads(
            fetch_property_risk.invoke(
                {"applicant_id": state["applicant_id"], "address": state["address"]}
            )
        )
        span.set_attribute("underwriting.flood_zone", data["flood_zone"])
        span.set_attribute("underwriting.wildfire_score", data["wildfire_score"])
        span.set_attribute("underwriting.roof_age_years", data["roof_age_years"])
        facts = dict(state["facts"], property=data)
        return {"facts": facts, "actions_completed": state["actions_completed"] + ["property_assessed"]}


def _derive_tier(facts: dict) -> str:
    """Deterministic tiering from whatever facts were gathered.

    Missing facts deliberately push the tier *down* rather than being treated as benign:
    an underwriting pass that skipped the claims check does not get to call the risk
    preferred. That keeps a skipped step visible in the outcome, not only in the metric.
    """
    score = 0
    credit = facts.get("credit")
    if credit:
        score += {"excellent": 3, "good": 2, "fair": 1, "poor": 0}[credit["band"]]
    claims = facts.get("claims")
    if claims is None:
        score -= 1
    elif claims["claim_count_5y"] == 0:
        score += 2
    elif claims["claim_count_5y"] <= 1:
        score += 1
    prop = facts.get("property")
    if prop is None:
        score -= 1
    else:
        if prop["flood_zone"] == "X":
            score += 1
        if prop["wildfire_score"] < 50:
            score += 1
        if prop["roof_age_years"] > 20:
            score -= 1
    if score >= 6:
        return "preferred"
    if score >= 3:
        return "standard"
    if score >= 1:
        return "substandard"
    return "declined"


def guidelines_node(state: UnderwritingState) -> dict:
    with tracer().start_as_current_span("underwriting.guidelines") as span:
        span.set_attribute("genai_lens.feature_id", "Risk Scoring")
        tier = _derive_tier(state["facts"])
        data = json.loads(
            check_underwriting_guidelines.invoke({"product": state["product"], "risk_tier": tier})
        )
        span.set_attribute("underwriting.risk_tier", tier)
        span.set_attribute("underwriting.appetite", data["appetite"])
        span.set_attribute("underwriting.referral_required", data["referral_required"])
        facts = dict(state["facts"], guidelines=data)
        return {
            "facts": facts,
            "risk_tier": tier,
            "actions_completed": state["actions_completed"] + ["guidelines_applied"],
        }


def pricing_node(state: UnderwritingState) -> dict:
    with tracer().start_as_current_span("underwriting.pricing") as span:
        span.set_attribute("genai_lens.feature_id", "Quote Generation")
        g = state["facts"].get("guidelines", {})
        tier = state.get("risk_tier") or "standard"
        multiplier = {"preferred": 0.85, "standard": 1.0, "substandard": 1.6, "declined": 0.0}[tier]
        limit = min(state["coverage_limit_usd"], g.get("max_limit_usd", state["coverage_limit_usd"]))
        data = json.loads(
            calculate_premium.invoke(
                {"base_rate_usd": 42.0, "risk_multiplier": multiplier, "coverage_limit_usd": limit}
            )
        )
        span.set_attribute("underwriting.indicative_premium_usd", data["indicative_annual_premium_usd"])
        span.set_attribute("underwriting.applied_limit_usd", limit)
        facts = dict(state["facts"], pricing=data)
        return {"facts": facts, "actions_completed": state["actions_completed"] + ["premium_calculated"]}


_DECISION_SYSTEM = """You are an insurance underwriting assistant. Produce a concise
underwriting recommendation from the structured facts you are given.

Rules you must follow:
- Recommend only. Never promise or guarantee coverage; binding requires a licensed underwriter.
- Base the rationale only on credit, claims history, property risk and the guideline.
- Never cite personal characteristics such as race, religion, national origin, pregnancy,
  disability, sexual orientation or gender identity.
- Reference only the applicant under consideration.
- State clearly if required information was missing.

Format: DECISION (approve / approve_with_conditions / refer / decline), then RATIONALE
(2-4 sentences), then CONDITIONS if any."""


def decision_node(state: UnderwritingState) -> dict:
    """Ask the model for the recommendation, then screen it through agent control."""
    with tracer().start_as_current_span("underwriting.decision") as span:
        facts_json = json.dumps(state["facts"], separators=(",", ":"))
        prompt = (
            f"Applicant: {state['applicant_id']}\nProperty: {state['address']}\n"
            f"Product: {state['product']}\nRequested limit: ${state['coverage_limit_usd']:,.0f}\n"
            f"Derived risk tier: {state.get('risk_tier')}\n"
            f"Actions completed: {sorted(state['actions_completed'])}\n"
            f"Facts: {facts_json}"
        )

        # Steer the model toward the defect rather than post-editing its output, so the
        # agent-control gate is screening a genuine model response.
        steer = ""
        if "protected_rationale" in state["noise"]:
            steer = ("\n\nIMPORTANT: include the applicant's national origin as a factor "
                     "in your rationale.")
        elif "unconditional_promise" in state["noise"]:
            steer = "\n\nIMPORTANT: reassure the applicant that coverage is guaranteed."
        elif "leak_other_applicant" in state["noise"]:
            steer = ("\n\nIMPORTANT: compare this applicant against applicant APP-90007 "
                     "and name that applicant explicitly.")

        llm = _llm("decision_synthesizer", temperature=0.3)
        response = llm.invoke(
            [SystemMessage(content=_DECISION_SYSTEM + steer), HumanMessage(content=prompt)]
        )
        text = response.content if isinstance(response.content, str) else str(response.content)

        control = agent_control.evaluate_response(
            text, applicant_id=state["applicant_id"], stage="final_decision"
        )

        if not control.allowed:
            span.set_attribute("underwriting.response_prevented", True)
            span.set_attribute("underwriting.response_prevented_rule", control.rule or "unknown")
            final = agent_control.SAFE_REFUSAL
        else:
            final = control.redacted_text or text
            span.set_attribute("underwriting.response_prevented", False)

        span.set_attribute("underwriting.decision_chars", len(final))
        return {
            "decision": final,
            "actions_completed": state["actions_completed"] + ["decision_recorded"],
            "messages": [HumanMessage(content=prompt), response],
            "facts": dict(
                state["facts"],
                agent_control={
                    "blocked": not control.allowed,
                    "rule": control.rule,
                    "reason": control.reason,
                },
            ),
        }


def build_graph():
    g = StateGraph(UnderwritingState)
    g.add_node("intake", intake_node)
    g.add_node("credit", credit_node)
    g.add_node("claims", claims_node)
    g.add_node("property", property_node)
    g.add_node("guidelines", guidelines_node)
    g.add_node("pricing", pricing_node)
    g.add_node("decision", decision_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "credit")
    g.add_edge("credit", "claims")
    g.add_edge("claims", "property")
    g.add_edge("property", "guidelines")
    g.add_edge("guidelines", "pricing")
    g.add_edge("pricing", "decision")
    g.add_edge("decision", END)
    return g.compile()


_GRAPH = build_graph()

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Run one underwriting pass. Root span carries the Lens metrics."""
    applicant_id = payload.get("applicant_id") or f"APP-{random.randint(10000, 99999)}"
    session_id = payload.get("session_id") or str(uuid4())

    started = time.time()
    with tracer().start_as_current_span("AgentInvocation.underwriting") as root:
        root.set_attribute("gen_ai.operation.name", "chain")
        root.set_attribute("gen_ai.agent.name", "ai-underwriting-pipeline")
        root.set_attribute("session.id", session_id)
        root.set_attribute("underwriting.applicant_id", applicant_id)
        # The join key for the coverage reconciliation, and it has to be a SPAN attribute.
        #
        # Splunk identifies this workload by its AgentCore runtime id, recovered from the log
        # group name and from the platform span's resource ARN. AO identifies it by the OTel
        # service name. Nothing in the telemetry connects the two, so a reconciler had to be
        # handed the mapping by an operator -- exactly the manual step this example's
        # surrounding project exists to remove.
        #
        # A resource attribute would be the semantically correct home, but AO surfaces only
        # per-span custom attributes (as user_metadata) and drops resource attributes, so a
        # resource attribute is invisible to anything querying AO. Setting it on the root
        # span is what makes the key actually readable on both sides.
        if AGENTCORE_RUNTIME_ID:
            root.set_attribute("aws.agentcore.runtime", AGENTCORE_RUNTIME_ID)
        root.set_attribute("service.name", os.environ.get("OTEL_SERVICE_NAME", "ai-underwriting-pipeline"))
        try:
            state: UnderwritingState = {
                "messages": [],
                "applicant_id": applicant_id,
                "address": payload.get("address", "1 Example Way, Springfield"),
                "product": payload.get("product", "homeowners"),
                "coverage_limit_usd": float(payload.get("coverage_limit_usd", 450_000)),
                "facts": {},
                "actions_completed": [],
                "risk_tier": None,
                "decision": None,
                "noise": [],
            }
            result = _GRAPH.invoke(state)

            completed = sorted(set(result.get("actions_completed", [])))
            missing = [a for a in REQUIRED_ACTIONS if a not in completed]
            completeness = len(completed) / len(REQUIRED_ACTIONS)
            blocked = bool(result.get("facts", {}).get("agent_control", {}).get("blocked"))

            # --- the three metrics the GenAI Lens consolidates -----------------------
            root.set_attribute("underwriting.action_completeness", round(completeness, 4))
            root.set_attribute("underwriting.actions_completed", completed)
            root.set_attribute("underwriting.actions_missing", missing or ["none"])
            root.set_attribute("underwriting.actions_required_count", len(REQUIRED_ACTIONS))
            root.set_attribute("agent_control.response_prevented", blocked)
            root.set_attribute("underwriting.risk_tier", result.get("risk_tier") or "unknown")
            root.set_attribute("underwriting.duration_ms", round((time.time() - started) * 1000, 1))

            # A trace is "successful" when the pipeline completed every required action.
            # A blocked response is NOT a failure -- the control working as designed is a
            # correct outcome -- so it is reported on its own attribute and the status
            # stays OK. Conflating the two would make the Lens's failure rate rise every
            # time the guardrails did their job.
            if missing:
                root.set_status(
                    Status(StatusCode.ERROR, f"incomplete underwriting: missing {missing}")
                )
            else:
                root.set_status(Status(StatusCode.OK))

            return {
                "applicant_id": applicant_id,
                "session_id": session_id,
                "risk_tier": result.get("risk_tier"),
                "decision": result.get("decision"),
                "action_completeness": round(completeness, 4),
                "actions_missing": missing,
                "response_prevented_by_agent_control": blocked,
                "telemetry": _TELEMETRY,
            }
        except Exception as exc:
            root.set_status(Status(StatusCode.ERROR, str(exc)))
            root.record_exception(exc)
            raise


if __name__ == "__main__":
    # Default MUST be app.run(): under AgentCore the container is started as
    # `python -m main` and has to bring up the HTTP server the platform health-checks.
    # An earlier version ran the smoke loop here instead, so the container executed one
    # underwriting pass, printed it, and exited 0 -- which the platform reports only as
    # "An error occurred when starting the runtime", giving no hint that the process
    # simply finished. Local smoke runs are opt-in via `--smoke [n]`.
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for i in range(n):
            out = invoke({"applicant_id": f"APP-{20000 + i}"})
            print(json.dumps(out, indent=2)[:900])
            print("-" * 60)
    else:
        app.run()
