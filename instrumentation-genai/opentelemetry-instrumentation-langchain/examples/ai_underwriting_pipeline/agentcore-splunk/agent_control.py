"""Agent-control gate for the underwriting pipeline.

The GenAI Lens needs a count of **responses prevented by agent control**, alongside
trace success/failure and evaluation coverage. That number has to come from somewhere
observable, so the gate records every decision as a span with a stable attribute set
rather than merely returning a boolean.

Underwriting is a good domain for this because the things you must not say are concrete
and regulated, not hypothetical:

* a decision that leaks another applicant's data
* an unconditional coverage promise the underwriter has not actually bound
* a rationale that cites a protected characteristic, which is unlawful discrimination
* raw PII (full SSN, full card number) echoed back into the response

This is a deliberately simple deterministic gate. Real Agent Control does far more --
policy sets, SLM classifiers, streaming interception. What matters for the demo is that
a block is *observable in telemetry* with a reason, so AO can aggregate it and the Lens
can chart it. Swapping the rule engine later does not change the telemetry contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from opentelemetry.trace import Status, StatusCode

from telemetry import tracer

# Protected characteristics that may not appear as underwriting rationale. Matching on a
# word boundary keeps "raced" or "originality" from tripping the "race"/"origin" rules.
_PROTECTED = re.compile(
    r"\b(race|racial|ethnicity|ethnic|religion|religious|national origin|"
    r"pregnan\w*|disabilit\w*|sexual orientation|gender identity)\b",
    re.IGNORECASE,
)
# Unconditional commitments. An underwriting assistant may recommend; it may not bind.
_UNCONDITIONAL = re.compile(
    r"\b(guarantee[ds]?|unconditionally approved|we will definitely (cover|pay)|"
    r"you are fully covered|coverage is guaranteed)\b",
    re.IGNORECASE,
)
# Unredacted PII. Deliberately narrow: a 9-digit SSN with separators, and 13-16 digit
# card-like runs. Broad digit matching would flag policy numbers and premiums.
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


@dataclass
class ControlDecision:
    allowed: bool
    reason: Optional[str] = None
    rule: Optional[str] = None
    redacted_text: Optional[str] = None
    matched: list[str] = field(default_factory=list)


def _cross_applicant_leak(text: str, applicant_id: str) -> Optional[str]:
    """Any applicant id in the text that is not the one under consideration."""
    ids = set(re.findall(r"\bAPP-\d{4,}\b", text))
    foreign = sorted(i for i in ids if i != applicant_id)
    return foreign[0] if foreign else None


def evaluate_response(
    text: str,
    *,
    applicant_id: str,
    stage: str = "final_decision",
) -> ControlDecision:
    """Screen an outbound response. Emits one span per evaluation, always."""
    with tracer().start_as_current_span("agent_control.evaluate") as span:
        span.set_attribute("agent_control.stage", stage)
        span.set_attribute("agent_control.applicant_id", applicant_id)
        span.set_attribute("agent_control.input_chars", len(text))

        decision = ControlDecision(allowed=True)

        leaked = _cross_applicant_leak(text, applicant_id)
        if leaked:
            decision = ControlDecision(
                allowed=False,
                rule="cross_applicant_data_leak",
                reason=f"response referenced a different applicant ({leaked})",
                matched=[leaked],
            )
        elif (m := _PROTECTED.search(text)) is not None:
            decision = ControlDecision(
                allowed=False,
                rule="protected_characteristic_in_rationale",
                reason="underwriting rationale cited a protected characteristic",
                matched=[m.group(0)],
            )
        elif (m := _UNCONDITIONAL.search(text)) is not None:
            decision = ControlDecision(
                allowed=False,
                rule="unconditional_coverage_commitment",
                reason="response made a binding coverage promise the assistant cannot make",
                matched=[m.group(0)],
            )
        else:
            # PII is redacted rather than blocked: the decision itself is legitimate and
            # useful, so removing the identifier is a better outcome than refusing.
            redacted, n_ssn = _SSN.subn("[SSN-REDACTED]", text)
            redacted, n_card = _CARD.subn("[CARD-REDACTED]", redacted)
            if n_ssn or n_card:
                decision = ControlDecision(
                    allowed=True,
                    rule="pii_redaction",
                    reason=f"redacted {n_ssn} SSN and {n_card} card-like values",
                    redacted_text=redacted,
                )

        # The telemetry contract the Lens depends on. `agent_control.blocked` is the
        # boolean it aggregates; the rule and reason are what make a spike explainable.
        span.set_attribute("agent_control.blocked", not decision.allowed)
        span.set_attribute("agent_control.rule", decision.rule or "none")
        if decision.reason:
            span.set_attribute("agent_control.reason", decision.reason)
        if decision.matched:
            span.set_attribute("agent_control.matched_terms", decision.matched)
        if decision.redacted_text is not None:
            span.set_attribute("agent_control.redaction_applied", True)

        if not decision.allowed:
            # ERROR status so a blocked response is visible as an abnormal span without
            # having to know the attribute name. It is not a pipeline *failure* -- the
            # gate working is the system behaving correctly -- so the pipeline records it
            # separately from a trace-level error.
            span.set_status(Status(StatusCode.ERROR, decision.reason or "blocked"))
            span.add_event(
                "agent_control.response_prevented",
                {"rule": decision.rule or "unknown", "stage": stage},
            )

        return decision


SAFE_REFUSAL = (
    "This application needs review by a licensed underwriter before a decision can be "
    "shared. A summary of the risk factors has been attached to the case file."
)
