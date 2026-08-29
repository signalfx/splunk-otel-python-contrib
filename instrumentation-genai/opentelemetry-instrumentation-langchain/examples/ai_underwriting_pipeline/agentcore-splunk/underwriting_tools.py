"""Mock underwriting data sources, exposed as LangChain tools.

Deterministic per applicant id: the same id always yields the same credit score, claims
history and property risk. That matters for a demo -- an evaluation score that moves
because the *data* moved is not evidence of anything, so the only intended source of
variation is the quality-noise injection in the pipeline itself.

No real data and no network calls. Every value is derived from a hash of the applicant id.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.tools import tool

_FLOOD_ZONES = ["X", "AE", "A", "VE"]
_ROOF_AGES = [2, 6, 11, 18, 27]
_OCCUPANCY = ["owner_occupied", "tenant_occupied", "seasonal", "vacant"]


def _h(applicant_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{applicant_id}:{salt}".encode()).hexdigest()[:8], 16)


def _tool_payload(**kw: Any) -> str:
    return json.dumps(kw, separators=(",", ":"))


@tool
def lookup_credit_score(applicant_id: str) -> str:
    """Return the applicant's credit score band and thin-file flag."""
    score = 520 + _h(applicant_id, "credit") % 300  # 520..819
    return _tool_payload(
        applicant_id=applicant_id,
        credit_score=score,
        band="poor" if score < 600 else "fair" if score < 680 else "good" if score < 740 else "excellent",
        thin_file=_h(applicant_id, "thin") % 10 == 0,
    )


@tool
def fetch_claims_history(applicant_id: str) -> str:
    """Return the applicant's prior claims over the last five years."""
    n = _h(applicant_id, "claims") % 4
    claims = []
    for i in range(n):
        amt = 800 + _h(applicant_id, f"amt{i}") % 24000
        kinds = ["water_damage", "theft", "fire", "wind_hail", "liability"]
        claims.append(
            {
                "year": 2021 + (_h(applicant_id, f"yr{i}") % 5),
                "type": kinds[_h(applicant_id, f"kind{i}") % len(kinds)],
                "paid_amount_usd": amt,
            }
        )
    return _tool_payload(
        applicant_id=applicant_id,
        claim_count_5y=n,
        claims=claims,
        total_paid_usd=sum(c["paid_amount_usd"] for c in claims),
    )


@tool
def fetch_property_risk(applicant_id: str, address: str) -> str:
    """Return catastrophe exposure and construction risk for the insured property."""
    return _tool_payload(
        applicant_id=applicant_id,
        address=address,
        flood_zone=_FLOOD_ZONES[_h(applicant_id, "flood") % len(_FLOOD_ZONES)],
        wildfire_score=_h(applicant_id, "fire") % 100,
        roof_age_years=_ROOF_AGES[_h(applicant_id, "roof") % len(_ROOF_AGES)],
        occupancy=_OCCUPANCY[_h(applicant_id, "occ") % len(_OCCUPANCY)],
        distance_to_hydrant_m=30 + _h(applicant_id, "hyd") % 500,
    )


@tool
def check_underwriting_guidelines(product: str, risk_tier: str) -> str:
    """Return the binding guideline for a product and risk tier: appetite, limits, referral rules."""
    tiers = {
        "preferred": {"appetite": "accept", "max_limit_usd": 1_500_000, "referral_required": False},
        "standard": {"appetite": "accept", "max_limit_usd": 750_000, "referral_required": False},
        "substandard": {"appetite": "accept_with_conditions", "max_limit_usd": 300_000, "referral_required": True},
        "declined": {"appetite": "decline", "max_limit_usd": 0, "referral_required": True},
    }
    g = tiers.get(risk_tier.lower(), tiers["standard"])
    return _tool_payload(
        product=product,
        risk_tier=risk_tier,
        **g,
        mandatory_exclusions=["flood" if risk_tier.lower() != "preferred" else "none"],
        note="Assistant may recommend only. Binding requires a licensed underwriter.",
    )


@tool
def calculate_premium(base_rate_usd: float, risk_multiplier: float, coverage_limit_usd: float) -> str:
    """Compute an indicative annual premium from base rate, risk multiplier and limit."""
    exposure_units = max(coverage_limit_usd, 0) / 1000.0
    premium = base_rate_usd * max(risk_multiplier, 0.1) * exposure_units / 100.0
    return _tool_payload(
        base_rate_usd=base_rate_usd,
        risk_multiplier=risk_multiplier,
        coverage_limit_usd=coverage_limit_usd,
        indicative_annual_premium_usd=round(premium, 2),
        basis="indicative only; not a bindable quote",
    )


ALL_TOOLS = [
    lookup_credit_score,
    fetch_claims_history,
    fetch_property_risk,
    check_underwriting_guidelines,
    calculate_premium,
]

# The steps a complete underwriting pass must perform. The action-completeness evaluation
# in the GenAI Lens is "did the pipeline actually do all of these", so the list is defined
# once here and both the pipeline and its self-check read it -- otherwise the metric drifts
# from the behaviour it claims to measure.
REQUIRED_ACTIONS = (
    "credit_checked",
    "claims_reviewed",
    "property_assessed",
    "guidelines_applied",
    "premium_calculated",
    "decision_recorded",
)
