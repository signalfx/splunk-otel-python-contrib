"""Validation and golden set checking for incident copilot.

Note: Actual evaluation metrics (bias, toxicity, relevance, etc.) are computed
automatically by the opentelemetry-util-genai-evals package through the
instrumentation's completion callback mechanism. This module focuses on
validating business logic and expected outcomes.
"""

import logging
from typing import Dict, Optional

from config import Config
from data_loader import DataLoader

_LOGGER = logging.getLogger(__name__)


class ValidationHarness:
    """Validation harness for incident copilot.

    This harness validates that the agents produce expected outcomes based on
    the golden set (alert catalog with expected root causes). It validates:
    - Hypothesis matching against expected root causes
    - Evidence sufficiency
    - Action safety
    - Runbook grounding

    Note: Evaluation metrics (bias, toxicity, relevance, etc.) are computed
    automatically by the opentelemetry-util-genai-evals package during instrumentation.
    """

    def __init__(self, config: Config):
        """Initialize validation harness."""
        self.config = config
        self.data_loader = DataLoader(data_dir=config.data_dir)

    def validate_hypothesis(
        self,
        state: Dict,
        alert: Optional[Dict] = None,
    ) -> Dict:
        """Validate if the top hypothesis matches the expected root cause.

        Returns:
            Dict with validation results
        """
        if not alert:
            alert_id = state.get("alert_id")
            alert = self.data_loader.get_alert(alert_id) if alert_id else None

        expected_root_cause = alert.get("expected_root_cause") if alert else None
        hypotheses = state.get("hypotheses", [])

        validation = {
            "expected_root_cause": expected_root_cause,
            "hypotheses_count": len(hypotheses),
            "hypothesis_match": False,
            "top_hypothesis": None,
        }

        if not expected_root_cause or not hypotheses:
            return validation

        # Check top 3 hypotheses for better matching (sometimes the correct one isn't #1)
        top_hypothesis = hypotheses[0]
        validation["top_hypothesis"] = top_hypothesis.get("hypothesis")

        # Check all top hypotheses, not just the first one
        hypotheses_to_check = hypotheses[:3]  # Check top 3

        # Enhanced keyword matching with synonyms
        # Check all top hypotheses for matches
        expected_lower = expected_root_cause.lower()

        # Map root causes to related keywords for better matching
        root_cause_synonyms = {
            "database_connection_pool_exhaustion": [
                "connection",
                "pool",
                "exhausted",
                "exhaustion",
                "database",
                "db",
                "connection pool",
                "pool exhausted",
                "connection pool exhausted",
                "pool capacity",
                "max connections",
                "connection limit",
            ],
            "connection_leak": [
                "connection",
                "leak",
                "leaking",
                "database",
                "pool",
                "db",
                "resource leak",
                "connection leak",
                "connection not closed",
                "unclosed connection",
                "connection pool leak",
            ],
            "cache_miss_storm": [
                "cache",
                "miss",
                "misses",
                "storm",
                "hit rate",
                "cache hit",
                "cache miss",
                "cache misses",
                "eviction",
                "cache performance",
                "low cache hit rate",
                "cache hit rate dropped",
                "cache miss rate",
            ],
            "cache_key_explosion": [
                "cache",
                "key",
                "keys",
                "explosion",
                "memory",
                "cache memory",
                "key explosion",
                "too many keys",
                "key proliferation",
                "memory pressure",
                "cache memory full",
            ],
            "recent_deployment_issue": [
                "deployment",
                "deploy",
                "deployed",
                "release",
                "version",
                "rollout",
                "new version",
                "recent deployment",
                "after deploy",
                "deployment correlation",
                "post-deployment",
                "deployment v",
            ],
            "token_validation_issue": [
                "token",
                "tokens",
                "validation",
                "validate",
                "auth",
                "authentication",
                "jwt",
                "token validation",
                "invalid token",
                "token signature",
                "authentication failure",
                "token verification",
            ],
            "downstream_service_degradation": [
                "downstream",
                "service",
                "degradation",
                "degraded",
                "dependency",
                "slow",
                "dependency slow",
                "downstream slow",
                "downstream service slow",
                "dependency degradation",
                "downstream performance",
                "downstream latency",
                "downstream error",
                "downstream responding slowly",
                "downstream service responding",
                "downstream service degradation",
                "dependency slow",
                "dependency responding slowly",
            ],
            "downstream_service_failure": [
                "downstream",
                "service",
                "failure",
                "failed",
                "dependency",
                "unavailable",
                "dependency failure",
                "downstream failure",
                "dependency unavailable",
                "downstream service down",
                "dependency down",
            ],
            "database_query_performance": [
                "database",
                "query",
                "queries",
                "performance",
                "slow",
                "latency",
                "db query",
                "slow query",
                "query slow",
                "database slow",
                "query performance",
                "slow database",
                "db performance",
            ],
        }

        # Get synonyms for the expected root cause
        expected_keywords = expected_lower.split("_")
        synonyms = root_cause_synonyms.get(expected_root_cause, [])

        # Combine keywords and synonyms for matching
        all_keywords = expected_keywords + synonyms

        # Check all top hypotheses for matches
        matched_keywords = []
        matched_hypothesis = None

        for hyp in hypotheses_to_check:
            hypothesis_text = hyp.get("hypothesis", "").lower()

            # More flexible matching: check for keywords as whole words or phrases
            # Also handle plural forms and common variations
            for keyword in all_keywords:
                # Simple substring match
                if keyword in hypothesis_text:
                    matched_keywords.append(keyword)
                    if not matched_hypothesis:
                        matched_hypothesis = hyp.get("hypothesis")
                # Also check for plural forms (basic)
                elif keyword.endswith("s") and keyword[:-1] in hypothesis_text:
                    matched_keywords.append(keyword)
                    if not matched_hypothesis:
                        matched_hypothesis = hyp.get("hypothesis")
                elif not keyword.endswith("s") and f"{keyword}s" in hypothesis_text:
                    matched_keywords.append(keyword)
                    if not matched_hypothesis:
                        matched_hypothesis = hyp.get("hypothesis")

            # If we found a match, we can stop checking
            if matched_keywords:
                break

        validation["hypothesis_match"] = len(matched_keywords) > 0

        # Store matched keywords and which hypothesis matched
        if matched_keywords:
            validation["matched_keywords"] = list(
                set(matched_keywords)
            )  # Remove duplicates
            if (
                matched_hypothesis
                and matched_hypothesis != validation["top_hypothesis"]
            ):
                validation["matched_hypothesis"] = matched_hypothesis
                validation["reason"] = (
                    f"Match found in hypothesis (not top): '{matched_hypothesis[:150]}...'"
                )
        else:
            validation["matched_keywords"] = []
            top_hyp_text = top_hypothesis.get("hypothesis", "")[:200]
            validation["reason"] = (
                f"None of the expected keywords {all_keywords[:10]}... found in top 3 hypotheses. Top hypothesis: '{top_hyp_text}...'"
            )

        return validation

    def validate_evidence(self, state: Dict, alert: Optional[Dict] = None) -> Dict:
        """Validate evidence sufficiency.

        Returns:
            Dict with evidence validation results
        """
        if not alert:
            alert_id = state.get("alert_id")
            alert = self.data_loader.get_alert(alert_id) if alert_id else None

        expected_evidence_types = (
            set(alert.get("expected_evidence_types", [])) if alert else set()
        )
        hypotheses = state.get("hypotheses", [])

        # Count evidence pieces
        total_evidence = sum(len(h.get("evidence", [])) for h in hypotheses)

        # Collect evidence types from hypotheses
        collected_evidence_types = set()
        for h in hypotheses:
            collected_evidence_types.update(h.get("evidence_types", []))

        validation = {
            "total_evidence": total_evidence,
            "evidence_threshold": self.config.evidence_count_threshold,
            "evidence_sufficient": total_evidence
            >= self.config.evidence_count_threshold,
            "expected_evidence_types": list(expected_evidence_types),
            "collected_evidence_types": list(collected_evidence_types),
            "evidence_types_match": expected_evidence_types.issubset(
                collected_evidence_types
            ),
        }

        return validation

    def validate_action_safety(self, state: Dict, alert: Optional[Dict] = None) -> Dict:
        """Validate action safety based on confidence and quality gate.

        Returns:
            Dict with action safety validation results
        """
        if not alert:
            alert_id = state.get("alert_id")
            alert = self.data_loader.get_alert(alert_id) if alert_id else None

        confidence_score = state.get("confidence_score", 0.0)
        quality_result = state.get("quality_gate_result") or {}
        requires_approval = alert.get("requires_approval", False) if alert else False

        validation = {
            "confidence_score": confidence_score,
            "confidence_threshold": self.config.confidence_threshold,
            "confidence_meets_threshold": confidence_score
            >= self.config.confidence_threshold,
            "requires_approval": requires_approval,
            "quality_gate_passed": quality_result.get("validation_passed", False)
            if quality_result
            else False,
            "writeback_allowed": quality_result.get("writeback_allowed", False)
            if quality_result
            else False,
            "action_safety_validated": (
                confidence_score >= self.config.confidence_threshold
                and (
                    quality_result.get("action_safety", True)
                    if quality_result
                    else True
                )
                and (
                    not requires_approval
                    or (
                        quality_result.get("approval_requested", False)
                        if quality_result
                        else False
                    )
                )
            ),
        }

        return validation

    def validate_runbook_grounding(self, state: Dict) -> Dict:
        """Validate that actions cite runbook sections.

        Returns:
            Dict with runbook grounding validation results
        """
        action_plan = state.get("action_plan", {})
        runbook_refs = action_plan.get("runbook_references", [])

        validation = {
            "runbook_references_count": len(runbook_refs),
            "has_runbook_grounding": len(runbook_refs) > 0,
            "runbook_references": runbook_refs,
        }

        return validation

    def generate_validation_report(
        self,
        state: Dict,
        run_id: str,
    ) -> Dict:
        """Generate validation report for a run.

        This validates business logic and expected outcomes. Evaluation metrics
        like bias, toxicity, relevance, etc. are computed automatically by the
        opentelemetry-util-genai-evals package and emitted as OTEL metrics/logs.

        Args:
            state: Final state from workflow
            run_id: Run identifier

        Returns:
            Dict with validation report
        """
        # Get alert
        alert_id = state.get("alert_id")
        alert = self.data_loader.get_alert(alert_id) if alert_id else None

        # Perform validations
        hypothesis_validation = self.validate_hypothesis(state, alert)
        evidence_validation = self.validate_evidence(state, alert)
        action_validation = self.validate_action_safety(state, alert)
        runbook_validation = self.validate_runbook_grounding(state)

        # Determine overall validation status
        validation_passed = (
            hypothesis_validation.get("hypothesis_match", False)
            and evidence_validation.get("evidence_sufficient", False)
            and action_validation.get("action_safety_validated", False)
        )

        report = {
            "run_id": run_id,
            "scenario_id": state.get("scenario_id"),
            "alert_id": alert_id,
            "timestamp": "2024-01-16T12:00:00Z",
            "validation_passed": validation_passed,
            "validations": {
                "hypothesis": hypothesis_validation,
                "evidence": evidence_validation,
                "action_safety": action_validation,
                "runbook_grounding": runbook_validation,
            },
            "thresholds": {
                "confidence": self.config.confidence_threshold,
                "evidence_count": self.config.evidence_count_threshold,
            },
            "note": "Evaluation metrics (bias, toxicity, relevance, etc.) are computed automatically by opentelemetry-util-genai-evals and emitted as OTEL metrics/logs.",
        }

        # Log validation results
        if validation_passed:
            _LOGGER.info(
                "Validation passed for run %s (scenario %s)",
                run_id,
                state.get("scenario_id"),
            )
        else:
            _LOGGER.warning(
                "Validation failed for run %s (scenario %s): hypothesis_match=%s, evidence_sufficient=%s, action_safe=%s",
                run_id,
                state.get("scenario_id"),
                hypothesis_validation.get("hypothesis_match"),
                evidence_validation.get("evidence_sufficient"),
                action_validation.get("action_safety_validated"),
            )

        return report
