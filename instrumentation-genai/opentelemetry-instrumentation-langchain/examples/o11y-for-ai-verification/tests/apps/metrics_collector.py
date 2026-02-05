"""
Metrics Collection, Performance Monitoring, and Cost Tracking for GenAI Applications

This module provides comprehensive observability for AI applications including:
- Evaluation metrics collection (pass/fail rates)
- Performance monitoring (latency, throughput)
- Cost tracking (token usage, API costs)
- Multi-environment support

Usage:
    from metrics_collector import MetricsCollector
    
    collector = MetricsCollector(
        service_name="my-ai-app",
        environment="production"
    )
    
    # Track evaluation results
    collector.record_evaluation(
        metric_name="bias",
        score=0.05,
        passed=True,
        trace_id="abc123"
    )
    
    # Track performance
    collector.record_llm_call(
        model="gpt-4",
        latency_ms=1500,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.015
    )
    
    # Get summary
    summary = collector.get_summary()
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class EvaluationMetric:
    """Single evaluation metric result"""
    metric_name: str
    score: float
    passed: bool
    trace_id: str
    timestamp: float
    scenario: Optional[str] = None
    model: Optional[str] = None


@dataclass
class LLMCallMetric:
    """Single LLM call performance and cost metrics"""
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    trace_id: str
    timestamp: float
    scenario: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MetricsSummary:
    """Aggregated metrics summary"""
    # Evaluation metrics
    total_evaluations: int
    evaluation_pass_rate: float
    evaluations_by_metric: Dict[str, Dict[str, Any]]
    
    # Performance metrics
    total_llm_calls: int
    total_llm_errors: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Cost metrics
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_cost_per_call_usd: float
    
    # By model
    metrics_by_model: Dict[str, Dict[str, Any]]
    
    # By scenario
    metrics_by_scenario: Dict[str, Dict[str, Any]]
    
    # Time range
    start_time: str
    end_time: str
    duration_seconds: float


# ============================================================================
# COST CALCULATOR
# ============================================================================

class CostCalculator:
    """Calculate costs for different LLM providers and models"""
    
    # Pricing per 1M tokens (as of Jan 2024)
    PRICING = {
        "gpt-4": {
            "input": 30.00,   # $30 per 1M input tokens
            "output": 60.00,  # $60 per 1M output tokens
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00,
        },
        "gpt-4o": {
            "input": 5.00,
            "output": 15.00,
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60,
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50,
        },
        # Azure OpenAI uses same pricing as OpenAI
        "azure/gpt-4": {
            "input": 30.00,
            "output": 60.00,
        },
        "azure/gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00,
        },
    }
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost in USD for a model call"""
        # Normalize model name
        model_key = model.lower()
        if "gpt-4o-mini" in model_key:
            model_key = "gpt-4o-mini"
        elif "gpt-4o" in model_key:
            model_key = "gpt-4o"
        elif "gpt-4-turbo" in model_key or "gpt-4-1106" in model_key:
            model_key = "gpt-4-turbo"
        elif "gpt-4" in model_key:
            model_key = "gpt-4"
        elif "gpt-3.5" in model_key:
            model_key = "gpt-3.5-turbo"
        
        pricing = cls.PRICING.get(model_key, cls.PRICING["gpt-4o-mini"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """
    Comprehensive metrics collection for GenAI applications
    
    Features:
    - Evaluation metrics tracking (pass/fail rates)
    - Performance monitoring (latency, throughput)
    - Cost tracking (token usage, API costs)
    - Multi-environment support
    - Export to JSON, CSV, or metrics backend
    """
    
    def __init__(
        self,
        service_name: str,
        environment: str = "dev",
        output_dir: Optional[Path] = None,
        auto_export: bool = True,
        export_interval_seconds: int = 300,
    ):
        """
        Initialize metrics collector
        
        Args:
            service_name: Name of the service
            environment: Environment (dev, staging, production)
            output_dir: Directory to save metrics files
            auto_export: Automatically export metrics periodically
            export_interval_seconds: How often to export (default 5 minutes)
        """
        self.service_name = service_name
        self.environment = environment
        self.output_dir = output_dir or Path("./metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.evaluations: List[EvaluationMetric] = []
        self.llm_calls: List[LLMCallMetric] = []
        
        # Timestamps
        self.start_time = time.time()
        self.last_export_time = self.start_time
        
        # Auto-export settings
        self.auto_export = auto_export
        self.export_interval_seconds = export_interval_seconds
        
        logger.info(
            f"MetricsCollector initialized: service={service_name}, "
            f"env={environment}, output_dir={self.output_dir}"
        )
    
    # ========================================================================
    # RECORD METRICS
    # ========================================================================
    
    def record_evaluation(
        self,
        metric_name: str,
        score: float,
        passed: bool,
        trace_id: str,
        scenario: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Record an evaluation metric result"""
        metric = EvaluationMetric(
            metric_name=metric_name,
            score=score,
            passed=passed,
            trace_id=trace_id,
            timestamp=time.time(),
            scenario=scenario,
            model=model,
        )
        self.evaluations.append(metric)
        
        logger.debug(
            f"Recorded evaluation: {metric_name}={score} "
            f"({'PASS' if passed else 'FAIL'})"
        )
        
        self._maybe_auto_export()
    
    def record_llm_call(
        self,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        trace_id: str,
        scenario: Optional[str] = None,
        error: Optional[str] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Record an LLM call with performance and cost metrics"""
        # Calculate cost if not provided
        if cost_usd is None:
            cost_usd = CostCalculator.calculate_cost(
                model, input_tokens, output_tokens
            )
        
        metric = LLMCallMetric(
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            trace_id=trace_id,
            timestamp=time.time(),
            scenario=scenario,
            error=error,
        )
        self.llm_calls.append(metric)
        
        logger.debug(
            f"Recorded LLM call: model={model}, latency={latency_ms}ms, "
            f"tokens={input_tokens}+{output_tokens}, cost=${cost_usd:.4f}"
        )
        
        self._maybe_auto_export()
    
    # ========================================================================
    # AGGREGATE METRICS
    # ========================================================================
    
    def get_summary(self) -> MetricsSummary:
        """Get aggregated metrics summary"""
        now = time.time()
        
        # Evaluation metrics
        total_evals = len(self.evaluations)
        passed_evals = sum(1 for e in self.evaluations if e.passed)
        pass_rate = (passed_evals / total_evals * 100) if total_evals > 0 else 0.0
        
        # Evaluations by metric
        evals_by_metric = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "scores": []})
        for eval_metric in self.evaluations:
            metric_stats = evals_by_metric[eval_metric.metric_name]
            metric_stats["total"] += 1
            if eval_metric.passed:
                metric_stats["passed"] += 1
            else:
                metric_stats["failed"] += 1
            metric_stats["scores"].append(eval_metric.score)
        
        # Calculate pass rates and avg scores
        for metric_name, stats in evals_by_metric.items():
            stats["pass_rate"] = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            del stats["scores"]  # Remove raw scores from summary
        
        # Performance metrics
        total_calls = len(self.llm_calls)
        total_errors = sum(1 for c in self.llm_calls if c.error)
        
        latencies = [c.latency_ms for c in self.llm_calls if not c.error]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_latencies = sorted(latencies)
        p50_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0.0
        
        # Cost metrics
        total_input_tokens = sum(c.input_tokens for c in self.llm_calls)
        total_output_tokens = sum(c.output_tokens for c in self.llm_calls)
        total_cost = sum(c.cost_usd for c in self.llm_calls)
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0.0
        
        # By model
        metrics_by_model = self._aggregate_by_field("model")
        
        # By scenario
        metrics_by_scenario = self._aggregate_by_field("scenario")
        
        return MetricsSummary(
            total_evaluations=total_evals,
            evaluation_pass_rate=pass_rate,
            evaluations_by_metric=dict(evals_by_metric),
            total_llm_calls=total_calls,
            total_llm_errors=total_errors,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost,
            avg_cost_per_call_usd=avg_cost_per_call,
            metrics_by_model=metrics_by_model,
            metrics_by_scenario=metrics_by_scenario,
            start_time=datetime.fromtimestamp(self.start_time).isoformat(),
            end_time=datetime.fromtimestamp(now).isoformat(),
            duration_seconds=now - self.start_time,
        )
    
    def _aggregate_by_field(self, field: str) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by a specific field (model or scenario)"""
        aggregated = defaultdict(lambda: {
            "calls": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        })
        
        for call in self.llm_calls:
            key = getattr(call, field, "unknown")
            if key is None:
                key = "unknown"
            
            stats = aggregated[key]
            stats["calls"] += 1
            if call.error:
                stats["errors"] += 1
            else:
                stats["total_latency_ms"] += call.latency_ms
            stats["total_input_tokens"] += call.input_tokens
            stats["total_output_tokens"] += call.output_tokens
            stats["total_cost_usd"] += call.cost_usd
        
        # Calculate averages
        for key, stats in aggregated.items():
            successful_calls = stats["calls"] - stats["errors"]
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] / successful_calls
                if successful_calls > 0 else 0.0
            )
            stats["avg_cost_per_call_usd"] = (
                stats["total_cost_usd"] / stats["calls"]
                if stats["calls"] > 0 else 0.0
            )
            del stats["total_latency_ms"]  # Remove intermediate value
        
        return dict(aggregated)
    
    # ========================================================================
    # EXPORT METRICS
    # ========================================================================
    
    def export_to_json(self, filename: Optional[str] = None) -> Path:
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{self.service_name}_{self.environment}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        summary = self.get_summary()
        data = {
            "service_name": self.service_name,
            "environment": self.environment,
            "summary": asdict(summary),
            "raw_evaluations": [asdict(e) for e in self.evaluations],
            "raw_llm_calls": [asdict(c) for c in self.llm_calls],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
        return filepath
    
    def print_summary(self) -> None:
        """Print metrics summary to console"""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print(f"📊 METRICS SUMMARY: {self.service_name} ({self.environment})")
        print("=" * 80)
        
        print(f"\n⏱️  Duration: {summary.duration_seconds:.1f}s")
        print(f"   Start: {summary.start_time}")
        print(f"   End:   {summary.end_time}")
        
        print(f"\n✅ Evaluation Metrics:")
        print(f"   Total Evaluations: {summary.total_evaluations}")
        print(f"   Pass Rate: {summary.evaluation_pass_rate:.1f}%")
        print(f"\n   By Metric:")
        for metric_name, stats in summary.evaluations_by_metric.items():
            print(f"     {metric_name}:")
            print(f"       Pass Rate: {stats['pass_rate']:.1f}% ({stats['passed']}/{stats['total']})")
            print(f"       Avg Score: {stats['avg_score']:.3f}")
        
        print(f"\n🚀 Performance Metrics:")
        print(f"   Total LLM Calls: {summary.total_llm_calls}")
        print(f"   Errors: {summary.total_llm_errors}")
        print(f"   Avg Latency: {summary.avg_latency_ms:.1f}ms")
        print(f"   P50 Latency: {summary.p50_latency_ms:.1f}ms")
        print(f"   P95 Latency: {summary.p95_latency_ms:.1f}ms")
        print(f"   P99 Latency: {summary.p99_latency_ms:.1f}ms")
        
        print(f"\n💰 Cost Metrics:")
        print(f"   Total Input Tokens: {summary.total_input_tokens:,}")
        print(f"   Total Output Tokens: {summary.total_output_tokens:,}")
        print(f"   Total Cost: ${summary.total_cost_usd:.4f}")
        print(f"   Avg Cost per Call: ${summary.avg_cost_per_call_usd:.4f}")
        
        if summary.metrics_by_model:
            print(f"\n📈 By Model:")
            for model, stats in summary.metrics_by_model.items():
                print(f"   {model}:")
                print(f"     Calls: {stats['calls']} (Errors: {stats['errors']})")
                print(f"     Avg Latency: {stats['avg_latency_ms']:.1f}ms")
                print(f"     Total Cost: ${stats['total_cost_usd']:.4f}")
        
        if summary.metrics_by_scenario:
            print(f"\n🎯 By Scenario:")
            for scenario, stats in summary.metrics_by_scenario.items():
                print(f"   {scenario}:")
                print(f"     Calls: {stats['calls']}")
                print(f"     Total Cost: ${stats['total_cost_usd']:.4f}")
        
        print("=" * 80 + "\n")
    
    def _maybe_auto_export(self) -> None:
        """Auto-export metrics if interval has passed"""
        if not self.auto_export:
            return
        
        now = time.time()
        if now - self.last_export_time >= self.export_interval_seconds:
            self.export_to_json()
            self.last_export_time = now
    
    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - export final metrics"""
        self.print_summary()
        self.export_to_json()
        return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_collector(
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> MetricsCollector:
    """
    Create a metrics collector with environment-based configuration
    
    Args:
        service_name: Override service name (default: from OTEL_SERVICE_NAME)
        environment: Override environment (default: from ENV or OTEL_RESOURCE_ATTRIBUTES)
    
    Returns:
        Configured MetricsCollector instance
    """
    if service_name is None:
        service_name = os.getenv("OTEL_SERVICE_NAME", "genai-app")
    
    if environment is None:
        # Try to extract from OTEL_RESOURCE_ATTRIBUTES
        resource_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if "deployment.environment=" in resource_attrs:
            environment = resource_attrs.split("deployment.environment=")[1].split(",")[0]
        else:
            environment = os.getenv("ENV", "dev")
    
    return MetricsCollector(
        service_name=service_name,
        environment=environment,
    )


__all__ = [
    "MetricsCollector",
    "MetricsSummary",
    "EvaluationMetric",
    "LLMCallMetric",
    "CostCalculator",
    "create_collector",
]
