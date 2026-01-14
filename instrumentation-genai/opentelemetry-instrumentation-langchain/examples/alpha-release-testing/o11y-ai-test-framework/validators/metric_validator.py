from typing import Dict, List, Any
from core.logger import get_logger


logger = get_logger(__name__)


class MetricValidator:
    """Validator for GenAI metrics with support for new naming convention."""
    
    # Old format: separate metrics
    OLD_FORMAT_METRICS = [
        "gen_ai.evaluation.bias",
        "gen_ai.evaluation.toxicity",
        "gen_ai.evaluation.hallucination",
        "gen_ai.evaluation.relevance",
        "gen_ai.evaluation.sentiment"
    ]
    
    # New format: single metric with name attribute
    NEW_FORMAT_METRIC = "gen_ai.evaluation.score"
    NEW_FORMAT_NAMES = ["bias", "toxicity", "hallucination", "relevance", "sentiment"]
    
    TOKEN_USAGE_METRICS = [
        "gen_ai.client.token.usage",
        "gen_ai.server.token.usage"
    ]
    
    COST_METRICS = [
        "gen_ai.client.operation.cost",
        "gen_ai.server.operation.cost"
    ]
    
    @staticmethod
    def validate_evaluation_metrics(
        metrics: List[Dict], 
        format_type: str = "auto"
    ) -> bool:
        """
        Validate evaluation metrics (supports both old and new formats).
        
        Args:
            metrics: List of metric dictionaries
            format_type: "old", "new", or "auto" (detect automatically)
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        if format_type == "auto":
            # Auto-detect format
            has_new_format = any(
                m.get("name") == MetricValidator.NEW_FORMAT_METRIC 
                for m in metrics
            )
            has_old_format = any(
                m.get("name") in MetricValidator.OLD_FORMAT_METRICS 
                for m in metrics
            )
            
            if has_new_format:
                format_type = "new"
            elif has_old_format:
                format_type = "old"
            else:
                raise AssertionError("No evaluation metrics found in either format")
        
        if format_type == "new":
            return MetricValidator._validate_new_format(metrics)
        else:
            return MetricValidator._validate_old_format(metrics)
    
    @staticmethod
    def _validate_new_format(metrics: List[Dict]) -> bool:
        """
        Validate new format: gen_ai.evaluation.score with gen_ai.evaluation.name attribute.
        
        Args:
            metrics: List of metric dictionaries
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        eval_metrics = [
            m for m in metrics 
            if m.get("name") == MetricValidator.NEW_FORMAT_METRIC
        ]
        
        assert len(eval_metrics) > 0, \
            f"No metrics found with name '{MetricValidator.NEW_FORMAT_METRIC}'"
        
        # Extract evaluation names
        eval_names = []
        for metric in eval_metrics:
            attributes = metric.get("attributes", {})
            eval_name = attributes.get("gen_ai.evaluation.name")
            
            assert eval_name is not None, \
                "Metric missing 'gen_ai.evaluation.name' attribute"
            assert eval_name in MetricValidator.NEW_FORMAT_NAMES, \
                f"Invalid evaluation name: {eval_name}. Must be one of {MetricValidator.NEW_FORMAT_NAMES}"
            
            eval_names.append(eval_name)
            
            # Validate score value
            score = metric.get("value")
            assert score is not None, f"Metric missing value for {eval_name}"
            assert isinstance(score, (int, float)), \
                f"Score must be numeric, got {type(score)}"
            assert 0.0 <= score <= 1.0, \
                f"Score must be between 0 and 1, got {score}"
        
        logger.info(
            "New format evaluation metrics valid",
            eval_names=eval_names,
            count=len(eval_metrics)
        )
        return True
    
    @staticmethod
    def _validate_old_format(metrics: List[Dict]) -> bool:
        """
        Validate old format: separate metrics (gen_ai.evaluation.bias, etc.).
        
        Args:
            metrics: List of metric dictionaries
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        found_metrics = []
        for metric in metrics:
            metric_name = metric.get("name")
            if metric_name in MetricValidator.OLD_FORMAT_METRICS:
                found_metrics.append(metric_name)
                
                # Validate score value
                score = metric.get("value")
                assert score is not None, f"Metric missing value for {metric_name}"
                assert isinstance(score, (int, float)), \
                    f"Score must be numeric, got {type(score)}"
                assert 0.0 <= score <= 1.0, \
                    f"Score must be between 0 and 1, got {score}"
        
        assert len(found_metrics) > 0, \
            f"No evaluation metrics found. Expected one of {MetricValidator.OLD_FORMAT_METRICS}"
        
        logger.info(
            "Old format evaluation metrics valid",
            found_metrics=found_metrics,
            count=len(found_metrics)
        )
        return True
    
    @staticmethod
    def validate_token_usage_metrics(metrics: List[Dict]) -> bool:
        """
        Validate token usage metrics.
        
        Args:
            metrics: List of metric dictionaries
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        token_metrics = [
            m for m in metrics 
            if m.get("name") in MetricValidator.TOKEN_USAGE_METRICS
        ]
        
        assert len(token_metrics) > 0, "No token usage metrics found"
        
        for metric in token_metrics:
            value = metric.get("value")
            assert value is not None, f"Token metric missing value"
            assert isinstance(value, (int, float)), \
                f"Token count must be numeric, got {type(value)}"
            assert value >= 0, f"Token count must be non-negative, got {value}"
        
        logger.info(
            "Token usage metrics valid",
            count=len(token_metrics)
        )
        return True
    
    @staticmethod
    def validate_cost_metrics(metrics: List[Dict]) -> bool:
        """
        Validate cost metrics.
        
        Args:
            metrics: List of metric dictionaries
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        cost_metrics = [
            m for m in metrics 
            if m.get("name") in MetricValidator.COST_METRICS
        ]
        
        assert len(cost_metrics) > 0, "No cost metrics found"
        
        for metric in cost_metrics:
            value = metric.get("value")
            assert value is not None, f"Cost metric missing value"
            assert isinstance(value, (int, float)), \
                f"Cost must be numeric, got {type(value)}"
            assert value >= 0, f"Cost must be non-negative, got {value}"
        
        logger.info(
            "Cost metrics valid",
            count=len(cost_metrics)
        )
        return True
    
    @staticmethod
    def find_metric_by_name(metrics: List[Dict], name: str) -> List[Dict]:
        """
        Find all metrics with given name.
        
        Args:
            metrics: List of metric dictionaries
            name: Metric name
        
        Returns:
            List of matching metrics
        """
        matching = [m for m in metrics if m.get("name") == name]
        logger.debug(f"Found {len(matching)} metrics with name: {name}")
        return matching
    
    @staticmethod
    def validate_metric_structure(metric: Dict) -> bool:
        """
        Validate basic metric structure.
        
        Args:
            metric: Metric dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        assert "name" in metric, "Metric missing 'name' field"
        assert "value" in metric, "Metric missing 'value' field"
        assert "timestamp" in metric or "time" in metric, \
            "Metric missing timestamp field"
        
        logger.debug(f"Metric structure valid: {metric.get('name')}")
        return True
    
    @staticmethod
    def compare_evaluation_scores(
        score1: float, 
        score2: float, 
        tolerance: float = 0.1
    ) -> bool:
        """
        Compare two evaluation scores with tolerance (for multi-judge comparison).
        
        Args:
            score1: First score
            score2: Second score
            tolerance: Acceptable difference (default 10%)
        
        Returns:
            True if scores are within tolerance
        """
        diff = abs(score1 - score2)
        within_tolerance = diff <= tolerance
        
        logger.info(
            "Score comparison",
            score1=score1,
            score2=score2,
            diff=diff,
            tolerance=tolerance,
            within_tolerance=within_tolerance
        )
        
        return within_tolerance
