from typing import Dict, List, Optional, Any
from core.logger import get_logger


logger = get_logger(__name__)


class TraceValidator:
    """Validator for trace structure and GenAI schema compliance."""
    
    REQUIRED_GENAI_ATTRIBUTES = [
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.operation.name"
    ]
    
    OPTIONAL_GENAI_ATTRIBUTES = [
        "gen_ai.request.temperature",
        "gen_ai.request.max_tokens",
        "gen_ai.response.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens"
    ]
    
    VALID_OPERATIONS = [
        "chat",
        "completion",
        "embedding",
        "invoke_agent",
        "invoke_workflow",
        "invoke_tool"
    ]
    
    @staticmethod
    def validate_trace_structure(trace: Dict) -> bool:
        """
        Validate basic trace structure.
        
        Args:
            trace: Trace dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        assert "trace_id" in trace, "Trace missing trace_id"
        assert "spans" in trace, "Trace missing spans"
        assert isinstance(trace["spans"], list), "Spans must be a list"
        assert len(trace["spans"]) > 0, "Trace has no spans"
        
        logger.info(
            "Trace structure valid",
            trace_id=trace["trace_id"],
            span_count=len(trace["spans"])
        )
        return True
    
    @staticmethod
    def validate_genai_schema(span: Dict, strict: bool = False) -> bool:
        """
        Validate span conforms to GenAI semantic conventions.
        
        Args:
            span: Span dictionary
            strict: If True, enforce all optional attributes
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        attributes = span.get("attributes", {})
        
        # Check required attributes
        for attr in TraceValidator.REQUIRED_GENAI_ATTRIBUTES:
            assert attr in attributes, f"Missing required attribute: {attr}"
        
        # Validate operation name
        operation = attributes.get("gen_ai.operation.name")
        if operation:
            assert operation in TraceValidator.VALID_OPERATIONS, \
                f"Invalid operation: {operation}. Must be one of {TraceValidator.VALID_OPERATIONS}"
        
        # Check optional attributes in strict mode
        if strict:
            for attr in TraceValidator.OPTIONAL_GENAI_ATTRIBUTES:
                assert attr in attributes, f"Missing optional attribute (strict mode): {attr}"
        
        logger.info(
            "GenAI schema valid",
            span_id=span.get("span_id"),
            operation=operation
        )
        return True
    
    @staticmethod
    def validate_parent_child(parent_span: Dict, child_span: Dict) -> bool:
        """
        Validate parent-child relationship between spans.
        
        Args:
            parent_span: Parent span dictionary
            child_span: Child span dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        parent_span_id = parent_span.get("span_id")
        child_parent_id = child_span.get("parent_span_id")
        
        assert parent_span_id is not None, "Parent span missing span_id"
        assert child_parent_id is not None, "Child span missing parent_span_id"
        assert parent_span_id == child_parent_id, \
            f"Parent-child mismatch: parent={parent_span_id}, child_parent={child_parent_id}"
        
        logger.info(
            "Parent-child relationship valid",
            parent_span_id=parent_span_id,
            child_span_id=child_span.get("span_id")
        )
        return True
    
    @staticmethod
    def find_span_by_operation(trace: Dict, operation: str) -> Optional[Dict]:
        """
        Find first span with given operation name.
        
        Args:
            trace: Trace dictionary
            operation: Operation name (e.g., 'chat', 'invoke_agent')
        
        Returns:
            Span dictionary or None if not found
        """
        for span in trace.get("spans", []):
            span_operation = span.get("attributes", {}).get("gen_ai.operation.name")
            if span_operation == operation:
                logger.debug(f"Found span with operation: {operation}")
                return span
        
        logger.warning(f"No span found with operation: {operation}")
        return None
    
    @staticmethod
    def find_spans_by_attribute(
        trace: Dict, 
        attribute_name: str, 
        attribute_value: Any
    ) -> List[Dict]:
        """
        Find all spans with specific attribute value.
        
        Args:
            trace: Trace dictionary
            attribute_name: Attribute name (e.g., 'gen_ai.system')
            attribute_value: Expected value
        
        Returns:
            List of matching spans
        """
        matching_spans = []
        for span in trace.get("spans", []):
            if span.get("attributes", {}).get(attribute_name) == attribute_value:
                matching_spans.append(span)
        
        logger.info(
            f"Found {len(matching_spans)} spans with {attribute_name}={attribute_value}"
        )
        return matching_spans
    
    @staticmethod
    def validate_span_hierarchy(trace: Dict) -> bool:
        """
        Validate all spans form a valid hierarchy.
        
        Args:
            trace: Trace dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        spans = trace.get("spans", [])
        span_ids = {span.get("span_id") for span in spans}
        
        # Find root span (no parent)
        root_spans = [s for s in spans if not s.get("parent_span_id")]
        assert len(root_spans) > 0, "No root span found"
        
        # Validate all parent references exist
        for span in spans:
            parent_id = span.get("parent_span_id")
            if parent_id:
                assert parent_id in span_ids, \
                    f"Span {span.get('span_id')} references non-existent parent {parent_id}"
        
        logger.info(
            "Span hierarchy valid",
            total_spans=len(spans),
            root_spans=len(root_spans)
        )
        return True
    
    @staticmethod
    def validate_token_usage(span: Dict) -> bool:
        """
        Validate token usage attributes are present and valid.
        
        Args:
            span: Span dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        attributes = span.get("attributes", {})
        
        input_tokens = attributes.get("gen_ai.usage.input_tokens")
        output_tokens = attributes.get("gen_ai.usage.output_tokens")
        
        if input_tokens is not None:
            assert isinstance(input_tokens, (int, float)), \
                f"input_tokens must be numeric, got {type(input_tokens)}"
            assert input_tokens >= 0, f"input_tokens must be non-negative, got {input_tokens}"
        
        if output_tokens is not None:
            assert isinstance(output_tokens, (int, float)), \
                f"output_tokens must be numeric, got {type(output_tokens)}"
            assert output_tokens >= 0, f"output_tokens must be non-negative, got {output_tokens}"
        
        logger.info(
            "Token usage valid",
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        return True
    
    @staticmethod
    def validate_streaming_attributes(span: Dict) -> bool:
        """
        Validate streaming-specific attributes (TTFT).
        
        Args:
            span: Span dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        attributes = span.get("attributes", {})
        
        # Check for streaming indicator
        is_streaming = attributes.get("gen_ai.response.streaming", False)
        
        if is_streaming:
            # Validate TTFT (Time To First Token) is present
            ttft = attributes.get("gen_ai.response.ttft")
            assert ttft is not None, "Streaming span missing gen_ai.response.ttft"
            assert isinstance(ttft, (int, float)), f"TTFT must be numeric, got {type(ttft)}"
            assert ttft > 0, f"TTFT must be positive, got {ttft}"
            
            logger.info(f"Streaming attributes valid, TTFT: {ttft}ms")
        
        return True
    
    @staticmethod
    def validate_ai_defense_attributes(span: Dict) -> bool:
        """
        Validate AI Defense security event attributes.
        
        Args:
            span: Span dictionary
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        attributes = span.get("attributes", {})
        
        security_event_id = attributes.get("gen_ai.security.event_id")
        if security_event_id:
            assert isinstance(security_event_id, str), \
                f"security_event_id must be string, got {type(security_event_id)}"
            assert len(security_event_id) > 0, "security_event_id cannot be empty"
            
            # Check for risk level
            risk_level = attributes.get("gen_ai.security.risk_level")
            if risk_level:
                valid_levels = ["low", "medium", "high", "critical"]
                assert risk_level in valid_levels, \
                    f"Invalid risk_level: {risk_level}. Must be one of {valid_levels}"
            
            logger.info(
                "AI Defense attributes valid",
                security_event_id=security_event_id,
                risk_level=risk_level
            )
        
        return True
