from typing import Dict
from core.logger import get_logger


logger = get_logger(__name__)


class SpanValidator:
    """Validator for individual span attributes and structure."""

    @staticmethod
    def validate_span_attributes(span: Dict, required_attrs: list) -> bool:
        """
        Validate span has required attributes.

        Args:
            span: Span dictionary
            required_attrs: List of required attribute names

        Returns:
            True if valid, raises AssertionError otherwise
        """
        attributes = span.get("attributes", {})

        for attr in required_attrs:
            assert attr in attributes, f"Span missing required attribute: {attr}"

        logger.info(
            "Span attributes valid",
            span_id=span.get("span_id"),
            required_count=len(required_attrs),
        )
        return True

    @staticmethod
    def validate_span_status(span: Dict, expected_status: str = "ok") -> bool:
        """
        Validate span status.

        Args:
            span: Span dictionary
            expected_status: Expected status ("ok", "error", "unset")

        Returns:
            True if valid, raises AssertionError otherwise
        """
        status = span.get("status", {}).get("code", "unset")

        assert (
            status == expected_status
        ), f"Expected status '{expected_status}', got '{status}'"

        logger.info("Span status valid", span_id=span.get("span_id"), status=status)
        return True

    @staticmethod
    def validate_span_timing(span: Dict) -> bool:
        """
        Validate span has valid timing information.

        Args:
            span: Span dictionary

        Returns:
            True if valid, raises AssertionError otherwise
        """
        start_time = span.get("start_time")
        end_time = span.get("end_time")

        assert start_time is not None, "Span missing start_time"
        assert end_time is not None, "Span missing end_time"
        assert (
            end_time >= start_time
        ), f"end_time ({end_time}) must be >= start_time ({start_time})"

        duration_ms = (end_time - start_time) / 1000000  # Convert to ms

        logger.info(
            "Span timing valid", span_id=span.get("span_id"), duration_ms=duration_ms
        )
        return True

    @staticmethod
    def validate_attribute_type(
        span: Dict, attr_name: str, expected_type: type
    ) -> bool:
        """
        Validate attribute has expected type.

        Args:
            span: Span dictionary
            attr_name: Attribute name
            expected_type: Expected Python type

        Returns:
            True if valid, raises AssertionError otherwise
        """
        attributes = span.get("attributes", {})
        value = attributes.get(attr_name)

        assert value is not None, f"Attribute '{attr_name}' not found"
        assert isinstance(
            value, expected_type
        ), f"Attribute '{attr_name}' must be {expected_type.__name__}, got {type(value).__name__}"

        logger.debug(f"Attribute '{attr_name}' type valid: {expected_type.__name__}")
        return True
