from typing import Dict, List, Optional, Any
from core.logger import get_logger


logger = get_logger(__name__)


class TraceHelpers:
    """Helper functions for trace manipulation and analysis."""

    @staticmethod
    def extract_span_by_id(trace: Dict, span_id: str) -> Optional[Dict]:
        """
        Extract span by ID from trace.

        Args:
            trace: Trace dictionary
            span_id: Span ID to find

        Returns:
            Span dictionary or None
        """
        for span in trace.get("spans", []):
            if span.get("span_id") == span_id:
                return span
        return None

    @staticmethod
    def get_root_span(trace: Dict) -> Optional[Dict]:
        """
        Get root span (span with no parent).

        Args:
            trace: Trace dictionary

        Returns:
            Root span or None
        """
        for span in trace.get("spans", []):
            if not span.get("parent_span_id"):
                return span
        return None

    @staticmethod
    def get_child_spans(trace: Dict, parent_span_id: str) -> List[Dict]:
        """
        Get all direct children of a span.

        Args:
            trace: Trace dictionary
            parent_span_id: Parent span ID

        Returns:
            List of child spans
        """
        children = []
        for span in trace.get("spans", []):
            if span.get("parent_span_id") == parent_span_id:
                children.append(span)
        return children

    @staticmethod
    def calculate_trace_duration(trace: Dict) -> float:
        """
        Calculate total trace duration in milliseconds.

        Args:
            trace: Trace dictionary

        Returns:
            Duration in milliseconds
        """
        spans = trace.get("spans", [])
        if not spans:
            return 0.0

        start_times = [s.get("start_time", float("inf")) for s in spans]
        end_times = [s.get("end_time", 0) for s in spans]

        earliest_start = min(start_times)
        latest_end = max(end_times)

        duration_ms = (latest_end - earliest_start) / 1000000  # Convert to ms
        return duration_ms

    @staticmethod
    def get_span_depth(trace: Dict, span_id: str) -> int:
        """
        Calculate depth of span in trace hierarchy.

        Args:
            trace: Trace dictionary
            span_id: Span ID

        Returns:
            Depth (0 for root span)
        """
        span = TraceHelpers.extract_span_by_id(trace, span_id)
        if not span:
            return -1

        depth = 0
        current_span = span

        while current_span.get("parent_span_id"):
            depth += 1
            parent_id = current_span.get("parent_span_id")
            current_span = TraceHelpers.extract_span_by_id(trace, parent_id)
            if not current_span:
                break

        return depth

    @staticmethod
    def build_span_tree(trace: Dict) -> Dict:
        """
        Build hierarchical tree structure from flat span list.

        Args:
            trace: Trace dictionary

        Returns:
            Tree structure with nested children
        """
        spans = trace.get("spans", [])
        span_map = {s.get("span_id"): {**s, "children": []} for s in spans}

        root = None
        for span_id, span_data in span_map.items():
            parent_id = span_data.get("parent_span_id")
            if parent_id and parent_id in span_map:
                span_map[parent_id]["children"].append(span_data)
            elif not parent_id:
                root = span_data

        return root or {}

    @staticmethod
    def extract_token_usage(trace: Dict) -> Dict[str, int]:
        """
        Extract total token usage from all spans in trace.

        Args:
            trace: Trace dictionary

        Returns:
            Dictionary with input_tokens and output_tokens
        """
        total_input = 0
        total_output = 0

        for span in trace.get("spans", []):
            attributes = span.get("attributes", {})
            input_tokens = attributes.get("gen_ai.usage.input_tokens", 0)
            output_tokens = attributes.get("gen_ai.usage.output_tokens", 0)

            if isinstance(input_tokens, (int, float)):
                total_input += int(input_tokens)
            if isinstance(output_tokens, (int, float)):
                total_output += int(output_tokens)

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    @staticmethod
    def extract_cost(trace: Dict) -> float:
        """
        Extract total cost from all spans in trace.

        Args:
            trace: Trace dictionary

        Returns:
            Total cost
        """
        total_cost = 0.0

        for span in trace.get("spans", []):
            attributes = span.get("attributes", {})
            cost = attributes.get("gen_ai.usage.cost", 0)

            if isinstance(cost, (int, float)):
                total_cost += float(cost)

        return total_cost

    @staticmethod
    def get_span_summary(span: Dict) -> Dict[str, Any]:
        """
        Get summary information for a span.

        Args:
            span: Span dictionary

        Returns:
            Summary dictionary
        """
        attributes = span.get("attributes", {})

        return {
            "span_id": span.get("span_id"),
            "name": span.get("name"),
            "operation": attributes.get("gen_ai.operation.name"),
            "system": attributes.get("gen_ai.system"),
            "model": attributes.get("gen_ai.request.model"),
            "status": span.get("status", {}).get("code"),
            "duration_ms": (span.get("end_time", 0) - span.get("start_time", 0))
            / 1000000,
        }

    @staticmethod
    def filter_spans_by_system(trace: Dict, system: str) -> List[Dict]:
        """
        Filter spans by gen_ai.system attribute.

        Args:
            trace: Trace dictionary
            system: System name (e.g., 'openai', 'anthropic')

        Returns:
            List of matching spans
        """
        matching = []
        for span in trace.get("spans", []):
            if span.get("attributes", {}).get("gen_ai.system") == system:
                matching.append(span)

        logger.debug(f"Found {len(matching)} spans for system: {system}")
        return matching
