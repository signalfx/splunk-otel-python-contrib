# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Filtering Span Processor to drop unwanted spans before export.

This processor is specifically designed to filter out DeepEval internal spans
and other evaluation framework spans that should not be exported.
"""

import logging
from typing import List, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

_logger = logging.getLogger(__name__)


class FilteringSpanProcessor(SpanProcessor):
    """
    A span processor that filters out spans based on name patterns and instrumentation scope.
    
    Spans matching the filter criteria are dropped before reaching the exporter,
    preventing them from being sent to the backend.
    """
    
    # Default patterns for span names to exclude
    DEFAULT_EXCLUDE_PATTERNS = [
        # DeepEval evaluation spans
        "Run evaluate",
        "Ran evaluate",
        "Ran test case",
        "Bias",
        "Toxicity",
        "Relevance",
        "Hallucination",
        "Sentiment",
        "Answer Relevancy",
        "[GEval]",
        "[geval]",
        "deepeval",
        # LangGraph internal spans
        "__start__",
        "__end__",
        "should_continue",
        "model_to_tools",
        "tools_to_model",
    ]
    
    # Instrumentation scopes to exclude
    DEFAULT_EXCLUDE_SCOPES = [
        "deepeval.telemetry",
        "deepeval",
    ]
    
    def __init__(
        self,
        next_processor: SpanProcessor,
        exclude_span_name_patterns: Optional[List[str]] = None,
        exclude_instrumentation_scopes: Optional[List[str]] = None,
        log_filtered_spans: bool = False,
    ):
        """
        Initialize the filtering span processor.
        
        Args:
            next_processor: The next processor in the chain (typically the exporter)
            exclude_span_name_patterns: List of patterns to match against span names (case-insensitive)
            exclude_instrumentation_scopes: List of instrumentation scope names to exclude
            log_filtered_spans: Whether to log when spans are filtered out
        """
        self._next_processor = next_processor
        self._exclude_patterns = exclude_span_name_patterns or self.DEFAULT_EXCLUDE_PATTERNS
        self._exclude_scopes = exclude_instrumentation_scopes or self.DEFAULT_EXCLUDE_SCOPES
        self._log_filtered = log_filtered_spans
        self._filtered_count = 0
        
        _logger.info(
            "[FILTER] Initialized FilteringSpanProcessor with %d name patterns and %d scope patterns",
            len(self._exclude_patterns),
            len(self._exclude_scopes)
        )
    
    def _should_filter(self, span: ReadableSpan) -> bool:
        """
        Determine if a span should be filtered out.
        
        Returns:
            True if the span should be dropped, False otherwise
        """
        # Check instrumentation scope
        if hasattr(span, "instrumentation_scope") and span.instrumentation_scope:
            scope_name = span.instrumentation_scope.name
            for exclude_scope in self._exclude_scopes:
                if exclude_scope.lower() in scope_name.lower():
                    if self._log_filtered:
                        _logger.debug(
                            "[FILTER] Dropping span due to scope match: span='%s', scope='%s', pattern='%s'",
                            span.name,
                            scope_name,
                            exclude_scope
                        )
                    return True
        
        # Check span name
        if span.name:
            span_name_lower = span.name.lower()
            for pattern in self._exclude_patterns:
                if pattern.lower() in span_name_lower:
                    if self._log_filtered:
                        _logger.debug(
                            "[FILTER] Dropping span due to name match: span='%s', pattern='%s'",
                            span.name,
                            pattern
                        )
                    return True
        
        return False
    
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span is started."""
        # Pass through to next processor
        self._next_processor.on_start(span, parent_context)
    
    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends. Filter out unwanted spans before passing to next processor.
        """
        # Check if this span should be filtered
        if self._should_filter(span):
            self._filtered_count += 1
            # Don't call next processor - drop the span
            return
        
        # Pass through to next processor (exporter)
        self._next_processor.on_end(span)
    
    def shutdown(self) -> None:
        """Called when the tracer provider is shutdown."""
        _logger.info(
            "[FILTER] FilteringSpanProcessor shutdown. Total spans filtered: %d",
            self._filtered_count
        )
        self._next_processor.shutdown()
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return self._next_processor.force_flush(timeout_millis)

