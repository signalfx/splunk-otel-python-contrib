"""
TTFT (Time To First Token) tracking for LlamaIndex using the instrumentation event system.

This module provides:
- TTFTTracker: Records start times and calculates TTFT when first streaming token arrives
- LlamaindexEventHandler: Listens to LLM streaming events and populates TTFTTracker
- ContextVar correlation: Bridges callback handler (event_id) with event handler (span_id)
"""

import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

from llama_index.core.instrumentation.events.llm import (
    LLMChatInProgressEvent,
    LLMChatStartEvent,
)

try:
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler
except ImportError:
    # For versions of llama_index that don't have BaseEventHandler
    BaseEventHandler = object  # type: ignore


# ContextVar to store the current LLM event_id from callback handler
# This allows EventHandler to correlate its span_id with callback's event_id
_current_llm_event_id: ContextVar[Optional[str]] = ContextVar(
    "_current_llm_event_id", default=None
)


def set_current_llm_event_id(event_id: Optional[str]) -> None:
    """Set the current LLM event_id from callback handler."""
    _current_llm_event_id.set(event_id)


def get_current_llm_event_id() -> Optional[str]:
    """Get the current LLM event_id from callback handler."""
    return _current_llm_event_id.get()


class TTFTTracker:
    """Track Time To First Token for streaming LLM responses.

    This class:
    - Records when an LLM call starts (by span_id from instrumentation events)
    - Records when the first streaming token arrives
    - Calculates TTFT = first_token_time - start_time
    - Maps callback event_id to instrumentation span_id for cross-correlation
    """

    def __init__(self) -> None:
        # span_id -> start_time (when LLM call started)
        self._start_times: Dict[str, float] = {}
        # span_id -> ttft (calculated time to first token)
        self._ttft_values: Dict[str, float] = {}
        # span_id -> bool (whether first token has been received)
        self._first_token_received: Dict[str, bool] = {}
        # event_id -> span_id (map callback event_id to instrumentation span_id)
        self._event_span_map: Dict[str, str] = {}

    def record_start(self, span_id: str) -> None:
        """Record the start time for an LLM call."""
        self._start_times[span_id] = time.perf_counter()
        self._first_token_received[span_id] = False

    def record_first_token(self, span_id: str) -> Optional[float]:
        """Record when the first token arrives and calculate TTFT.

        Returns TTFT in seconds if this is the first token, None otherwise.
        """
        if span_id not in self._start_times:
            return None

        if self._first_token_received.get(span_id, False):
            # Already received first token
            return None

        self._first_token_received[span_id] = True
        ttft = time.perf_counter() - self._start_times[span_id]
        self._ttft_values[span_id] = ttft
        return ttft

    def get_ttft(self, span_id: str) -> Optional[float]:
        """Get the TTFT for a span_id, if available."""
        return self._ttft_values.get(span_id)

    def is_streaming(self, span_id: str) -> bool:
        """Check if streaming has started for this span."""
        return self._first_token_received.get(span_id, False)

    def associate_event_span(self, event_id: str, span_id: str) -> None:
        """Associate a callback event_id with an instrumentation span_id."""
        self._event_span_map[event_id] = span_id

    def get_span_for_event(self, event_id: str) -> Optional[str]:
        """Get the span_id associated with an event_id."""
        return self._event_span_map.get(event_id)

    def get_ttft_by_event(self, event_id: str) -> Optional[float]:
        """Get TTFT using callback event_id."""
        span_id = self._event_span_map.get(event_id)
        if span_id:
            return self.get_ttft(span_id)
        return None

    def is_streaming_by_event(self, event_id: str) -> bool:
        """Check if streaming has started using callback event_id."""
        span_id = self._event_span_map.get(event_id)
        if span_id:
            return self.is_streaming(span_id)
        return False

    def cleanup(self, span_id: str) -> None:
        """Clean up tracking data for a completed span."""
        self._start_times.pop(span_id, None)
        self._ttft_values.pop(span_id, None)
        self._first_token_received.pop(span_id, None)
        # Also clean up event mapping
        event_ids_to_remove = [
            eid for eid, sid in self._event_span_map.items() if sid == span_id
        ]
        for event_id in event_ids_to_remove:
            self._event_span_map.pop(event_id, None)

    def cleanup_by_event(self, event_id: str) -> None:
        """Clean up tracking data using callback event_id."""
        span_id = self._event_span_map.pop(event_id, None)
        if span_id:
            self.cleanup(span_id)


class LlamaindexEventHandler(BaseEventHandler):
    """Event handler that captures LLM streaming events for TTFT calculation.

    This handler:
    1. Listens for LLMChatStartEvent to record start time
    2. Listens for LLMChatInProgressEvent (first token) to calculate TTFT
    3. Associates callback event_id with instrumentation span_id via ContextVar
    """

    def __init__(self, ttft_tracker: TTFTTracker) -> None:
        self._tracker = ttft_tracker

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for LlamaIndex dispatcher."""
        return "LlamaindexTTFTEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> None:
        """Handle LlamaIndex instrumentation events."""
        if isinstance(event, LLMChatStartEvent):
            self._handle_start(event)
        elif isinstance(event, LLMChatInProgressEvent):
            self._handle_progress(event)

    def _handle_start(self, event: LLMChatStartEvent) -> None:
        """Handle LLM chat start event - record start time."""
        span_id = str(event.span_id) if hasattr(event, "span_id") else None
        if not span_id:
            return

        # Record start time
        self._tracker.record_start(span_id)

        # Associate with callback event_id if available
        event_id = get_current_llm_event_id()
        if event_id:
            self._tracker.associate_event_span(event_id, span_id)

    def _handle_progress(self, event: LLMChatInProgressEvent) -> None:
        """Handle LLM chat in-progress event - record first token."""
        span_id = str(event.span_id) if hasattr(event, "span_id") else None
        if not span_id:
            return

        # Record first token (TTFTTracker handles deduplication)
        self._tracker.record_first_token(span_id)
