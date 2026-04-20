"""Test TTFT (Time To First Token) tracking for LlamaIndex instrumentation.

Tests the TTFTTracker, LlamaindexEventHandler, and the correlation between
callback event_id and instrumentation span_id via ContextVar.
"""

import time

from opentelemetry.instrumentation.llamaindex.event_handler import (
    TTFTTracker,
    LlamaindexEventHandler,
    set_current_llm_event_id,
    get_current_llm_event_id,
)
from opentelemetry.instrumentation.llamaindex.invocation_manager import (
    _InvocationManager,
)


# ==================== TTFTTracker Unit Tests ====================


class TestTTFTTracker:
    """Test TTFTTracker in isolation."""

    def test_record_start_and_first_token(self):
        tracker = TTFTTracker()
        tracker.record_start("span-1")
        time.sleep(0.01)  # small delay to get measurable TTFT
        ttft = tracker.record_first_token("span-1")

        assert ttft is not None
        assert ttft > 0
        assert ttft < 1.0  # should be much less than 1 second

    def test_second_token_returns_none(self):
        tracker = TTFTTracker()
        tracker.record_start("span-1")
        tracker.record_first_token("span-1")
        # Second call should return None
        result = tracker.record_first_token("span-1")
        assert result is None

    def test_get_ttft(self):
        tracker = TTFTTracker()
        tracker.record_start("span-1")
        tracker.record_first_token("span-1")

        ttft = tracker.get_ttft("span-1")
        assert ttft is not None
        assert ttft > 0

    def test_get_ttft_no_token(self):
        tracker = TTFTTracker()
        tracker.record_start("span-1")
        assert tracker.get_ttft("span-1") is None

    def test_get_ttft_unknown_span(self):
        tracker = TTFTTracker()
        assert tracker.get_ttft("nonexistent") is None

    def test_is_streaming(self):
        tracker = TTFTTracker()
        tracker.record_start("span-1")
        assert not tracker.is_streaming("span-1")

        tracker.record_first_token("span-1")
        assert tracker.is_streaming("span-1")

    def test_associate_event_span(self):
        tracker = TTFTTracker()
        tracker.associate_event_span("event-1", "span-1")
        tracker.record_start("span-1")
        tracker.record_first_token("span-1")

        ttft = tracker.get_ttft_by_event("event-1")
        assert ttft is not None
        assert ttft > 0

    def test_is_streaming_by_event(self):
        tracker = TTFTTracker()
        tracker.associate_event_span("event-1", "span-1")
        tracker.record_start("span-1")

        assert not tracker.is_streaming_by_event("event-1")
        tracker.record_first_token("span-1")
        assert tracker.is_streaming_by_event("event-1")

    def test_cleanup(self):
        tracker = TTFTTracker()
        tracker.associate_event_span("event-1", "span-1")
        tracker.record_start("span-1")
        tracker.record_first_token("span-1")

        # Verify data exists
        assert tracker.get_ttft("span-1") is not None
        assert tracker.get_ttft_by_event("event-1") is not None

        # Cleanup
        tracker.cleanup("span-1")

        assert tracker.get_ttft("span-1") is None
        assert tracker.get_ttft_by_event("event-1") is None
        assert not tracker.is_streaming("span-1")

    def test_cleanup_by_event(self):
        tracker = TTFTTracker()
        tracker.associate_event_span("event-1", "span-1")
        tracker.record_start("span-1")
        tracker.record_first_token("span-1")

        tracker.cleanup_by_event("event-1")

        assert tracker.get_ttft("span-1") is None
        assert tracker.get_ttft_by_event("event-1") is None

    def test_multiple_concurrent_spans(self):
        tracker = TTFTTracker()
        tracker.associate_event_span("event-1", "span-1")
        tracker.associate_event_span("event-2", "span-2")

        tracker.record_start("span-1")
        time.sleep(0.01)
        tracker.record_start("span-2")
        time.sleep(0.01)

        tracker.record_first_token("span-2")  # span-2 gets token first
        time.sleep(0.01)
        tracker.record_first_token("span-1")  # span-1 gets token later

        ttft1 = tracker.get_ttft_by_event("event-1")
        ttft2 = tracker.get_ttft_by_event("event-2")

        assert ttft1 is not None
        assert ttft2 is not None
        # span-1 started earlier but got token later, so its TTFT should be larger
        assert ttft1 > ttft2


# ==================== ContextVar Correlation Tests ====================


class TestContextVarCorrelation:
    """Test the ContextVar-based event_id <-> span_id correlation."""

    def test_set_and_get_event_id(self):
        set_current_llm_event_id("evt-123")
        assert get_current_llm_event_id() == "evt-123"

        set_current_llm_event_id(None)
        assert get_current_llm_event_id() is None

    def test_event_handler_associates_on_start(self):
        """When LLMChatStartEvent fires, EventHandler should associate
        the current event_id with the event's span_id."""
        tracker = TTFTTracker()
        handler = LlamaindexEventHandler(ttft_tracker=tracker)

        # Simulate: CallbackHandler sets event_id before LLM call
        set_current_llm_event_id("callback-event-42")

        # Simulate: LLMChatStartEvent fires
        from llama_index.core.instrumentation.events.llm import LLMChatStartEvent

        start_event = LLMChatStartEvent(
            messages=[],
            model_dict={},
            additional_kwargs={},
            span_id="llama-span-99",
        )
        handler.handle(start_event)

        # Verify association
        assert tracker._event_span_map["callback-event-42"] == "llama-span-99"

        # Verify start time recorded
        assert "llama-span-99" in tracker._start_times

        # Clean up
        set_current_llm_event_id(None)

    def test_end_to_end_ttft_flow(self):
        """Full flow: CallbackHandler sets event_id -> EventHandler records TTFT
        -> InvocationManager retrieves TTFT by event_id."""
        tracker = TTFTTracker()
        handler = LlamaindexEventHandler(ttft_tracker=tracker)
        inv_mgr = _InvocationManager()
        inv_mgr.set_ttft_tracker(tracker)

        # Step 1: CallbackHandler._handle_llm_start sets event_id
        set_current_llm_event_id("cb-event-1")

        # Step 2: LLMChatStartEvent fires (inside LlamaIndex LLM call)
        from llama_index.core.instrumentation.events.llm import (
            LLMChatStartEvent,
            LLMChatInProgressEvent,
        )
        from llama_index.core.llms import ChatResponse, ChatMessage

        start_event = LLMChatStartEvent(
            messages=[],
            model_dict={},
            additional_kwargs={},
            span_id="internal-span-1",
        )
        handler.handle(start_event)

        # Step 3: Simulate some processing time
        time.sleep(0.02)

        # Step 4: LLMChatInProgressEvent fires (first streaming chunk)
        progress_event = LLMChatInProgressEvent(
            messages=[],
            response=ChatResponse(message=ChatMessage(content="Hello")),
            span_id="internal-span-1",
        )
        handler.handle(progress_event)

        # Step 5: Second chunk - should NOT update TTFT
        time.sleep(0.01)
        handler.handle(progress_event)

        # Step 6: CallbackHandler._handle_llm_end retrieves TTFT
        ttft = inv_mgr.get_ttft_for_event("cb-event-1")
        assert ttft is not None
        assert ttft >= 0.02  # at least the sleep time
        assert ttft < 1.0

        # Also check streaming flag
        assert inv_mgr.is_streaming_event("cb-event-1")

        # Step 7: Cleanup
        inv_mgr.cleanup_event_tracking("cb-event-1")
        set_current_llm_event_id(None)

        assert inv_mgr.get_ttft_for_event("cb-event-1") is None

    def test_non_streaming_no_ttft(self):
        """Non-streaming calls should not have TTFT."""
        tracker = TTFTTracker()
        handler = LlamaindexEventHandler(ttft_tracker=tracker)
        inv_mgr = _InvocationManager()
        inv_mgr.set_ttft_tracker(tracker)

        set_current_llm_event_id("cb-event-2")

        # Only start event, no in-progress (non-streaming)
        from llama_index.core.instrumentation.events.llm import LLMChatStartEvent

        start_event = LLMChatStartEvent(
            messages=[],
            additional_kwargs={},
            model_dict={},
            span_id="internal-span-2",
        )
        handler.handle(start_event)

        # No TTFT for non-streaming
        assert inv_mgr.get_ttft_for_event("cb-event-2") is None
        assert not inv_mgr.is_streaming_event("cb-event-2")

        set_current_llm_event_id(None)

    def test_no_tracker_graceful(self):
        """InvocationManager without tracker should not crash."""
        inv_mgr = _InvocationManager()
        # No tracker set
        assert inv_mgr.get_ttft_for_event("any") is None
        assert not inv_mgr.is_streaming_event("any")
        inv_mgr.cleanup_event_tracking("any")  # should not crash
