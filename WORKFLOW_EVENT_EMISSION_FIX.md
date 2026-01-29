# Workflow Event Emission Fix

## Summary

Fixed missing workflow event emission in CrewAI instrumentation by adding `Workflow` support to the Splunk conversation events emitter and fixing a syntax error in the base content events emitter.

## Problem

When using `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,splunk"`, workflow events (`gen_ai.client.inference.operation.details`) were not being emitted for `Workflow` objects in CrewAI instrumentation, even though agent and LLM events were working correctly.

## Root Cause

1. **Configuration**: The `span_metric_event,splunk` configuration causes the Splunk emitter to **replace** the standard `ContentEventsEmitter` in the `content_events` category (due to `mode="replace-category"`).

2. **Missing Workflow Support**: The `SplunkConversationEventsEmitter` only handled `LLMInvocation` and `AgentInvocation` objects, but not `Workflow` objects.

3. **Syntax Error**: A minor whitespace syntax error in `content_events.py` (line 70) prevented the base emitter from being used even when not replaced.

## Solution

### 1. Added Workflow Support to Splunk Emitter

**File**: `util/opentelemetry-util-genai-emitters-splunk/src/opentelemetry/util/genai/emitters/splunk.py`

- **Added imports** (lines 24-36):
  ```python
  from opentelemetry.util.genai.emitters.utils import (
      _workflow_to_log_record,  # ← ADDED
  )
  from opentelemetry.util.genai.types import (
      Workflow,  # ← ADDED
  )
  ```

- **Added workflow handling in `SplunkConversationEventsEmitter.on_end()`** (lines 182-196):
  ```python
  if isinstance(obj, Workflow):
      try:
          genai_debug_log("emitter.splunk.conversation.on_end.workflow", obj)
      except Exception:
          pass
      try:
          rec = _workflow_to_log_record(obj, self._capture_content)
          if rec:
              self._event_logger.emit(rec)
      except Exception:
          pass
  ```

- **Updated `SplunkEvaluationResultsEmitter.handles()`** (line 259):
  ```python
  def handles(self, obj: Any) -> bool:
      return isinstance(obj, (Workflow, LLMInvocation, AgentInvocation))
  ```

### 2. Fixed Syntax Error in Content Events Emitter

**File**: `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/content_events.py`

- **Fixed line 70**:
  ```python
  # BEFORE:
  self.   _emit_workflow_event(obj)  # Extra whitespace causing syntax error
  
  # AFTER:
  self._emit_workflow_event(obj)
  ```

### 3. Enhanced Debug Logging

**File**: `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/content_events.py`

- **Added debug logging in `on_end()`** (lines 61-66):
  ```python
  if self._py_logger.isEnabledFor(logging.DEBUG):
      self._py_logger.debug(
          "ContentEventsEmitter.on_end called obj_type=%s capture_content=%s",
          type(obj).__name__,
          self._capture_content,
      )
  ```

- **Added debug logging in `_emit_workflow_event()`** (lines 120-134):
  ```python
  if self._py_logger.isEnabledFor(logging.DEBUG):
      self._py_logger.debug(
          "Emitting workflow content event trace_id=%s span_id=%s name=%s",
          getattr(workflow, "trace_id", None),
          getattr(workflow, "span_id", None),
          getattr(workflow, "name", None),
      )
  ```

## Impact

### Before Fix
- ❌ No workflow events emitted for CrewAI applications using `span_metric_event,splunk`
- ❌ Missing `gen_ai.client.inference.operation.details` events for `Workflow` objects
- ⚠️ Syntax error prevented fallback to `ContentEventsEmitter`

### After Fix
- ✅ Workflow events properly emitted with `gen_ai.client.inference.operation.details`
- ✅ Works with both `span_metric_event` and `span_metric_event,splunk` configurations
- ✅ Consistent event schema across Workflow, Agent, and LLM invocations
- ✅ Enhanced debug logging for troubleshooting

## Testing

### Manual Verification

```python
import os
os.environ['OTEL_INSTRUMENTATION_GENAI_EMITTERS'] = 'span_metric_event,splunk'
os.environ['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'] = 'true'
os.environ['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE'] = 'SPAN_AND_EVENT'
os.environ['OTEL_INSTRUMENTATION_GENAI_DEBUG'] = 'true'

from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import Workflow, InputMessage, OutputMessage, Text

handler = get_telemetry_handler()
workflow = Workflow(name='test_workflow', description='Test workflow')
workflow.input_messages.append(InputMessage(role='user', parts=[Text(content='test input')]))
handler.start_workflow(workflow)
workflow.output_messages.append(OutputMessage(role='assistant', parts=[Text(content='test output')], finish_reason='stop'))
handler.stop_workflow(workflow)

# Expected output:
# GENAIDEBUG event=emitter.splunk.conversation.on_end.workflow class=Workflow
```

### Integration Testing

Run CrewAI example with workflow events:
```bash
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples
./run_zero_code.sh
```

Expected: Workflow events with `gen_ai.client.inference.operation.details` event name in OTLP output.

## Architecture Notes

### Emitter Configuration Flow

When using `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,splunk"`:

1. **`span_metric_event`** baseline registers:
   - `SpanEmitter` (category: `span`)
   - `MetricsEmitter` (category: `metrics`)
   - `ContentEventsEmitter` (category: `content_events`)

2. **`,splunk`** plugin loads and:
   - Registers `SplunkConversationEventsEmitter` with `mode="replace-category"`
   - **Replaces** `ContentEventsEmitter` in the `content_events` category
   - Registers `SplunkEvaluationResultsEmitter` in `evaluation` category

3. **Result**: `SplunkConversationEventsEmitter` handles all content events, including workflows.

### Why Duplication is Acceptable

Both `ContentEventsEmitter` and `SplunkConversationEventsEmitter` have workflow handling code because:
- They operate independently based on configuration
- They share the core logic via `_workflow_to_log_record()` utility function
- This is the correct pattern for a pluggable emitter architecture
- Ensures workflows work regardless of configuration

## Related Files

### Modified Files
- `util/opentelemetry-util-genai/src/opentelemetry/util/genai/emitters/content_events.py`
- `util/opentelemetry-util-genai-emitters-splunk/src/opentelemetry/util/genai/emitters/splunk.py`

### Affected Packages
- `splunk-otel-util-genai` (0.1.8)
- `splunk-otel-genai-emitters-splunk` (0.1.5)

### Test Applications
- `instrumentation-genai/opentelemetry-instrumentation-crewai/examples/customer_support.py`
- `instrumentation-genai/opentelemetry-instrumentation-langchain/examples/*`
- `instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/*`

## Breaking Changes

None. This is a bug fix that adds missing functionality.

## Migration Guide

No migration required. The fix is backward compatible and automatically applies when packages are updated.

## Checklist

- [x] Code changes implemented
- [x] Manual testing completed
- [x] Debug logging added for troubleshooting
- [x] Documentation updated
- [ ] Unit tests added (if applicable)
- [ ] CHANGELOG.md updated
- [ ] Version bumped (if needed)

## References

- OpenTelemetry GenAI Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Event Schema: `gen_ai.client.inference.operation.details`
- Related Issue: Workflow events missing in CrewAI instrumentation

