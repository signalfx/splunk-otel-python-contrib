# TC-PI2-TRACELOOP-01: Test Report

## Summary

| Field | Value |
|-------|-------|
| **Test Case** | TC-PI2-TRACELOOP-01 |
| **Objective** | Validate Traceloop instrumentation data processed correctly by backend, spans visible in UI, and all platform features work (instrumentation-side evals, AI Defense) |
| **Status** | **✅ PASS** |
| **Date** | February 2, 2026 |
| **Trace ID** | `fac59c5efd3df301df9171981b40839e` |

---

## Test Steps Executed

| Step | Action | Result |
|------|--------|--------|
| 1 | Run `traceloop_travel_planner_app.py` with Traceloop SDK | ✅ App executed successfully |
| 2 | Verify traces exported to OTel Collector | ✅ Traces flushed to `localhost:4318` |
| 3 | Search trace in Splunk APM by trace ID | ✅ Trace found in RC0 realm |
| 4 | Verify spans visible in trace view | ✅ Both `ChatOpenAI.chat` and `chat gpt-4o-mini` spans visible |
| 5 | Open AI Details tab on `chat gpt-4o-mini` span | ✅ AI Details loaded without errors |
| 6 | Verify evaluation metrics displayed | ✅ All 5 metrics visible (Bias, Toxicity, Hallucination, Relevance, Sentiment) |
| 7 | Verify messages displayed | ✅ Input/output messages visible in Parsed and JSON views |

---

## Verification Checklist

| Criteria | Verified | Evidence Location |
|----------|----------|-------------------|
| Traceloop SDK spans created | ✅ | Splunk APM → Trace `fac59c5efd3df301df9171981b40839e` |
| `traceloop.*` → `gen_ai.*` translation | ✅ | `chat gpt-4o-mini` spans have `gen_ai.*` attributes |
| AI Details tab works | ✅ | Click any `chat gpt-4o-mini` span → AI Details tab |
| Evaluation metrics visible | ✅ | AI Details → Metrics section shows 5 evaluations |
| Messages visible | ✅ | AI Details → Messages section (Parsed/JSON) |
| All 5 agents in trace | ✅ | Trace hierarchy shows Coordinator → Flight → Hotel → Activity → Synthesizer |

---

## Evidence

**Splunk APM Query:**
```
sf_service:alpha-test-unified-app AND trace_id:fac59c5efd3df301df9171981b40839e
```

**Evaluation Results (from AI Details tab):**
- Bias: Not Biased (0.00)
- Toxicity: Not Toxic (0.00)
- Hallucination: Not Hallucinated
- Relevance: Pass (1.00)
- Sentiment: Positive

---

## Issues Resolved During Testing

| Issue | Fix |
|-------|-----|
| "Error loading events" in AI Details | Added `EventLoggerProvider` configuration |
| Evaluation metrics not appearing | Added `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk` |

---

## Artifacts

| File | Path |
|------|------|
| Application | `tests/apps/traceloop_travel_planner_app.py` |
| README | `tests/apps/TRACELOOP_TRAVEL_PLANNER_README.md` |

---

**Verdict: ✅ PASS** - All validation criteria met.
