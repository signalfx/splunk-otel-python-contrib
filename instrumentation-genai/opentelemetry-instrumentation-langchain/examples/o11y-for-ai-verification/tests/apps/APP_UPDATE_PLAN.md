# Test App Update Plan

## ✅ Completed: Cleanup Phase

1. ✅ Deleted `direct_azure_openai_app_v2_backup.py`
2. ✅ Created `archive/` with `.gitignore`
3. ✅ Moved `direct_azure_openai_app.py` → `archive/multi_agent_hierarchy_test_app.py`
4. ✅ Created `archive/README.md`
5. ✅ Updated main `README.md` with v2 as primary GA app
6. ✅ Updated `EVALUATION_ISSUE_SUMMARY.md` with resolution

---

## 🎯 Primary GA Test App (Complete)

**File:** `direct_azure_openai_app_v2.py`  
**Status:** ✅ Production-ready  
**Features:**
- Single-trace architecture (3 scenarios)
- Timing fix applied (5s/120s)
- All eval metrics working (bias, toxicity, hallucination)
- Config-driven, retry logic, structured logging
- Metrics collection integration

---

## 📋 Remaining Apps to Update

### 1. **langgraph_travel_planner_app.py** (863 lines)

**Current State:**
- LangGraph multi-agent travel planner
- Shows eval metrics in telemetry example
- Uses LangChain instrumentation (auto-instrumented)

**Required Updates:**
1. **Add Circuit API Integration**
   - Client ID: `0oarbtuuh0w0QsPrJ5d7`
   - Client Secret: `7sXq_UC3xFBVP5UWdj53wIxQqLax402er-UPUM-0FJIUL7kvlMSICYHTj9X7uhPT`
   - App Key: `egai-prd-other-123028255-coding-1762238492681`
   - Replace OpenAI client with Circuit client

2. **Apply Timing Fix**
   - Find telemetry flush section
   - Change wait time from 60s → 120s
   - Add delays between major workflow steps if needed

3. **Verify Eval Metrics Configuration**
   - Ensure `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` is set
   - Ensure `OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS=evaluations`
   - Ensure `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=1.0`

4. **Apply Best Practices from v2**
   - Config-driven environment variables
   - Structured logging with trace correlation
   - Error handling with retries
   - Configuration validation

**Complexity:** High (Circuit API integration + LangGraph framework)

---

### 2. **traceloop_travel_planner_app.py** (size unknown)

**Current State:**
- Traceloop SDK integration
- Travel planner workflow
- Needs analysis

**Required Updates:**
1. **Apply Timing Fix**
   - Change wait time from 60s → 120s
   - Add delays between workflow steps

2. **Verify Eval Metrics Configuration**
   - Check if evaluations are enabled
   - Verify sample rate and callbacks

3. **Apply Best Practices from v2**
   - Config-driven approach
   - Structured logging
   - Error handling

**Complexity:** Medium (Traceloop SDK specific)

---

### 3. **unified_genai_test_app.py** (107KB / ~2700 lines)

**Current State:**
- Comprehensive multi-framework testing
- Very large file
- Needs analysis

**Required Updates:**
1. **Apply Timing Fix**
   - Find all telemetry flush sections
   - Change wait times to 120s
   - Add delays between test scenarios

2. **Verify Eval Metrics Configuration**
   - Check all framework tests have eval config
   - Verify sample rates

3. **Apply Best Practices from v2**
   - May need refactoring due to size
   - Config-driven approach
   - Structured logging

**Complexity:** Very High (large file, multiple frameworks)

---

## 🚀 Implementation Strategy

### Phase 1: Quick Wins (Priority)
1. Apply timing fix to all 3 apps (search for `time.sleep(60)` and change to `120`)
2. Verify eval configuration in all apps
3. Test each app to ensure eval metrics appear

### Phase 2: Circuit API Integration
1. Add Circuit API client to langgraph app
2. Replace OpenAI calls with Circuit calls
3. Test travel planner workflow with Circuit

### Phase 3: Best Practices Application
1. Review each app for config-driven approach
2. Add structured logging where missing
3. Add error handling and retries where needed

### Phase 4: Testing & Validation
1. Run each app and verify eval metrics in Splunk
2. Verify all 3 apps work with timing fix
3. Document any issues or limitations

---

## ⚠️ Considerations

1. **Time Estimate:** 4-6 hours for all updates
2. **Testing Required:** Each app needs to be run and verified in Splunk
3. **Circuit API:** May require additional dependencies or configuration
4. **Large Files:** `unified_genai_test_app.py` may need refactoring
5. **Framework-Specific:** Each app uses different instrumentation approach

---

## 📝 Next Steps

1. Start with Phase 1 (timing fix) - quick and high impact
2. Analyze each app's current eval configuration
3. Apply Circuit API to langgraph app
4. Test and verify all changes

---

**Created:** January 30, 2026  
**Status:** In Progress  
**Owner:** QA/Dev Team
