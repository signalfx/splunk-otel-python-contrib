# Alpha Release - Test Execution Checklist

## Pre-Execution Setup

### Environment Preparation
- [ ] lab0 tenant access verified
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All required packages installed
- [ ] OTEL Collector running on lab0
- [ ] Splunk APM access confirmed

### Configuration Files
- [ ] `.env` file configured with lab0 credentials
- [ ] Azure OpenAI credentials valid
- [ ] Splunk access token configured
- [ ] Test data prepared

---

## Test Execution Tracking

### 1. Instrumentation Methods (5 tests)
- [ ] TC-1.1: Zero-Code vs Code-Based distinction
- [ ] TC-2.1: Prerequisites verification
- [ ] TC-2.2: Zero-Code LangChain instrumentation
- [ ] TC-2.3: Code-Based LangChain instrumentation
- [ ] TC-3.1: Direct AI app prerequisites

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 2. Agent and Workflow Configuration (3 tests)
- [ ] TC-2.4: agent_name configuration
- [ ] TC-2.4: workflow_name configuration
- [ ] TC-3.2: LLMInvocation for Azure OpenAI
- [ ] TC-3.3: AgentInvocation implementation

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 3. Evaluation Results (4 tests)
- [ ] TC-2.5: LangChain evaluation results
- [ ] TC-3.4: Direct AI evaluation results
- [ ] Verify bias scores
- [ ] Verify toxicity scores
- [ ] Verify hallucination scores
- [ ] Verify relevance scores
- [ ] Verify sentiment scores

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 4. Traceloop Integration (2 tests)
- [ ] TC-4.1: Traceloop prerequisites
- [ ] TC-4.2: Attribute translation verification
- [ ] Verify traceloop.* → gen_ai.* translation
- [ ] Verify DeepEval telemetry opt-out

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 5. Configuration Settings (10 tests)
- [ ] TC-5.1: DELTA temporality
- [ ] TC-5.1: CUMULATIVE temporality
- [ ] TC-5.1: LOWMEMORY temporality
- [ ] TC-5.2: Message content capture ON
- [ ] TC-5.2: Message content capture OFF
- [ ] TC-5.3: NO_CONTENT mode
- [ ] TC-5.3: SPAN_AND_EVENT mode
- [ ] TC-5.3: SPAN_ONLY mode
- [ ] TC-5.3: EVENT_ONLY mode
- [ ] TC-5.4: span emitter only
- [ ] TC-5.4: span_metric emitters
- [ ] TC-5.4: span_metric_event emitters
- [ ] TC-5.4: splunk emitter
- [ ] TC-5.5: 10% evaluation sampling
- [ ] TC-5.5: 50% evaluation sampling
- [ ] TC-5.5: 100% evaluation sampling
- [ ] TC-5.6: Debug logging enabled

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 6. Splunk APM UI - Agents Page (5 tests)
- [ ] TC-6.1: Agents page exists
- [ ] TC-6.1: Aggregate metrics display
- [ ] TC-6.1: Agent table displays
- [ ] TC-6.1: Individual agent metrics
- [ ] TC-6.2: Filter by environment
- [ ] TC-6.2: Filter by provider
- [ ] TC-6.2: Filter by model
- [ ] TC-6.2: Sort by requests
- [ ] TC-6.2: Sort by errors
- [ ] TC-6.2: Sort by latency
- [ ] TC-6.2: Search functionality

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 7. Splunk APM UI - Navigation (4 tests)
- [ ] TC-6.3: Related traces navigation
- [ ] TC-6.3: Trace Analyzer filters applied
- [ ] TC-6.3: AI traces only filter
- [ ] TC-6.4: Related logs navigation
- [ ] TC-6.4: Log Observer filters applied
- [ ] TC-6.4: Trace/span correlation
- [ ] TC-6.5: Agent detail view loads
- [ ] TC-6.5: Charts display correctly
- [ ] TC-6.5: Time range filters work

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 8. Splunk APM UI - Trace View (4 tests)
- [ ] TC-6.6: AI traces only filter
- [ ] TC-6.6: Agent attribute filtering
- [ ] TC-6.7: AI details tab visible
- [ ] TC-6.7: Metadata displayed
- [ ] TC-6.7: Quality scores shown
- [ ] TC-6.7: Agent input/output visible
- [ ] TC-6.7: Token usage displayed
- [ ] TC-6.8: Agent flow visualization
- [ ] TC-6.8: Steps displayed correctly
- [ ] TC-6.8: Tool calls visible
- [ ] TC-6.8: LLM calls highlighted

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 9. Log Observer (1 test)
- [ ] TC-6.9: AI call logs parsed
- [ ] TC-6.9: Trace/span information present
- [ ] TC-6.9: Navigation to traces works
- [ ] TC-6.9: Log fields extracted

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

### 10. Metrics and Dimensions (4 tests)
- [ ] TC-7.1: agent MMS exists
- [ ] TC-7.1: Accessible in Chart Builder
- [ ] TC-7.1: Accessible in SignalFlow
- [ ] TC-7.2: sf_environment dimension
- [ ] TC-7.2: gen_ai.agent.name dimension
- [ ] TC-7.2: sf_error dimension
- [ ] TC-7.2: gen_ai.provider.name dimension
- [ ] TC-7.2: gen_ai.request.model dimension
- [ ] TC-7.3: Custom dimensions addable
- [ ] TC-7.4: count() function works
- [ ] TC-7.4: min() function works
- [ ] TC-7.4: max() function works
- [ ] TC-7.4: median() function works
- [ ] TC-7.4: percentile() function works

**Status**: ⬜ Not Started | 🟡 In Progress | ✅ Complete  
**Blocker**: None  
**Notes**: _______________

---

## Test Summary

### Overall Progress
- **Total Test Cases**: 50+
- **Completed**: _____ / _____
- **Pass Rate**: _____%
- **Blockers**: _____
- **Critical Issues**: _____

### Test Categories Status
| Category | Total | Pass | Fail | Blocked | Pass % |
|----------|-------|------|------|---------|--------|
| Instrumentation | 5 | | | | |
| Agent/Workflow | 3 | | | | |
| Evaluations | 4 | | | | |
| Traceloop | 2 | | | | |
| Configuration | 10 | | | | |
| UI - Agents Page | 5 | | | | |
| UI - Navigation | 4 | | | | |
| UI - Trace View | 4 | | | | |
| Log Observer | 1 | | | | |
| Metrics/Dimensions | 4 | | | | |
| **TOTAL** | **42** | | | | |

---

## Issues and Blockers

### Critical Issues (P0)
1. Issue ID: _____ | Description: _____ | Status: _____
2. Issue ID: _____ | Description: _____ | Status: _____

### Major Issues (P1)
1. Issue ID: _____ | Description: _____ | Status: _____
2. Issue ID: _____ | Description: _____ | Status: _____

### Minor Issues (P2)
1. Issue ID: _____ | Description: _____ | Status: _____
2. Issue ID: _____ | Description: _____ | Status: _____

---

## Sign-Off

### Test Execution
- **Executed By**: _____________________
- **Date**: _____________________
- **Environment**: lab0
- **Build/Version**: _____________________

### Review
- **Reviewed By**: _____________________
- **Date**: _____________________
- **Approval**: ⬜ Approved | ⬜ Rejected | ⬜ Conditional

### Notes
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

---

## Next Steps

- [ ] Document all findings
- [ ] Create JIRA tickets for issues
- [ ] Update TestRail with results
- [ ] Schedule regression testing
- [ ] Prepare test report
- [ ] Present findings to team

---

**Checklist Version**: 1.0  
**Last Updated**: November 2025
