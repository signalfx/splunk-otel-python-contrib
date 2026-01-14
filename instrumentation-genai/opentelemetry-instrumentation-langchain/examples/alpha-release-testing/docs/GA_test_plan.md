
Test Plan: O11y for AI - GA Release (PI2) 
 

Jira Story: QSE-4796 (PI2 Scope) | Related Epics: QSE-3131, QSE-3667, HYBIM-413, O11YDOCS-7779, RDMP-3228
Version: 1.1 (Final - Post-Review) | Date: December 18, 2025 | Owner: Ankur Kumar Shandilya
Target GA: February 25, 2026 | Testing Deadline: January 20, 2026 | Status: ✅ Approved

Test Plan: O11y for AI - GA Release (PI2) 
1. Executive Summary
1.1 Purpose
1.2 Scope
1.3 Test Strategy (UPDATED)
APP 1: Foundation Demo (COMPREHENSIVE) - PRIMARY FOCUS 
APP 2: Traceloop Translator Demo - BACKEND VALIDATION
APP 3: Hybrid Setup Demo (OPTIONAL)
1.4 Key Metrics (FINAL)
1.5 Success Criteria
2. Prerequisites and Dependencies 
2.1 Critical Dependencies
3. Test Scope 
3.1 Functional Requirements Coverage
4. Test Scenarios (30 P0 CASES)
4.1 High-Level Test Scenarios
4.2 Complete Test Case List (30 P0 CASES )
5. UI Functional Testing 
5.1 UI Test Coverage
6. Regression Testing
6.1 Regression Test Suite 
7. Security Testing
7.1 Security Test Cases
8. Reliability Testing
8.1 Reliability Test Cases
9. Configuration Testing
9.1 Configuration Matrix
10. Internationalization
10.1 Non-ASCII Input Validation
10.2 Time Zone Handling
11. Browser Support
11.1 Supported Browsers
12. Tools
12.1 Test Automation Tools
12.2 CI/CD
12.3 Monitoring
13. Deployment
13.1 Deployment Strategy
13.2 Testing Per Realm
14. Test Execution Schedule 
14.1 Weekly Breakdown (App 1 Focus Strategy)
15. Risk Assessment 
15.1 Critical Risks
DOCUMENT STATUS
1. Executive Summary
1.1 Purpose
This test plan validates the O11y for AI - GA Release (PI2) for GA on February 25, 2026, with testing completion by January 20, 2026, using 3 demo setups focused on App 1 first strategy.

1.2 Scope
In Scope (GA Blocking - P0 Focus):

✅ Foundation Components Demo App (HYBIM-413) - Orchestrator pattern, multi-agent

✅ Traceloop Translator Validation - Backend span processing verification

✅ Trace Details Backend Migration - Span Store API (replacing logs)

✅ AI Defense Integration with eventID correlation and risk visualization

✅ Platform-Side Evaluations with multi-judge backends (OpenAI + Circuit)

✅ LangGraph multi-agent workflows (confirmed working)

✅ RBAC for conversation content access

✅ Cost Tracking and alerting integration

✅ Streaming support with TTFT metrics

✅ Interaction List (AI Trace Explorer)

✅ LiteLLM Proxy telemetry (TIAA requirement)

Out of Scope (Deferred to PI3):

❌ Multi-turn Session Support (de-prioritized to PI3 - semantic convention discussion ongoing)

❌ Session List View and Session Detail View UI (dependent on session support)

❌ Session ID attribute (gen_ai.session.id) - not in PI2

❌ Enhanced LOC Configuration (instrumentation-side evals discontinued)

❌ JavaScript/Node.js instrumentation (PI3 scope - not started by dev team)

❌ Custom evaluations (PI3 scope)

❌ Workflow visualization (PI3 scope)

❌ MCP Protocol in PI1 apps (only in Foundation demo)

1.3 Test Strategy (UPDATED)
Approach: Use 3 demo setups (not feature-split apps) with phased execution focusing on App 1 first, then validating feature parity across Apps 2 and 3. High automation (≥50%) to meet January 20th deadline with single QE resource.

Key Strategy Change:

"Focus on 1 app first where everything should be working, then introduce demo 2, demo 3. This ensures at least ONE working demo by Week 3."

Consolidated Test Applications (3 DEMO SETUPS):

APP 1: Foundation Demo (COMPREHENSIVE) - PRIMARY FOCUS 
File: foundation_unified_demo_app.py
Architecture: Microservices, distributed agents (TIAA reference architecture)
Epic: HYBIM-413 | Customer: TIAA

Instrumentation:

✅ LangChain/LangGraph (zero-code + custom instrumentation)

✅ In-house framework instrumentation

✅ MCP Protocol support

⚠️ NOT Traceloop (separate app)

Platform Features (ALL features in this one app):

✅ AI Defense integration (eventID correlation)

✅ Platform-side evaluations (multi-judge: OpenAI + Circuit)

✅ Cost tracking and alerting

✅ RBAC content access

✅ Streaming with TTFT

✅ RAG pipeline

✅ LiteLLM routing scenarios

Test Coverage: 28 test cases
Timeline: Week 2-3 focus - Get this 100% working first

APP 2: Traceloop Translator Demo - BACKEND VALIDATION
File: traceloop_travel_planner_app.py (enhanced)
Architecture: Traceloop SDK instrumentation
Epic: RDMP-3228

Purpose: Validate backend processes Traceloop span data differently than native instrumentation

Platform Features (validate parity):

✅ Platform evaluations work with Traceloop

✅ AI Defense works with Traceloop

✅ Cost tracking works with Traceloop

✅ All UI features work with Traceloop data

Test Coverage: 2 test cases (Traceloop validation + feature parity)
Timeline: Week 4 - After App 1 proven

APP 3: Hybrid Setup Demo (OPTIONAL)
File: Custom hybrid app (if time permits)
Architecture: Mix of instrumentation types, edge cases
Epic: QSE-3667

Purpose: Validate platform features work with mixed instrumentation setups

Test Coverage: 2 test cases (edge case validation)
Timeline: Week 5 - If time permits, otherwise defer to post-GA

1.4 Key Metrics (FINAL)
Metric

Value

Justification

Total Test Cases

30

26 original + 3 new + 1 updated

P0 Test Cases

30 (100%)

All GA-blocking

Test Applications

3 demo setups

App 1 focus, then Apps 2-3

Total Effort

22 days

4.4 days/week × 5 weeks

Timeline

4 weeks

Dec 16 → Jan 20, 2026

Automation Target

≥50%

20/30 tests automated

Budget

$750

Includes new test cases

1.5 Success Criteria
✅ 100% P0 test cases pass (30/30 - GA blocking)

✅ App 1 100% functional by Week 3 (primary demo ready)

✅ No P0 bugs at GA (zero tolerance)

✅ Automation ≥50% (15/30 tests automated)

✅ Multi-realm validated (US1, RC0)

✅ Foundation demo app operational (TIAA requirement)

✅ LiteLLM telemetry working (prevent Lumenova switch)

✅ Traceloop backend processing validated

✅ Budget ≤$750

2. Prerequisites and Dependencies 
2.1 Critical Dependencies
Dependency

Impact

Owner

Mitigation

Foundation Demo App (HYBIM-413)

Unified architecture validation

Dev Team

Daily check-ins

Span Store API for Trace Details

Backend migration from logs

Dev Team

Test with current API if available

Traceloop Backend Support

Span data processing for Traceloop

Dev Team

Can test with existing if works

LiteLLM Proxy Fix

TIAA metrics not visible

Dev Team

OTel dependency fix

Platform Evaluation Service

Quality scoring backend

Dev Team

No fallback - GA blocker

AI Defense Integration

EventID correlation support

Dev Team

Mock eventID for testing

MCP Protocol Support

Agent-as-tool pattern

Dev Team

Required for Foundation

Circuit LLM Judge

Second evaluation backend

Dev Team

Can launch with OpenAI only

3. Test Scope 
3.1 Functional Requirements Coverage
Epic

Requirements

Test Cases

Priority

Primary App

HYBIM-413

Foundation, Orchestrator, MCP, RAG

7

P0

App 1

RDMP-3228

AI Defense, Alerting, LiteLLM, Cost, Traceloop

11

P0

App 1, 2

QSE-3131

Platform evals, RBAC, LangGraph, Multi-judge

7

P0

App 1, 3

QSE-3667

UI automation, Trace Details backend

5

P0

All

Total: 4 epics, 30 test cases, 3 demo setups

4. Test Scenarios (30 P0 CASES)
4.1 High-Level Test Scenarios
ID

Scenario

Test Cases

App

Week

Change

TS-PI2-16

Foundation Components

5

App 1

Week 2-3

No change

TS-PI2-17

LiteLLM Proxy

2

App 1

Week 2

Changed to App 1

TS-PI2-NEW-01

Traceloop Translator

1

App 2

Week 4

✅ NEW

TS-PI2-NEW-02

Trace Details Backend

1

All

Week 2

✅ NEW

TS-PI2-01

Session Tracking

3

App 1

Week 2

❌ REMOVED

TS-PI2-02

Alerting & Notifications

2

App 1

Week 3

Changed to App 1

TS-PI2-03

AI Defense + EventID

3

App 1

Week 2-3

Changed to App 1

TS-PI2-04

Platform Evals (Multi-Judge)

3

App 1

Week 3

+1 test, App 1

TS-PI2-05

LangGraph Multi-Agent

1

App 1

Week 2

No change

TS-PI2-06

RBAC Content Access

2

App 1

Week 4

Changed to App 1

TS-PI2-07

Cost Tracking

2

App 1

Week 3-4

Changed to App 1

TS-PI2-08

Interaction List

1

All

Week 3

No change

TS-PI2-09

Streaming with TTFT

1

App 1

Week 2

Reduced to 1, App 1

TS-PI2-10

RAG Observability

1

App 1

Week 3

No change

TS-PI2-11

Test Data

1

All

Week 1

No change

TS-PI2-13

Multi-Realm

1

All

Week 4

No change

Summary: 15 scenarios | 30 P0 test cases | 3 demo setups (App 1 primary focus)

4.2 Complete Test Case List (30 P0 CASES )
#

Test ID

Test Description

App

Priority

Effort

Week

Automation

Foundation Components (5 test cases)

 

 

 

 

 

 

 

1

TC-PI2-FOUNDATION-01

Validate orchestrator pattern with sub-agents captures workflow hierarchy and agent coordination

App 1

P0

1d

Week 2

✅ Yes

2

TC-PI2-FOUNDATION-02

Verify parallel agent execution shows overlapping timestamps and correct parent-child relationships

App 1

P0

1d

Week 2

✅ Yes

3

TC-PI2-FOUNDATION-03

Test MCP protocol agent-as-tool pattern captures MCP spans and agent invocation correctly

App 1

P0

1d

Week 3

✅ Yes

4

TC-PI2-FOUNDATION-04

Verify telemetry parity across in-house, LangChain, Traceloop flavors produces identical spans

App 1

P0

1d

Week 3

✅ Yes

5

TC-PI2-FOUNDATION-05

Validate RAG pipeline captures vector database operations, embedding costs, retrieval quality metrics

App 1

P0

1d

Week 3

⚠️ Partial

LangGraph Multi-Agent (1 test case)

 

 

 

 

 

 

 

6

TC-PI2-PLATFORM-05

Verify LangGraph multi-agent workflow captures all agents, handoffs, and tools in complete trace hierarchy

App 1

P0

1d

Week 2

✅ Yes

RAG Observability (1 test case)

 

 

 

 

 

 

 

7

TC-PI2-RAG-01

Validate RAG pipeline end-to-end with vector DB queries, retrieval metrics, embedding cost attribution

App 1

P0

1d

Week 3

⚠️ Partial

LiteLLM Proxy (2 test cases)

 

 

 

 

 

 

 

8

TC-PI2-LITELLM-01

Verify LiteLLM Proxy metrics show request counts, latency, backend provider attribution in dashboard

App 1

P0

1d

Week 2

✅ Yes

9

TC-PI2-LITELLM-02

Validate end-to-end trace correlation from client through proxy to backend LLM with complete spans

App 1

P0

1d

Week 2

✅ Yes

AI Defense Integration with EventID (3 test cases)

 

 

 

 

 

 

 

10

TC-PI2-AIDEF-01

Verify AI Defense API mode detects security/privacy risks and instrumentation captures eventID for correlation

App 1

P0

1d

Week 2

⚠️ Partial

11

TC-PI2-AIDEF-02

Test AI Defense Proxy mode with eventID tracking and validate instrumentation supports eventID attribute

App 1

P0

1d

Week 2

⚠️ Partial

12

TC-PI2-AIDEF-03

Validate security/privacy risks visible in Agent List (Risks), Agent Details (Risks dashboard), Trace Details (Risk tab with eventID)

App 1

P0

1d

Week 3

✅ Yes

Platform Evaluations with Multi-Judge (3 test cases)

 

 

 

 

 

 

 

13

TC-PI2-EVAL-01

Verify platform evaluation configuration allows sampling rate and evaluator selection with persistence

App 1

P0

0.5d

Week 3

⚠️ Partial

14

TC-PI2-EVAL-02

Validate OpenAI judge (gpt-4o-mini) generates quality scores (toxicity, bias, hallucination, relevance) correctly

App 1

P0

1d

Week 3

⚠️ Partial

15

TC-PI2-EVAL-03

Validate Circuit judge generates quality scores and compare with OpenAI judge for consistency (non-deterministic models)

App 1

P0

1d

Week 3

⚠️ Partial

Alerting with Troubleshoot Flow (2 test cases)

 

 

 

 

 

 

 

16

TC-PI2-ALERT-01

Test alert creation for anomalous token usage, quality degradation, security risks with autodetect and thresholds

App 1

P0

0.5d

Week 3

⚠️ Partial

17

TC-PI2-ALERT-02

Verify alert email with Troubleshoot button navigates to AI Overview or Agent Details with alert context preserved

App 1

P0

1d

Week 3

✅ Yes

Cost Tracking (2 test cases)

 

 

 

 

 

 

 

18

TC-PI2-COST-01

Validate cost calculation accuracy for different models using published token costs within 3%

App 1

P0

1d

Week 3

✅ Yes

19

TC-PI2-COST-02

Verify cost displays in Agent List (Cost column), Agent Details charts, cost dashboards accurately

App 1

P0

1d

Week 4

✅ Yes

RBAC Content Access (2 test cases)

 

 

 

 

 

 

 

20

TC-PI2-RBAC-01

Verify RBAC role configuration creates AI Conversation Viewer role with correct content access permissions

App 1

P0

0.5d

Week 4

⚠️ Partial

21

TC-PI2-RBAC-03

Test Viewer role users see redacted conversation content in Trace Details while metadata remains visible

App 1

P0

1d

Week 4

✅ Yes

Streaming with TTFT (1 test case)

 

 

 

 

 

 

 

22

TC-PI2-STREAMING-01

Verify streaming captures time-to-first-token metrics with P95 under 500ms and handles mid-stream failures

App 1

P0

1d

Week 2

✅ Yes

Traceloop Translator Validation (1 test case) - NEW

 

 

 

 

 

 

 

23

TC-PI2-TRACELOOP-01

Validate Traceloop instrumentation data processed correctly by backend and all platform features work (evals, AI Defense, cost)

App 2

P0

1d

Week 4

✅ Yes

Trace Details Backend Migration (1 test case) - NEW

 

 

 

 

 

 

 

24

TC-PI2-TRACE-DETAILS-01

Verify Trace Details page uses Span Store API (not logs), conversation content displays correctly, performance acceptable

All

P0

1d

Week 2

✅ Yes

Interaction List (1 test case)

 

 

 

 

 

 

 

25

TC-PI2-EXPLORER-01

Test AI Trace Data filtering by agent, quality issues, security/privacy risks with accurate result display

All

P0

1d

Week 3

✅ Yes

Azure Provider (1 test case)

 

 

 

 

 

 

 

26

TC-PI2-PLATFORM-06

Test Azure OpenAI provider produces schema-compliant telemetry with accurate token counts and cost

App 1

P0

1d

Week 2

✅ Yes

Test Data (1 test case)

 

 

 

 

 

 

 

27

TC-PI2-DATA-01

Validate Git LFS test data versioning with 5000 prompts, PII masking, monthly refresh automation

All

P0

1d

Week 1

⚠️ Partial

Multi-Realm (1 test case)

 

 

 

 

 

 

 

28

TC-PI2-02

Test multi-realm deployment to RC0 and US1 with realm metadata, data isolation, configuration

All

P0

0.5d

Week 4

✅ Yes

Feature Parity Validation (2 test cases) - Apps 2-3

 

 

 

 

 

 

 

29

TC-PI2-PARITY-01

Validate all platform features work identically on App 2 (Traceloop) as App 1

App 2

P0

1d

Week 4

✅ Yes

30

TC-PI2-PARITY-02

Validate all platform features work identically on App 3 (hybrid) as App 1 (if time permits)

App 3

P1→P0

0.5d

Week 5

✅ Yes

Total: 30 P0 test cases

Changes:

✅ Added 3 new tests: Traceloop, Trace Details Backend, Multi-Judge

✅ Added 2 parity tests: Cross-app validation

❌ Removed 4 session tests: SESSION-01, 02, 03 + STREAMING-02

❌ Removed 1 test: Zero-Code (already validated)

✅ Net change: 29 - 4 + 5 = 30 tests

5. UI Functional Testing 
5.1 UI Test Coverage
UI Component

What's Being Tested

Test Cases

Priority

App

Week

Agent List Page

NEW: "Cost" column, "Risks" column

2

P0

App 1

Week 3

Agent Details Page

NEW: "Quality Issues" dashboard, "Risks" dashboard

2

P0

App 1

Week 3

Trace Details Page

NEW: "Security" waterfall option, "Risk" tab (5 categories), Span Store backend

3

P0

App 1

Week 2-3

AI Trace Data

Filter by security/privacy risks, quality issues

1

P0

App 1

Week 3

Alert Email Flow

Troubleshoot button → AI Overview or Agent Details navigation

1

P0

App 1

Week 3

APM Navigation

NEW: AI Agent Monitoring section (AI Overview, AI Trace Data, AI Agents)

1

P0

App 1

Week 2

Session List

Session filtering, sorting

1

P0

--

❌ REMOVED

Session Detail

Turn-by-turn display

1

P0

--

❌ REMOVED

Total: 10 UI test cases (down from 12)

All UI tests focus on App 1 first, then validate parity on Apps 2-3

6. Regression Testing
6.1 Regression Test Suite 
Suite

Test Cases

Automation

Apps

Notes

Alpha Regression

12

50%

All 3

Validate PI1 features still work

PI2 Core Regression

15

50%

App 1

Foundation, AI Defense, Platform Eval

Foundation App Regression

7

50%

App 1

Orchestrator, MCP, RAG specific

UI Regression

8

50%

App 1

New UI elements (Cost, Risks columns, Risk tab)

Traceloop Regression

2

50%

App 2

Backend span processing validation

Total

44

50%

3 Apps

Sessions removed (-4 tests)

Regression Strategy:

Week 2-3: Focus on App 1 regression (25 tests)

Week 4: Add Apps 2-3 regression (19 tests)

7. Security Testing
7.1 Security Test Cases
Test

Focus

App

Priority

Week

Notes

RBAC Enforcement

Unauthorized content access prevention

App 1

P0

Week 4

Viewer role cannot see prompts/responses

Secrets Exposure

API keys not in telemetry

All

P0

Week 2

Automated scan of spans

PII Leakage

Customer PII not in traces

All

P0

Week 2

Automated PII detection

Audit Logging

Sensitive operations logged

All

P0

Week 4

Test 10 sensitive actions

Prompt Injection (AI Defense)

Malicious prompts detected

App 1

P0

Week 2

Test 50 injection attacks

Toxic Content (AI Defense)

Harmful content flagged

App 1

P0

Week 2

Test 50 toxic prompts

Total: 6 security tests | Security Review: Week 3 with Dev Team

8. Reliability Testing
8.1 Reliability Test Cases
Test

Failure Scenario

Recovery Target

App

Priority

Notes

LLM Provider Outage

OpenAI down 5min

<5min recovery

All

P1

Graceful degradation

Span Store Outage

Span Store unavailable

UI shows error

All

P1

No data loss

AI Defense Outage

AI Defense API down

No risk detection shown

App 1

P1

Spans still captured

Foundation Orchestrator Failure

Orchestrator crash

Checkpoint recovery

App 1

P1

Workflow resumes

Total: 4 reliability tests | All P1 (important but not GA blocking)

9. Configuration Testing
9.1 Configuration Matrix
Configuration

Variations

App

Priority

Notes

Instrumentation Flavor

In-house, LangChain, Traceloop

App 1, 2

P0

Validate telemetry parity

Evaluation Judge

OpenAI, Circuit

App 1

P0

Multi-backend validation

Evaluation Sampling

10%, 50%, 100%

App 1

P0

Platform-side only

LangGraph Agents

2, 3, 5 agents

App 1

P0

Multi-agent complexity

AI Defense Mode

API, Proxy, Disabled

App 1

P0

Both modes + no AI Defense

RBAC Roles

Admin, Viewer, None

App 1

P0

Content access levels

Streaming

Enabled, Disabled

App 1

P0

TTFT metrics presence

MCP Protocol

Enabled, Disabled

App 1

P0

Agent-as-tool pattern

Matrix: 8 dimensions × 2-3 variations = 15 combinations for efficiency

10. Internationalization
10.1 Non-ASCII Input Validation
Test

Languages

Expected

Priority

App

Prompt/Response Encoding

Japanese, Emoji

UTF-8 preserved in traces

P0

App 1

UI Display

Non-ASCII characters

Correct rendering in Agent List, Trace Details

P0

App 1

10.2 Time Zone Handling
Test

Formats

Expected

Priority

App

Time Zone Handling

PST, UTC

Correct conversion, UTC storage

P0

All

Alert Timestamps

Cross-timezone alerts

Consistent UTC storage

P0

App 1

Localization: ⚠️ Out of scope - English only for GA

11. Browser Support
11.1 Supported Browsers
Browser

Version

Platform

Coverage

Priority

Chrome

Latest 2

Windows, Mac, Linux

Full Playwright suite

P0

Firefox

Latest 2

Windows, Mac, Linux

Full Playwright suite

P0

Safari

Latest 2

Mac

Full Playwright suite

P0

Edge

Latest 2

Windows

Smoke tests only

P1

Strategy: Playwright cross-browser parallel execution
Smoke Tests: Agent List, Trace Details, AI Trace Data

12. Tools
12.1 Test Automation Tools
Tool

Purpose

Version

Setup Status

Week

Playwright

UI automation

≥1.40.0

⚠️ Setup

Week 1

PyTest

Backend/API testing

≥7.4.0

✅ Available

Week 1

Git LFS

Test data versioning

Latest

⚠️ Setup

Week 1

Docker

App containerization

Latest

✅ Available

Week 1

Kubernetes

App orchestration

≥1.25

✅ Available

Week 1

Postman

API testing (optional)

Latest

⚠️ Optional

Week 2

12.2 CI/CD
Platform: Jenkins
Integration: Slack webhooks for notifications
Automation: Nightly regression runs

12.3 Monitoring
Splunk APM: Primary observability platform

Splunk Log Observer: Log correlation

AI Defense Dashboard: Security risk monitoring

13. Deployment
13.1 Deployment Strategy
CI/CD: Jenkins automation
Realms: RC0 (comprehensive testing), US1 (smoke validation)
Applications: 3 demo setups (App 1 primary, Apps 2-3 parity)

Deployment Approach:

Week 1: Deploy all 3 apps to RC0

Week 2-3: Focus testing on App 1 only

Week 4: Enable testing on Apps 2-3

13.2 Testing Per Realm
Realm

Purpose

Test Scope

Apps

Timeline

RC0

Pre-production

All 30 P0 tests

All 3

Week 1-3

US1

Production validation

Smoke (12 critical tests)

App 1 only

Week 4

Smoke Test Selection (12 tests for US1):

Foundation: 2 tests (Orchestrator, Multi-agent)

AI Defense: 2 tests (API mode, UI visibility)

Platform Eval: 1 test (Execution)

Cost: 1 test (Calculation)

LiteLLM: 1 test (Metrics)

Trace Details: 1 test (Backend)

Alerting: 1 test (Triggering)

RBAC: 1 test (Viewer access)

UI: 2 tests (Agent List, Trace Details)

14. Test Execution Schedule 
14.1 Weekly Breakdown (App 1 Focus Strategy)
Week

Phase

Focus

Test Cases

Primary App

Deliverables

Week 1

Setup

Environment, 3 demo setups

1

Setup

Test env + 3 apps enhanced

Week 2

App 1 Core

Foundation, LiteLLM, AI Defense, Trace Details

12

App 1

App 1 core features working

Week 3

App 1 Complete

Platform Evals, RAG, UI, Alerts, Cost

11

App 1

App 1 100% functional

Week 4

Apps 2-3

Traceloop, RBAC, Parity validation

4

App 2, 3

Regression, Multi-realm, Sign-off + (optional)Feature parity confirmed

Total: 30 test cases

Key Strategy:

✅ Week 2-3: Get App 1 (Foundation) 100% working with ALL features

✅ Week 4: Validate Apps 2-3 work identically

✅ Early Demo: Can demonstrate to TIAA by end of Week 3

15. Risk Assessment 
15.1 Critical Risks
Risk

Probability

Impact

Mitigation

Status

TIAA → Lumenova

60%

CRITICAL

App 1 focus Week 2-3, demo by end Week 3

Active

Foundation App Delays

40%

HIGH

Daily sync, Dec 31 target, can demo partial

Active

Platform Eval Service

50%

CRITICAL

No fallback - Jan 12 ETA, monitor daily

Active

Single QE Resource

90%

HIGH

50% automation, phased approach reduces risk

Active

Span Store API Migration

30%

MEDIUM

Test current API, validate migration smooth

Active

Circuit Judge Integration

25%

LOW

Can launch with OpenAI only, Circuit nice-to-have

Low

Risk Mitigation: Phased approach (App 1 first) = Working demo by Week 3 even if Apps 2-3 delayed

DOCUMENT STATUS
Final Numbers:

Test Cases: 30 P0 (0 P1)

Applications: 3 demo setups

Primary Focus: App 1 (Weeks 2-3)

Budget: $750

Timeline: 4 weeks

Automation: 50% (20/30)