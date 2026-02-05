# Archived Test Applications

This directory contains test applications that are archived for reference purposes only.

## 📦 Archived Apps

### `multi_agent_hierarchy_test_app.py`

**Original Name:** `direct_azure_openai_app.py`

**Purpose:** Multi-agent hierarchy testing with complex organizational workflows

**Architecture:**
```
Research Department (Parent Agent)
├── Customer Service Department
├── Legal & Compliance Department
├── Research & Analysis Department
└── Human Resources Department
```

**Key Features:**
- 4-department organizational structure
- Parent-child agent relationships
- Multi-level LLM call hierarchy
- Department-specific evaluation metrics

**Test Scenarios:**
1. Baseline Positive (Control) - Professional customer support
2. Normal Request - Market analysis workflow

**Why Archived:**
- Focused on multi-agent hierarchy patterns rather than evaluation metrics validation
- Only tests baseline scenarios (both PASS) - doesn't validate evaluation failures
- Superseded by `direct_azure_openai_app_v2.py` for GA verification testing
- Older configuration approach (pre-timing fix)

**When to Use:**
- Testing multi-level agent hierarchies
- Validating parent-child agent communication
- Reference for complex organizational workflows
- Future multi-department test scenarios

**Status:** Working code, but requires timing fix (5s/120s) if used for evaluation testing

---

## 🔄 Migration Notes

If you need to use this archived app:

1. **Apply Timing Fix:**
   - Change delay between scenarios: 2s → 5s
   - Change evaluation wait time: 60s → 120s

2. **Update Configuration:**
   - Use config-driven environment variables (see `direct_azure_openai_app_v2.py`)
   - Enable evaluation completion callbacks
   - Set appropriate evaluation sample rate

3. **Consider Using Primary App:**
   - For evaluation metrics testing, use `direct_azure_openai_app_v2.py` instead
   - This archived app is best for multi-agent hierarchy patterns

---

## 📚 Related Documentation

- **Primary GA Test App:** `../direct_azure_openai_app_v2.py`
- **Main README:** `../README.md`
- **Evaluation Issue Resolution:** `../EVALUATION_ISSUE_SUMMARY.md`

---

**Last Updated:** January 30, 2026  
**Archived By:** QA/Dev Team  
**Reason:** Consolidation to single primary GA verification test app
