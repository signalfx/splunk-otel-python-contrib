# Recent Deployment Issue

## Overview
This runbook addresses incidents that occur shortly after a deployment, indicating a potential issue with the new release.

## Symptoms
- Error rate spike within 30 minutes of deployment
- Latency increase after deployment
- New error types in logs
- Service degradation correlated with deployment timestamp

## Root Causes
1. Code bug introduced in new release
2. Configuration change causing issues
3. Dependency version incompatibility
4. Resource allocation changes
5. Feature flag misconfiguration

## Investigation Steps

### 1. Correlate Timeline
- Identify deployment timestamp
- Check when symptoms started
- Verify correlation (< 30 minutes is strong indicator)

### 2. Review Deployment Changes
- Check git diff for recent changes
- Review configuration changes
- Check dependency updates
- Review feature flags enabled

### 3. Check Error Patterns
- Query logs: `error.*after.*deployment|deployment.*error`
- Compare error rates before/after deployment
- Identify new error types
- Check for stack traces

### 4. Check Metrics Comparison
- Compare metrics before/after deployment
- Query: `error_rate`, `latency_p95`, `throughput`
- Identify which metrics degraded

### 5. Review Traces
- Check trace samples for new failure patterns
- Identify slow endpoints
- Check for new dependency calls

## Mitigation Steps

### Immediate Actions (High Confidence)
1. **Rollback Deployment** (if safe)
   - Revert to previous known-good version
   - Monitor for improvement
   - Document rollback reason

2. **Disable Feature Flags** (if applicable)
   - Disable newly enabled features
   - Monitor for improvement
   - Re-enable gradually if needed

### Short-term Actions
1. **Hotfix Critical Issues** (if rollback not possible)
   - Identify and fix critical bugs
   - Deploy hotfix quickly
   - Monitor for resolution

2. **Scale Up Resources** (if resource-related)
   - Increase resource allocation temporarily
   - Monitor for improvement
   - Plan proper resource sizing

### Long-term Actions
1. **Improve Deployment Process**
   - Add canary deployments
   - Implement gradual rollouts
   - Add automated rollback triggers

2. **Enhance Testing**
   - Improve test coverage
   - Add integration tests
   - Implement chaos testing

## Rollback Plan
- **CRITICAL**: Always have rollback plan ready
- Test rollback procedure regularly
- Document rollback steps
- Coordinate with team before rollback

## Safety Notes
- **WARNING**: Rollbacks may cause data inconsistency if not handled carefully
- Always coordinate with team before rollback
- Verify rollback target version is stable
- Monitor closely after rollback

## Post-Incident Actions
1. Conduct postmortem
2. Document root cause
3. Update deployment process
4. Add monitoring/alerting for similar issues

## References
- Deployment process documentation
- Rollback procedure guide
- Incident history: INC-2024-003

