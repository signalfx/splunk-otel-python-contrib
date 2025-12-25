# Downstream Service Failure

## Overview
This runbook addresses incidents where a downstream dependency service fails, causing cascading failures in dependent services.

## Symptoms
- Increased error rates in dependent services
- Timeout errors when calling downstream services
- Circuit breaker activations
- Cascading failures across multiple services

## Root Causes
1. Downstream service outage
2. Downstream service performance degradation
3. Network issues between services
4. Downstream service rate limiting
5. Downstream service configuration issues

## Investigation Steps

### 1. Identify Affected Services
- Query: `error_rate by (service)` - identify spikes
- Check service dependency graph
- Identify common downstream dependency

### 2. Check Downstream Service Health
- Query: `downstream_service_availability`
- Query: `downstream_service_latency_p95`
- Query: `downstream_service_error_rate`
- Check downstream service status page/dashboard

### 3. Check Circuit Breaker Status
- Query: `circuit_breaker_state`
- Check if circuit breakers are open
- Review circuit breaker logs

### 4. Review Network Metrics
- Query: `network_latency between services`
- Query: `network_error_rate`
- Check for network partition issues

### 5. Check Traces
- Review trace samples for downstream calls
- Identify timeout patterns
- Check for retry exhaustion

## Mitigation Steps

### Immediate Actions (High Confidence)
1. **Enable Circuit Breaker** (if not already enabled)
   - Open circuit to failing downstream service
   - Return fallback responses if available
   - Monitor circuit breaker state

2. **Implement Fallback Logic**
   - Return cached responses if available
   - Use default values where appropriate
   - Gracefully degrade functionality

### Short-term Actions
1. **Scale Up Retry Logic** (if appropriate)
   - Increase retry attempts temporarily
   - Implement exponential backoff
   - Monitor retry success rate

2. **Contact Downstream Team**
   - Notify downstream service team
   - Share error logs and metrics
   - Coordinate resolution

### Long-term Actions
1. **Implement Resilience Patterns**
   - Add circuit breakers to all downstream calls
   - Implement fallback mechanisms
   - Add timeout configurations

2. **Improve Monitoring**
   - Set up alerts for downstream service health
   - Monitor circuit breaker activations
   - Track dependency health metrics

## Rollback Plan
- Disable circuit breakers if causing issues
- Revert timeout/retry changes
- Restore previous fallback logic

## Safety Notes
- **WARNING**: Circuit breakers may cause functionality loss
- Always have fallback responses ready
- Coordinate with downstream teams
- Test circuit breaker behavior in staging

## References
- Circuit breaker pattern documentation
- Service dependency architecture
- Incident history: INC-2024-004

