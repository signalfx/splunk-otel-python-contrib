# Database Connection Pool Exhaustion

## Overview
This runbook addresses issues where database connection pools are exhausted, preventing new connections from being established.

## Symptoms
- High number of active database connections (near pool limit)
- Connection timeout errors in application logs
- Increased latency for database operations
- New connection requests being rejected

## Root Causes
1. Connection leaks (connections not properly closed)
2. Long-running transactions holding connections
3. Insufficient pool size for current load
4. Database server performance degradation

## Investigation Steps

### 1. Check Connection Pool Metrics
- Query: `sum(active_connections) by (service, database)`
- Threshold: > 80% of max_pool_size
- Action: Identify services with high connection usage

### 2. Check for Connection Leaks
- Query logs for: `connection.*not.*closed|connection.*leak`
- Review application code for missing connection.close() calls
- Check for uncommitted transactions

### 3. Check Database Server Health
- Query: `database_cpu_usage`, `database_memory_usage`
- Check for slow queries: `database_query_duration_p95`
- Review database connection limits

### 4. Review Recent Deployments
- Check if recent code changes affected connection handling
- Review connection pool configuration changes

## Mitigation Steps

### Immediate Actions (High Confidence)
1. **Restart Connection Pool** (if safe)
   - Gracefully drain connections
   - Restart pool manager
   - Monitor connection establishment

2. **Scale Up Database Connections** (if infrastructure allows)
   - Increase max_pool_size temporarily
   - Monitor for improvement

### Short-term Actions
1. **Identify and Fix Connection Leaks**
   - Review application code
   - Add connection monitoring
   - Implement connection timeout policies

2. **Optimize Long-running Queries**
   - Identify queries holding connections
   - Add query timeouts
   - Optimize slow queries

### Long-term Actions
1. **Implement Connection Pool Monitoring**
   - Set up alerts for pool exhaustion
   - Add metrics for connection lifecycle
   - Implement connection pool health checks

2. **Review Connection Pool Sizing**
   - Analyze historical usage patterns
   - Right-size pool configurations
   - Implement auto-scaling if possible

## Rollback Plan
- If mitigation causes issues, revert pool size changes
- Restore previous connection pool configuration
- Monitor for stability

## Safety Notes
- **WARNING**: Restarting connection pools may cause brief service interruption
- Always coordinate with database team before scaling connections
- Test connection pool changes in staging first

## References
- Database connection pool documentation
- Application connection management guide
- Incident history: INC-2024-001

