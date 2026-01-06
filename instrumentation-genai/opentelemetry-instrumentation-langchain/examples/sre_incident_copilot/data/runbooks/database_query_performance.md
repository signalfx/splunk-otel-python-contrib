# Database Query Performance Degradation

## Overview
This runbook addresses slow database queries causing service latency degradation.

## Symptoms
- Increased database query duration
- High P95 latency for database operations
- Slow response times for queries
- Database CPU/memory pressure

## Root Causes
1. Missing or inefficient database indexes
2. Large result sets without pagination
3. N+1 query problems
4. Database connection issues
5. Table locks or blocking queries
6. Outdated query plans

## Investigation Steps

### 1. Check Database Performance Metrics
- Query: `db_query_duration_p95`, `db_query_duration_p99`
- Query: `db_cpu_usage`, `db_memory_usage`
- Threshold: Query duration > 1 second
- Action: Identify slow queries

### 2. Review Slow Query Logs
- Query logs for: `slow.*query|query.*duration|long.*running`
- Identify specific slow queries
- Check query execution plans

### 3. Check Database Locks
- Query: `db_locks`, `db_blocking_queries`
- Identify table locks
- Check for blocking queries

### 4. Review Query Patterns
- Analyze query frequency
- Check for N+1 query patterns
- Review result set sizes

### 5. Check Indexes
- Verify indexes exist on queried columns
- Check index usage statistics
- Identify missing indexes

## Mitigation Steps

### Immediate Actions (High Confidence)
1. **Kill Blocking Queries** (if safe)
   - Identify and terminate long-running queries
   - Release table locks
   - Monitor for improvement

2. **Add Missing Indexes** (if identified)
   - Create indexes on frequently queried columns
   - Monitor query performance improvement
   - Verify index usage

### Short-term Actions
1. **Optimize Slow Queries**
   - Rewrite inefficient queries
   - Add proper pagination
   - Reduce result set sizes

2. **Implement Query Caching**
   - Cache frequently accessed data
   - Reduce database load
   - Improve response times

### Long-term Actions
1. **Implement Query Monitoring**
   - Set up alerts for slow queries
   - Monitor query performance trends
   - Track query execution plans

2. **Database Optimization**
   - Regular query plan reviews
   - Index maintenance
   - Database statistics updates

## Rollback Plan
- Revert query changes if performance degrades
- Remove indexes if causing issues
- Restore previous query logic

## Safety Notes
- **WARNING**: Killing queries may cause data inconsistency
- Always coordinate with database team
- Test query optimizations in staging first
- Monitor database performance after changes

## References
- Database performance tuning guide
- Query optimization best practices
- Incident history: INC-2024-008

