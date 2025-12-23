"""MCP tools for observability data (metrics, logs, traces)."""
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastmcp import FastMCP

# Suppress MCP server startup logs
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

mcp = FastMCP("observability-tools")

# Seeded data for consistent responses
SEEDED_METRICS_DATA = {
    "scenario-001": {
        "error_rate": [
            {"timestamp": "2024-01-16T10:30:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T10:35:00Z", "value": 0.02},
            {"timestamp": "2024-01-16T10:40:00Z", "value": 2.5},
            {"timestamp": "2024-01-16T10:45:00Z", "value": 5.2},
        ],
        "latency_p95": [
            {"timestamp": "2024-01-16T10:30:00Z", "value": 200},
            {"timestamp": "2024-01-16T10:35:00Z", "value": 250},
            {"timestamp": "2024-01-16T10:40:00Z", "value": 450},
            {"timestamp": "2024-01-16T10:45:00Z", "value": 680},
        ],
        "active_connections": [
            {"timestamp": "2024-01-16T10:30:00Z", "value": 45},
            {"timestamp": "2024-01-16T10:35:00Z", "value": 78},
            {"timestamp": "2024-01-16T10:40:00Z", "value": 95},
            {"timestamp": "2024-01-16T10:45:00Z", "value": 100},
        ],
    },
    "scenario-002": {
        "latency_p95": [
            {"timestamp": "2024-01-16T10:50:00Z", "value": 150},
            {"timestamp": "2024-01-16T10:55:00Z", "value": 320},
            {"timestamp": "2024-01-16T11:00:00Z", "value": 850},
        ],
        "cache_hit_rate": [
            {"timestamp": "2024-01-16T10:50:00Z", "value": 0.85},
            {"timestamp": "2024-01-16T10:55:00Z", "value": 0.45},
            {"timestamp": "2024-01-16T11:00:00Z", "value": 0.12},
        ],
    },
    "scenario-003": {
        "error_rate": [
            {"timestamp": "2024-01-16T11:10:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T11:12:00Z", "value": 5.5},
            {"timestamp": "2024-01-16T11:15:00Z", "value": 12.0},
        ],
        "latency_p95": [
            {"timestamp": "2024-01-16T11:10:00Z", "value": 50},
            {"timestamp": "2024-01-16T11:12:00Z", "value": 200},
            {"timestamp": "2024-01-16T11:15:00Z", "value": 500},
        ],
    },
    "scenario-004": {
        "active_connections": [
            {"timestamp": "2024-01-16T11:25:00Z", "value": 80},
            {"timestamp": "2024-01-16T11:28:00Z", "value": 95},
            {"timestamp": "2024-01-16T11:30:00Z", "value": 100},
        ],
        "connection_pool_utilization": [
            {"timestamp": "2024-01-16T11:25:00Z", "value": 0.80},
            {"timestamp": "2024-01-16T11:28:00Z", "value": 0.95},
            {"timestamp": "2024-01-16T11:30:00Z", "value": 1.0},
        ],
    },
    "scenario-005": {
        "cache_memory_usage": [
            {"timestamp": "2024-01-16T11:40:00Z", "value": 85},
            {"timestamp": "2024-01-16T11:42:00Z", "value": 92},
            {"timestamp": "2024-01-16T11:45:00Z", "value": 95},
        ],
        "cache_eviction_rate": [
            {"timestamp": "2024-01-16T11:40:00Z", "value": 50},
            {"timestamp": "2024-01-16T11:42:00Z", "value": 200},
            {"timestamp": "2024-01-16T11:45:00Z", "value": 500},
        ],
    },
    "scenario-006": {
        "error_rate": [
            {"timestamp": "2024-01-16T11:55:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T11:58:00Z", "value": 4.0},
            {"timestamp": "2024-01-16T12:00:00Z", "value": 8.0},
        ],
        "authentication_failure_rate": [
            {"timestamp": "2024-01-16T11:55:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T11:58:00Z", "value": 0.05},
            {"timestamp": "2024-01-16T12:00:00Z", "value": 0.08},
        ],
    },
    "scenario-007": {
        "queue_depth": [
            {"timestamp": "2024-01-16T12:10:00Z", "value": 1000},
            {"timestamp": "2024-01-16T12:20:00Z", "value": 25000},
            {"timestamp": "2024-01-16T12:30:00Z", "value": 50000},
        ],
        "queue_processing_rate": [
            {"timestamp": "2024-01-16T12:10:00Z", "value": 100},
            {"timestamp": "2024-01-16T12:20:00Z", "value": 80},
            {"timestamp": "2024-01-16T12:30:00Z", "value": 50},
        ],
        "downstream_service_latency_p95": [
            {"timestamp": "2024-01-16T12:10:00Z", "value": 200},
            {"timestamp": "2024-01-16T12:20:00Z", "value": 1200},
            {"timestamp": "2024-01-16T12:30:00Z", "value": 2500},
        ],
        "downstream_service_error_rate": [
            {"timestamp": "2024-01-16T12:10:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T12:20:00Z", "value": 0.15},
            {"timestamp": "2024-01-16T12:30:00Z", "value": 0.25},
        ],
    },
    "scenario-008": {
        "latency_p95": [
            {"timestamp": "2024-01-16T12:35:00Z", "value": 500},
            {"timestamp": "2024-01-16T12:40:00Z", "value": 1500},
            {"timestamp": "2024-01-16T12:45:00Z", "value": 2500},
        ],
        "db_query_duration_p95": [
            {"timestamp": "2024-01-16T12:35:00Z", "value": 400},
            {"timestamp": "2024-01-16T12:40:00Z", "value": 1400},
            {"timestamp": "2024-01-16T12:45:00Z", "value": 2400},
        ],
    },
    "scenario-009": {
        "error_rate": [
            {"timestamp": "2024-01-16T10:45:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T10:48:00Z", "value": 3.0},
            {"timestamp": "2024-01-16T10:50:00Z", "value": 6.5},
        ],
        "deployment_correlation": [
            {"timestamp": "2024-01-16T10:45:00Z", "value": 0},
            {"timestamp": "2024-01-16T10:48:00Z", "value": 1},
            {"timestamp": "2024-01-16T10:50:00Z", "value": 1},
        ],
    },
    "scenario-010": {
        "error_rate": [
            {"timestamp": "2024-01-16T12:55:00Z", "value": 0.01},
            {"timestamp": "2024-01-16T13:00:00Z", "value": 15.0},
            {"timestamp": "2024-01-16T13:05:00Z", "value": 25.0},
        ],
        "downstream_error_rate": [
            {"timestamp": "2024-01-16T12:55:00Z", "value": 0.0},
            {"timestamp": "2024-01-16T13:00:00Z", "value": 0.20},
            {"timestamp": "2024-01-16T13:05:00Z", "value": 0.35},
        ],
    },
}

SEEDED_LOGS_DATA = {
    "scenario-001": [
        {
            "timestamp": "2024-01-16T10:42:00Z",
            "level": "ERROR",
            "message": "Database connection pool exhausted, unable to acquire connection",
            "service": "payment-service",
            "trace_id": "trace-001",
        },
        {
            "timestamp": "2024-01-16T10:43:00Z",
            "level": "ERROR",
            "message": "Connection timeout after 30s waiting for available connection",
            "service": "payment-service",
            "trace_id": "trace-002",
        },
        {
            "timestamp": "2024-01-16T10:44:00Z",
            "level": "WARN",
            "message": "Active connections at 95% capacity",
            "service": "payment-service",
        },
    ],
    "scenario-002": [
        {
            "timestamp": "2024-01-16T10:58:00Z",
            "level": "WARN",
            "message": "Cache miss rate increased to 55%",
            "service": "user-service",
        },
        {
            "timestamp": "2024-01-16T10:59:00Z",
            "level": "INFO",
            "message": "Cache eviction rate: 12%",
            "service": "user-service",
        },
    ],
    "scenario-003": [
        {
            "timestamp": "2024-01-16T11:13:00Z",
            "level": "ERROR",
            "message": "Downstream service error rate increased",
            "service": "api-gateway",
            "trace_id": "trace-005",
        },
        {
            "timestamp": "2024-01-16T11:14:00Z",
            "level": "ERROR",
            "message": "Recent deployment v3.1.0 causing errors",
            "service": "api-gateway",
        },
    ],
    "scenario-004": [
        {
            "timestamp": "2024-01-16T11:28:00Z",
            "level": "ERROR",
            "message": "Database connection pool exhausted",
            "service": "database-primary",
            "trace_id": "trace-007",
        },
        {
            "timestamp": "2024-01-16T11:28:30Z",
            "level": "ERROR",
            "message": "Connection error: unable to acquire connection from pool",
            "service": "database-primary",
            "trace_id": "trace-007",
        },
        {
            "timestamp": "2024-01-16T11:29:00Z",
            "level": "WARN",
            "message": "Connection requests being rejected",
            "service": "database-primary",
        },
    ],
    "scenario-005": [
        {
            "timestamp": "2024-01-16T11:43:00Z",
            "level": "WARN",
            "message": "Redis memory usage at 95%",
            "service": "redis-cache",
        },
        {
            "timestamp": "2024-01-16T11:44:00Z",
            "level": "INFO",
            "message": "Cache evictions increasing",
            "service": "redis-cache",
        },
    ],
    "scenario-006": [
        {
            "timestamp": "2024-01-16T11:57:00Z",
            "level": "ERROR",
            "message": "Token validation failed",
            "service": "auth-service",
            "trace_id": "trace-009",
        },
        {
            "timestamp": "2024-01-16T11:59:00Z",
            "level": "ERROR",
            "message": "Invalid token signature",
            "service": "auth-service",
            "trace_id": "trace-010",
        },
    ],
    "scenario-007": [
        {
            "timestamp": "2024-01-16T12:25:00Z",
            "level": "WARN",
            "message": "Notification queue depth increasing rapidly",
            "service": "notification-service",
        },
        {
            "timestamp": "2024-01-16T12:26:00Z",
            "level": "WARN",
            "message": "Downstream email service responding slowly, latency increased to 2.5s",
            "service": "notification-service",
        },
        {
            "timestamp": "2024-01-16T12:28:00Z",
            "level": "ERROR",
            "message": "Queue processing rate below consumption rate",
            "service": "notification-service",
        },
        {
            "timestamp": "2024-01-16T12:29:00Z",
            "level": "WARN",
            "message": "Downstream service degradation detected, retries increasing",
            "service": "notification-service",
        },
    ],
    "scenario-008": [
        {
            "timestamp": "2024-01-16T12:42:00Z",
            "level": "WARN",
            "message": "Slow database queries detected",
            "service": "analytics-service",
            "trace_id": "trace-012",
        },
        {
            "timestamp": "2024-01-16T12:43:00Z",
            "level": "INFO",
            "message": "Query duration exceeding 2 seconds",
            "service": "analytics-service",
        },
    ],
    "scenario-009": [
        {
            "timestamp": "2024-01-16T10:47:00Z",
            "level": "ERROR",
            "message": "New error pattern after deployment v2.3.1",
            "service": "payment-service",
            "trace_id": "trace-013",
        },
        {
            "timestamp": "2024-01-16T10:49:00Z",
            "level": "ERROR",
            "message": "Error rate spike correlated with deployment",
            "service": "payment-service",
        },
    ],
    "scenario-010": [
        {
            "timestamp": "2024-01-16T13:02:00Z",
            "level": "ERROR",
            "message": "Downstream dependency auth-service unavailable",
            "service": "user-service",
            "trace_id": "trace-014",
        },
        {
            "timestamp": "2024-01-16T13:03:00Z",
            "level": "WARN",
            "message": "Circuit breaker opened for auth-service",
            "service": "user-service",
        },
    ],
}

SEEDED_TRACES_DATA = {
    "scenario-001": [
        {
            "trace_id": "trace-001",
            "span_id": "span-001",
            "service": "payment-service",
            "operation": "process_payment",
            "duration_ms": 3500,
            "status": "error",
            "error": "Connection timeout",
            "attributes": {
                "db.connection.wait_time_ms": 30000,
                "db.pool.active": 100,
            },
        },
        {
            "trace_id": "trace-002",
            "span_id": "span-002",
            "service": "payment-service",
            "operation": "process_payment",
            "duration_ms": 3200,
            "status": "error",
            "error": "Connection timeout",
        },
    ],
    "scenario-002": [
        {
            "trace_id": "trace-003",
            "span_id": "span-003",
            "service": "user-service",
            "operation": "get_user_profile",
            "duration_ms": 850,
            "status": "ok",
            "attributes": {
                "cache.hit": False,
                "cache.miss": True,
                "db.query.duration_ms": 750,
            },
        },
        {
            "trace_id": "trace-004",
            "span_id": "span-004",
            "service": "user-service",
            "operation": "get_user_profile",
            "duration_ms": 920,
            "status": "ok",
            "attributes": {
                "cache.hit": False,
                "cache.miss": True,
                "db.query.duration_ms": 810,
            },
        },
    ],
    "scenario-003": [
        {
            "trace_id": "trace-005",
            "span_id": "span-005",
            "service": "api-gateway",
            "operation": "route_request",
            "duration_ms": 1200,
            "status": "error",
            "error": "Downstream service error",
            "attributes": {
                "downstream.service": "payment-service",
                "downstream.error_rate": 0.12,
            },
        },
        {
            "trace_id": "trace-006",
            "span_id": "span-006",
            "service": "api-gateway",
            "operation": "route_request",
            "duration_ms": 1500,
            "status": "error",
            "error": "Downstream service timeout",
        },
    ],
    "scenario-004": [
        {
            "trace_id": "trace-007",
            "span_id": "span-007",
            "service": "database-primary",
            "operation": "query",
            "duration_ms": 5000,
            "status": "error",
            "error": "Connection pool exhausted",
            "attributes": {
                "db.connections.active": 100,
                "db.connections.max": 100,
                "db.connection.wait_time_ms": 45000,
            },
        },
    ],
    "scenario-005": [
        {
            "trace_id": "trace-008",
            "span_id": "span-008",
            "service": "redis-cache",
            "operation": "get",
            "duration_ms": 50,
            "status": "ok",
            "attributes": {
                "cache.memory.usage_percent": 95,
                "cache.evictions": 1200,
            },
        },
    ],
    "scenario-006": [
        {
            "trace_id": "trace-009",
            "span_id": "span-009",
            "service": "auth-service",
            "operation": "validate_token",
            "duration_ms": 200,
            "status": "error",
            "error": "Token validation failed",
            "attributes": {
                "auth.failure_rate": 0.08,
            },
        },
        {
            "trace_id": "trace-010",
            "span_id": "span-010",
            "service": "auth-service",
            "operation": "validate_token",
            "duration_ms": 180,
            "status": "error",
            "error": "Invalid token signature",
        },
    ],
    "scenario-007": [
        {
            "trace_id": "trace-011",
            "span_id": "span-011",
            "service": "notification-service",
            "operation": "send_notification",
            "duration_ms": 3000,
            "status": "ok",
            "attributes": {
                "queue.depth": 50000,
                "queue.processing_rate": 50,
                "downstream.service": "email-service",
                "downstream.latency_ms": 2500,
                "downstream.error_rate": 0.25,
            },
        },
    ],
    "scenario-008": [
        {
            "trace_id": "trace-012",
            "span_id": "span-012",
            "service": "analytics-service",
            "operation": "process_analytics",
            "duration_ms": 2500,
            "status": "ok",
            "attributes": {
                "db.query.duration_ms": 2400,
                "db.query.slow": True,
            },
        },
    ],
    "scenario-009": [
        {
            "trace_id": "trace-013",
            "span_id": "span-013",
            "service": "payment-service",
            "operation": "process_payment",
            "duration_ms": 2800,
            "status": "error",
            "error": "New error after deployment",
            "attributes": {
                "deployment.version": "v2.3.1",
                "deployment.timestamp": "2024-01-16T10:45:00Z",
            },
        },
    ],
    "scenario-010": [
        {
            "trace_id": "trace-014",
            "span_id": "span-014",
            "service": "user-service",
            "operation": "get_user_profile",
            "duration_ms": 5000,
            "status": "error",
            "error": "Downstream dependency failure",
            "attributes": {
                "downstream.service": "auth-service",
                "downstream.status": "unavailable",
                "circuit_breaker.open": True,
            },
        },
    ],
}


@mcp.tool()
async def metrics_query(
    service_id: str,
    metric_name: str,
    time_window: str = "15m",
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Query metrics for a service.
    
    Args:
        service_id: The service identifier (e.g., "payment-service")
        metric_name: Name of the metric (e.g., "error_rate", "latency_p95")
        time_window: Time window for the query (e.g., "15m", "1h")
        scenario_id: Optional scenario ID for seeded data
    
    Returns:
        Dict containing metric data with timestamps and values
    
    Metric Selection Guidance by Alert Type:
    - latency_spike alerts: Query latency_p95, cache_hit_rate, cache_miss_rate, 
      database query performance metrics (query_performance)
    - error_rate_critical alerts: Query error_rate, downstream_service_error_rate,
      authentication_failure_rate (for auth services), token_validation_errors
    - cache_miss_storm alerts: Query cache_hit_rate, cache_miss_rate, cache_eviction_rate,
      cache_memory_usage
    - queue_depth_runaway alerts: Query queue_depth, queue_processing_rate,
      downstream_service_latency_p95, downstream_service_error_rate
      (queue buildup is often caused by slow downstream services)
    - database_connection_pool alerts: Query connection_pool_usage, connection_pool_exhaustion_rate,
      database query performance metrics
    
    Common metrics: error_rate, latency_p95, cache_hit_rate, downstream_service_error_rate,
    downstream_service_latency_p95, queue_depth, queue_processing_rate
    """
    if scenario_id and scenario_id in SEEDED_METRICS_DATA:
        data = SEEDED_METRICS_DATA[scenario_id].get(metric_name, [])
        if data:
            return {
                "service_id": service_id,
                "metric_name": metric_name,
                "time_window": time_window,
                "data_points": data,
                "status": "success",
            }
    
    # Generate synthetic data if not seeded
    now = datetime.now(timezone.utc)
    data_points = []
    for i in range(4):
        timestamp = now - timedelta(minutes=15 - i * 5)
        value = random.uniform(0.1, 5.0) if "rate" in metric_name else random.uniform(100, 1000)
        data_points.append({
            "timestamp": timestamp.isoformat() + "Z",
            "value": round(value, 2),
        })
    
    return {
        "service_id": service_id,
        "metric_name": metric_name,
        "time_window": time_window,
        "data_points": data_points,
        "status": "success",
    }


@mcp.tool()
async def logs_search(
    service_id: str,
    query: str,
    time_window: str = "15m",
    limit: int = 10,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Search logs for a service.
    
    Args:
        service_id: The service identifier
        query: Search query (e.g., "error", "timeout", "connection")
        time_window: Time window for the search
        limit: Maximum number of results
        scenario_id: Optional scenario ID for seeded data
    
    Returns:
        Dict containing matching log entries
    """
    if scenario_id and scenario_id in SEEDED_LOGS_DATA:
        logs = SEEDED_LOGS_DATA[scenario_id]
        # Filter by query if provided - use word-based matching for better results
        query_lower = query.lower()
        query_words = query_lower.split()
        filtered_logs = [
            log for log in logs
            if query_lower in log.get("message", "").lower()
            or query_lower in log.get("level", "").lower()
            or any(word in log.get("message", "").lower() for word in query_words if len(word) > 3)  # Match individual words (min 4 chars)
        ]
        return {
            "service_id": service_id,
            "query": query,
            "time_window": time_window,
            "results": filtered_logs[:limit],
            "count": len(filtered_logs),
            "status": "success",
        }
    
    # Generate synthetic logs if not seeded
    synthetic_logs = []
    now = datetime.now(timezone.utc)
    for i in range(min(limit, 5)):
        synthetic_logs.append({
            "timestamp": (now - timedelta(minutes=10 - i * 2)).isoformat() + "Z",
            "level": random.choice(["ERROR", "WARN", "INFO"]),
            "message": f"Synthetic log entry for {service_id}: {query}",
            "service": service_id,
        })
    
    return {
        "service_id": service_id,
        "query": query,
        "time_window": time_window,
        "results": synthetic_logs,
        "count": len(synthetic_logs),
        "status": "success",
    }


@mcp.tool()
async def trace_query(
    service_id: str,
    operation: Optional[str] = None,
    time_window: str = "15m",
    limit: int = 5,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Query traces for a service.
    
    Args:
        service_id: The service identifier
        operation: Optional operation name to filter by
        time_window: Time window for the query
        limit: Maximum number of traces
        scenario_id: Optional scenario ID for seeded data
    
    Returns:
        Dict containing trace data
    """
    if scenario_id and scenario_id in SEEDED_TRACES_DATA:
        traces = SEEDED_TRACES_DATA[scenario_id]
        if operation:
            traces = [t for t in traces if t.get("operation") == operation]
        return {
            "service_id": service_id,
            "operation": operation,
            "time_window": time_window,
            "traces": traces[:limit],
            "count": len(traces),
            "status": "success",
        }
    
    # Generate synthetic traces if not seeded
    synthetic_traces = []
    for i in range(min(limit, 3)):
        synthetic_traces.append({
            "trace_id": f"trace-{i:03d}",
            "span_id": f"span-{i:03d}",
            "service": service_id,
            "operation": operation or "unknown",
            "duration_ms": random.randint(100, 5000),
            "status": random.choice(["ok", "error"]),
        })
    
    return {
        "service_id": service_id,
        "operation": operation,
        "time_window": time_window,
        "traces": synthetic_traces,
        "count": len(synthetic_traces),
        "status": "success",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)

