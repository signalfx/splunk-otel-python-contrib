import time
from typing import Dict, List, Optional
from core.api_client import APIClient
from core.logger import get_logger


logger = get_logger(__name__)


class APMClient(APIClient):
    """Client for Splunk APM API operations."""
    
    def __init__(self, realm: str, access_token: str):
        """
        Initialize APM client.
        
        Args:
            realm: Splunk realm (rc0, us1, lab0)
            access_token: Splunk access token
        """
        base_url = f"https://api.{realm}.signalfx.com"
        super().__init__(base_url, access_token)
        self.realm = realm
        logger.info(f"APMClient initialized for realm: {realm}")
    
    def get_trace(self, trace_id: str, max_wait: int = 120) -> Dict:
        """
        Get trace by ID with automatic retry for availability.
        
        Args:
            trace_id: Trace ID
            max_wait: Maximum wait time in seconds
        
        Returns:
            Trace data dictionary
        
        Raises:
            TimeoutError: If trace not available after max_wait
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                trace = self.get(f"/v2/apm/traces/{trace_id}")
                logger.info(
                    f"Trace {trace_id} retrieved",
                    span_count=len(trace.get('spans', []))
                )
                return trace
                
            except Exception as e:
                if "404" in str(e):
                    elapsed = int(time.time() - start_time)
                    logger.debug(
                        f"Trace not available yet, waiting... ({elapsed}s/{max_wait}s)"
                    )
                    time.sleep(5)
                else:
                    raise
        
        raise TimeoutError(
            f"Trace {trace_id} not available after {max_wait}s. "
            f"Check if telemetry is being sent correctly."
        )
    
    def query_traces(
        self,
        filters: Dict,
        time_range: str = "1h",
        limit: int = 100
    ) -> List[Dict]:
        """
        Query traces with filters.
        
        Args:
            filters: Filter dictionary (e.g., {'sf_environment': 'rc0'})
            time_range: Time range string (e.g., '1h', '24h')
            limit: Maximum results
        
        Returns:
            List of trace dictionaries
        """
        payload = {
            "filters": filters,
            "timeRange": time_range,
            "limit": limit
        }
        
        response = self.post("/v2/apm/traces/search", json=payload)
        traces = response.get("traces", [])
        
        logger.info(
            f"Query returned {len(traces)} traces",
            filters=filters,
            time_range=time_range
        )
        return traces
    
    def get_session(self, session_id: str) -> Dict:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session data dictionary
        """
        session = self.get(f"/v2/apm/sessions/{session_id}")
        logger.info(f"Session {session_id} retrieved")
        return session
    
    def query_agents(
        self, 
        environment: str, 
        time_range: str = "1h"
    ) -> List[Dict]:
        """
        Query AI agents in environment.
        
        Args:
            environment: Deployment environment
            time_range: Time range for metrics
        
        Returns:
            List of agent dictionaries
        """
        params = {
            "environment": environment,
            "timeRange": time_range
        }
        
        response = self.get("/v2/apm/agents", params=params)
        agents = response.get("agents", [])
        
        logger.info(
            f"Found {len(agents)} agents in {environment}",
            environment=environment
        )
        return agents
    
    def get_span(self, span_id: str) -> Dict:
        """
        Get span by ID.
        
        Args:
            span_id: Span ID
        
        Returns:
            Span data dictionary
        """
        span = self.get(f"/v2/apm/spans/{span_id}")
        logger.info(f"Span {span_id} retrieved")
        return span
    
    def query_metrics(
        self,
        metric_name: str,
        filters: Optional[Dict] = None,
        time_range: str = "1h"
    ) -> List[Dict]:
        """
        Query metrics by name.
        
        Args:
            metric_name: Metric name (e.g., 'gen_ai.client.token.usage')
            filters: Optional filters
            time_range: Time range
        
        Returns:
            List of metric data points
        """
        payload = {
            "metric": metric_name,
            "filters": filters or {},
            "timeRange": time_range
        }
        
        response = self.post("/v2/apm/metrics/query", json=payload)
        metrics = response.get("data", [])
        
        logger.info(
            f"Query returned {len(metrics)} data points for {metric_name}"
        )
        return metrics
