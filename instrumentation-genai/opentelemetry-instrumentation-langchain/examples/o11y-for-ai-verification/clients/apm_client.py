import time
import requests
from typing import Dict, List, Optional
from core.api_client import APIClient
from core.logger import get_logger


logger = get_logger(__name__)


class APMClient(APIClient):
    """Client for Splunk APM API operations using session-based authentication."""

    def __init__(
        self,
        realm: str,
        email: str = None,
        password: str = None,
        session_token: str = None,
        target_org: str = None,
    ):
        """
        Initialize APM client with session-based authentication.

        Args:
            realm: Splunk realm (rc0, us1, lab0)
            email: User email for session login (optional if session_token provided)
            password: User password for session login (optional if session_token provided)
            session_token: Pre-existing session token (optional)
            target_org: Target organization name to switch to (e.g., 'qaregression')
        """
        # Use app URL for session-based API
        base_url = f"https://app.{realm}.signalfx.com"

        # Get or create session token
        if session_token:
            self.session_token = session_token
        elif email and password:
            self.session_token = self._create_session(base_url, email, password)
        else:
            raise ValueError("Must provide either session_token or email+password")

        super().__init__(base_url, self.session_token)
        self.realm = realm
        self.current_org = None
        
        # Switch to target org if specified
        if target_org:
            self._switch_to_org(target_org)
        
        logger.info(f"APMClient initialized for realm: {realm}, org: {self.current_org or 'default'}")

    def _create_session(self, base_url: str, email: str, password: str) -> str:
        """
        Create session and return session token.

        Args:
            base_url: Base URL for session endpoint
            email: User email
            password: User password

        Returns:
            Session access token
        """
        session_url = f"{base_url}/v2/session"
        payload = {"email": email, "password": password}

        logger.info(f"Creating session for {email}")
        response = requests.post(session_url, json=payload, timeout=30)
        response.raise_for_status()

        session_data = response.json()
        access_token = session_data.get("accessToken")

        if not access_token:
            raise ValueError("No accessToken in session response")

        logger.info(f"Session created successfully, token: {access_token[:10]}...")
        return access_token

    def _get_organizations(self) -> List[Dict]:
        """
        Get list of organizations the user has access to.

        Returns:
            List of organization dictionaries
        """
        try:
            response = self.get("/v2/user/organizations")
            orgs = response.get("organizations", [])
            logger.info(f"Found {len(orgs)} organizations")
            return orgs
        except Exception as e:
            logger.error(f"Failed to get organizations: {e}")
            return []

    def _switch_to_org(self, org_name: str) -> bool:
        """
        Switch session to a specific organization by name.

        Args:
            org_name: Organization name to switch to (e.g., 'qaregression')

        Returns:
            True if switch successful, False otherwise
        """
        # Get list of organizations
        orgs = self._get_organizations()
        
        # Find org by name (case-insensitive)
        target_org = None
        for org in orgs:
            if org.get("organizationName", "").lower() == org_name.lower():
                target_org = org
                break
        
        if not target_org:
            available_orgs = [o.get("organizationName") for o in orgs]
            logger.error(f"Organization '{org_name}' not found. Available: {available_orgs}")
            raise ValueError(f"Organization '{org_name}' not found. Available: {available_orgs}")
        
        org_id = target_org.get("id")
        logger.info(f"Switching to org: {org_name} (id: {org_id})")
        
        # Switch session to the target org
        try:
            switch_url = f"/v2/session/{org_id}"
            response = self.post(switch_url, json={})
            
            # Update token if new one is returned
            new_token = response.get("accessToken")
            if new_token:
                self.session_token = new_token
                self.access_token = new_token
                self.session.headers["x-sf-token"] = new_token
                logger.info(f"Session switched, new token: {new_token[:10]}...")
            
            self.current_org = org_name
            logger.info(f"Successfully switched to org: {org_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to org {org_name}: {e}")
            raise

    def get_trace(self, trace_id: str, max_wait: int = 120) -> Dict:
        """
        Get trace by ID using direct GraphQL lookup.

        Args:
            trace_id: Trace ID
            max_wait: Maximum wait time in seconds (for retries)

        Returns:
            Trace data dictionary with spans

        Raises:
            TimeoutError: If trace not available after max_wait
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Try direct trace lookup first (faster and more reliable)
                trace = self._get_trace_by_id(trace_id)

                if trace:
                    logger.info(
                        f"Trace {trace_id} retrieved via direct lookup",
                        service=trace.get("initiatingService"),
                    )
                    return trace

                # Trace not found yet
                elapsed = int(time.time() - start_time)
                logger.debug(
                    f"Trace not available yet, waiting... ({elapsed}s/{max_wait}s)"
                )
                time.sleep(5)

            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
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

    def _get_trace_by_id(self, trace_id: str) -> Dict:
        """
        Get trace directly by ID using TraceFullDetailsLessValidation query.
        This is faster than searching and works for any trace regardless of age.

        Args:
            trace_id: Trace ID

        Returns:
            Trace summary dict (compatible with legacyTraceExamples format)
        """
        graphql_url = "/v2/apm/graphql?op=TraceFullDetailsLessValidation"

        # Use the same query structure as the UI for better compatibility
        query = """
        query TraceFullDetailsLessValidation($id: ID!, $spanLimit: Float = 5000, $returnPartialTrace: Boolean = false) {
          trace: traceLessValidation(
            id: $id
            spanLimit: $spanLimit
            returnPartialTrace: $returnPartialTrace
          ) {
            traceID
            rootOperation
            duration
            startTime
            spans
            isAITrace
            state
          }
        }
        """

        payload = {
            "operationName": "TraceFullDetailsLessValidation",
            "query": query,
            "variables": {
                "id": trace_id,
                "spanLimit": 1000,
                "returnPartialTrace": True,
            },
        }

        try:
            response = self.post(graphql_url, json=payload)
            trace_data = response.get("data", {}).get("trace")

            if not trace_data:
                return None

            # Convert to format compatible with tests
            # Extract service info from spans and normalize span format
            spans = trace_data.get("spans", [])
            services = {}
            normalized_spans = []

            for span in spans:
                svc = span.get("serviceName")
                if svc:
                    if svc not in services:
                        services[svc] = {"service": svc, "spanCount": 0, "errors": []}
                    services[svc]["spanCount"] += 1

                # Convert tags array to attributes dict for test compatibility
                attributes = {}
                for tag in span.get("tags", []):
                    key = tag.get("key")
                    value = tag.get("value")
                    if key and value is not None:
                        attributes[key] = value

                # Normalize span format with both camelCase and snake_case fields
                normalized_span = {
                    "spanId": span.get("spanID"),  # snake_case for tests
                    "spanID": span.get("spanID"),  # camelCase from API
                    "span_id": span.get("spanID"),  # alternative snake_case
                    "traceId": span.get("traceID"),
                    "serviceName": span.get("serviceName"),
                    "operationName": span.get("operationName"),
                    "startTime": span.get("startTime"),
                    "start_time": span.get("startTime"),  # snake_case alias
                    "duration": span.get("duration"),
                    "attributes": attributes,  # Flattened attributes dict
                    "tags": span.get("tags", []),  # Keep original tags array
                    "references": span.get("references", []),
                    "logs": span.get("logs", []),
                }
                normalized_spans.append(normalized_span)

            # Return format that includes both summary fields and full span data
            return {
                "traceId": trace_data.get("traceID"),
                "initiatingService": spans[0].get("serviceName") if spans else None,
                "initiatingOperation": trace_data.get("rootOperation"),
                "durationMicros": trace_data.get("duration"),
                "startTimeMicros": trace_data.get("startTime"),
                "serviceSpanCounts": list(services.values()),
                "isAITrace": trace_data.get("isAITrace", False),
                "state": trace_data.get("state"),
                # Include normalized spans array for test compatibility
                "spans": normalized_spans,
            }

        except Exception as e:
            logger.debug(f"Error getting trace by ID: {e}")
            return None

    def _search_traces_graphql(
        self, trace_ids: List[str] = None, time_range_minutes: int = 480
    ) -> List[Dict]:
        """
        Search traces using GraphQL API (matches UI behavior).

        Args:
            trace_ids: Optional list of specific trace IDs to search
            time_range_minutes: Time range in minutes to search

        Returns:
            List of trace dictionaries
        """
        end_time_ms = int(time.time() * 1000)
        start_time_ms = end_time_ms - (time_range_minutes * 60 * 1000)

        logger.info(
            f"Searching for traces via GraphQL (time range: {time_range_minutes}m)"
        )

        # Step 1: Create analytics search job
        job_id = self._create_trace_search_job(start_time_ms, end_time_ms, trace_ids)

        # Step 2: Poll until job completes
        max_polls = 30
        poll_interval = 2

        for attempt in range(max_polls):
            result = self._get_search_results(job_id)

            if result and result.get("sections"):
                # Check if trace examples section has data
                for section in result["sections"]:
                    if section.get("sectionType") == "traceExamples":
                        traces = section.get("legacyTraceExamples", [])
                        # Return traces if we have any, even if not complete
                        # (for specific trace ID searches, we get results immediately)
                        if traces:
                            logger.info(f"Found {len(traces)} traces from GraphQL")
                            return traces
                        # If complete but no traces, stop polling
                        if section.get("isComplete"):
                            logger.info("Search complete but no traces found")
                            return []

            time.sleep(poll_interval)

        logger.warning(f"GraphQL search job {job_id} did not complete in time")
        return []

    def _create_trace_search_job(
        self, start_time_ms: int, end_time_ms: int, trace_ids: List[str] = None
    ) -> str:
        """
        Create analytics search job via GraphQL (matches actual UI API).

        Args:
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            trace_ids: Optional specific trace IDs to search

        Returns:
            Job ID string
        """
        graphql_url = "/v2/apm/graphql?op=StartAnalyticsSearch"

        # Build trace filter with optional trace ID filter
        trace_filter_tags = []
        if trace_ids:
            # Filter by specific trace IDs
            trace_filter_tags.append(
                {"tag": "sf_traceId", "operation": "IN", "values": trace_ids}
            )
        else:
            # Match all traces with gen_ai operations
            trace_filter_tags.append(
                {"tag": "gen_ai.operation.name", "operation": "IN", "values": ["*"]}
            )

        # GraphQL query matching actual UI structure
        query = """
        query StartAnalyticsSearch($parameters: JSON!) {
          startAnalyticsSearch(parameters: $parameters)
        }
        """

        # Parameters structure matching actual API
        parameters = {
            "sharedParameters": {
                "timeRangeMillis": {"gte": start_time_ms, "lte": end_time_ms},
                "filters": [
                    {
                        "traceFilter": {"tags": trace_filter_tags},
                        "filterType": "traceFilter",
                    }
                ],
                "samplingFactor": 100,
            },
            "sectionsParameters": [
                {"sectionType": "traceExamples", "limit": 1000},
                {"sectionType": "traceCountTimeBucketed"},
            ],
        }

        payload = {"query": query, "variables": {"parameters": parameters}}

        try:
            response = self.post(graphql_url, json=payload)
            # StartAnalyticsSearch returns full result object with jobId and sections
            result = response.get("data", {}).get("startAnalyticsSearch")

            if not result or not isinstance(result, dict):
                raise ValueError(f"Invalid response structure: {response}")

            job_id = result.get("jobId")
            if not job_id:
                raise ValueError(f"No jobId in response: {response}")

            logger.info(f"Created search job: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to create search job: {e}")
            raise

    def _get_search_results(self, job_id: str) -> Dict:
        """
        Get search results for a job (matches actual UI API).

        Args:
            job_id: Job ID from create operation

        Returns:
            Search results dictionary with sections
        """
        graphql_url = "/v2/apm/graphql?op=GetAnalyticsSearch"

        # Query structure matching actual UI
        query = """
        query GetAnalyticsSearch($jobId: ID!) {
          getAnalyticsSearch(jobId: $jobId)
        }
        """

        payload = {
            "operationName": "GetAnalyticsSearch",
            "query": query,
            "variables": {"jobId": job_id},
        }

        try:
            response = self.post(graphql_url, json=payload)

            # The response structure is: {data: {getAnalyticsSearch: {...}}}
            # where getAnalyticsSearch contains the actual result object
            result = response.get("data", {}).get("getAnalyticsSearch")

            # Result should be a dict with jobId and sections
            if result and isinstance(result, dict):
                return result

            logger.debug(f"Unexpected result structure: {result}")
            return None

        except Exception as e:
            logger.debug(f"Error getting search results: {e}")
            return None

    def query_traces(
        self, filters: Dict, time_range: str = "1h", limit: int = 100
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
        payload = {"filters": filters, "timeRange": time_range, "limit": limit}

        response = self.post("/v2/apm/traces/search", json=payload)
        traces = response.get("traces", [])

        logger.info(
            f"Query returned {len(traces)} traces",
            filters=filters,
            time_range=time_range,
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

    def query_agents(self, environment: str, time_range: str = "1h") -> List[Dict]:
        """
        Query AI agents in environment.

        Args:
            environment: Deployment environment
            time_range: Time range for metrics

        Returns:
            List of agent dictionaries
        """
        params = {"environment": environment, "timeRange": time_range}

        response = self.get("/v2/apm/agents", params=params)
        agents = response.get("agents", [])

        logger.info(
            f"Found {len(agents)} agents in {environment}", environment=environment
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
        self, metric_name: str, filters: Optional[Dict] = None, time_range: str = "1h"
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
            "timeRange": time_range,
        }

        response = self.post("/v2/apm/metrics/query", json=payload)
        metrics = response.get("data", [])

        logger.info(f"Query returned {len(metrics)} data points for {metric_name}")
        return metrics
