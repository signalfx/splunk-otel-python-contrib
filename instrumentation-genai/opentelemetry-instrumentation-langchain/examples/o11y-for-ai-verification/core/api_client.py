import requests
from typing import Dict, Any, Optional
from core.logger import get_logger
from core.retry_handler import retry_with_backoff


logger = get_logger(__name__)


class APIClient:
    """Generic API client for all HTTP operations."""

    def __init__(self, base_url: str, access_token: str, timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: Base URL for API (e.g., https://api.rc0.signalfx.com)
            access_token: Authentication token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-sf-token": access_token,  # Lowercase for GraphQL API compatibility
                "Content-Type": "application/json",
                "User-Agent": "O11y-AI-Test-Framework/2.0",
            }
        )
        logger.info(f"APIClient initialized for {base_url}")

    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        GET request with retry logic.

        Args:
            endpoint: API endpoint (e.g., '/v2/apm/traces/12345')
            params: Query parameters

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"GET {url}", params=params)

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            logger.debug(f"GET {url} - Status: {response.status_code}")
            return response.json()

        except requests.exceptions.RequestException:
            logger.error(f"GET request failed: {url}", exc_info=True)
            raise

    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def post(
        self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        POST request with retry logic.

        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON payload

        Returns:
            Response JSON as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"POST {url}", json_data=json, form_data=data)

        try:
            response = self.session.post(
                url, data=data, json=json, timeout=self.timeout
            )
            response.raise_for_status()

            logger.debug(f"POST {url} - Status: {response.status_code}")
            return response.json()

        except requests.exceptions.RequestException:
            logger.error(f"POST request failed: {url}", exc_info=True)
            raise

    def put(
        self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """PUT request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"PUT {url}")

        try:
            response = self.session.put(url, data=data, json=json, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            logger.error(f"PUT request failed: {url}", exc_info=True)
            raise

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"DELETE {url}")

        try:
            response = self.session.delete(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException:
            logger.error(f"DELETE request failed: {url}", exc_info=True)
            raise

    def close(self):
        """Close session and cleanup resources."""
        self.session.close()
        logger.info("APIClient session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
