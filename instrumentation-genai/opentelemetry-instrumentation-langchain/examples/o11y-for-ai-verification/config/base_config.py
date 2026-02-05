"""
Base configuration for O11y AI Verification tests.

Provides environment-based configuration loading from environment variables.
"""

import os
from typing import Dict, Any, Optional


class BaseConfig:
    """Configuration loader for test environments."""

    def __init__(self, environment: str = "rc0"):
        """
        Initialize configuration for the specified environment.

        Args:
            environment: Environment name (rc0, us1, prod)
        """
        self.environment = environment
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "splunk": {
                "realm": os.getenv("SPLUNK_REALM", "rc0"),
                "access_token": os.getenv("SPLUNK_ACCESS_TOKEN", ""),
                "api_url": os.getenv("SPLUNK_API_URL", f"https://api.{os.getenv('SPLUNK_REALM', 'rc0')}.signalfx.com"),
                "ingest_url": os.getenv("SPLUNK_INGEST_URL", f"https://ingest.{os.getenv('SPLUNK_REALM', 'rc0')}.signalfx.com"),
            },
            "otel": {
                "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                "service_name": os.getenv("OTEL_SERVICE_NAME", "alpha-test-unified-app"),
            },
            "environment": self.environment,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path (e.g., 'splunk.realm')."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    @property
    def splunk_realm(self) -> str:
        """Get Splunk realm."""
        return self.config["splunk"]["realm"]

    @property
    def splunk_access_token(self) -> str:
        """Get Splunk access token."""
        return self.config["splunk"]["access_token"]

    @property
    def splunk_api_url(self) -> str:
        """Get Splunk API URL."""
        return self.config["splunk"]["api_url"]

    @property
    def otel_endpoint(self) -> str:
        """Get OTEL endpoint."""
        return self.config["otel"]["endpoint"]

    @property
    def service_name(self) -> str:
        """Get service name."""
        return self.config["otel"]["service_name"]
