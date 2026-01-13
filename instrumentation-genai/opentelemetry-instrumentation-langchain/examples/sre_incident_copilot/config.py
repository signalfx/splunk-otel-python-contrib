"""Configuration management for SRE Incident Copilot."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""

    # OpenAI / LLM
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None

    # OAuth2 for Cisco (optional)
    oauth_token_url: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    oauth_app_key: Optional[str] = None

    # OpenTelemetry
    otel_service_name: str = "sre-incident-copilot"
    otel_exporter_otlp_endpoint: Optional[str] = None
    otel_exporter_otlp_protocol: str = "grpc"

    # Data paths
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    # Scenario
    scenario_id: Optional[str] = None

    # Drift simulation
    drift_enabled: bool = False
    drift_mode: Optional[str] = None
    drift_intensity: float = 0.0

    # Eval thresholds
    confidence_threshold: float = 0.7
    evidence_count_threshold: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            oauth_token_url=os.getenv("OAUTH2_TOKEN_URL"),
            oauth_client_id=os.getenv("OAUTH2_CLIENT_ID"),
            oauth_client_secret=os.getenv("OAUTH2_CLIENT_SECRET"),
            oauth_app_key=os.getenv("OAUTH2_APP_KEY"),
            otel_service_name=os.getenv("OTEL_SERVICE_NAME", "sre-incident-copilot"),
            otel_exporter_otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otel_exporter_otlp_protocol=os.getenv(
                "OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"
            ),
            data_dir=os.getenv("DATA_DIR", "data"),
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
            scenario_id=os.getenv("SCENARIO_ID"),
            drift_enabled=os.getenv("DRIFT_ENABLED", "false").lower() == "true",
            drift_mode=os.getenv("DRIFT_MODE"),
            drift_intensity=float(os.getenv("DRIFT_INTENSITY", "0.0")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
            evidence_count_threshold=int(os.getenv("EVIDENCE_COUNT_THRESHOLD", "3")),
        )
