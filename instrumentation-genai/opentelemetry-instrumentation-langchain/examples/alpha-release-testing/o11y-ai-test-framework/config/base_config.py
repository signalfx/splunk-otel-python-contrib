import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class BaseConfig:
    """Base configuration class with environment override support."""
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize configuration for specified environment.
        
        Args:
            environment: Environment name (rc0, us1, lab0). 
                        Defaults to TEST_ENVIRONMENT env var.
        """
        self.environment = environment or os.getenv("TEST_ENVIRONMENT", "rc0")
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration from YAML."""
        config_file = Path(__file__).parent / "environments" / f"{self.environment}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                f"Available environments: rc0, us1, lab0"
            )
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        if os.getenv('SPLUNK_ACCESS_TOKEN'):
            config['splunk']['access_token'] = os.getenv('SPLUNK_ACCESS_TOKEN')
        
        if os.getenv('SPLUNK_HEC_TOKEN'):
            config['splunk']['hec_token'] = os.getenv('SPLUNK_HEC_TOKEN')
        
        if os.getenv('FOUNDATION_APP_URL'):
            config['applications']['foundation']['url'] = os.getenv('FOUNDATION_APP_URL')
        
        return config
    
    def _validate_config(self):
        """Validate required configuration keys."""
        required_keys = [
            'splunk.realm',
            'splunk.api_base_url',
            'applications.foundation.url'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Missing required configuration: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (e.g., 'splunk.realm')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
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
        return self.get('splunk.realm')
    
    @property
    def splunk_access_token(self) -> str:
        """Get Splunk access token (from env var or config)."""
        return os.getenv('SPLUNK_ACCESS_TOKEN') or self.get('splunk.access_token')
    
    @property
    def apm_base_url(self) -> str:
        """Get APM base URL."""
        return f"https://app.{self.splunk_realm}.signalfx.com"
    
    @property
    def api_base_url(self) -> str:
        """Get API base URL."""
        return f"https://api.{self.splunk_realm}.signalfx.com"
    
    @property
    def foundation_app_url(self) -> str:
        """Get Foundation app URL."""
        return self.get('applications.foundation.url')
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BaseConfig(environment='{self.environment}', realm='{self.splunk_realm}')"
