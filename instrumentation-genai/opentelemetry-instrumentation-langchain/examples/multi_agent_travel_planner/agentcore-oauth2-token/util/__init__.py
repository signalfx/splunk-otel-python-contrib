"""Utility modules for AgentCore examples."""

from util.oauth2_token_manager import OAuth2TokenManager

# Backward compatibility alias
CiscoTokenManager = OAuth2TokenManager

__all__ = ["OAuth2TokenManager", "CiscoTokenManager"]
