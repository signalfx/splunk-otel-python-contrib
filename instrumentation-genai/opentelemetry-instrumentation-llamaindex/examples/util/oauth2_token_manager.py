"""OAuth2 Token Manager for CircuIT.

Handles OAuth2 token lifecycle with file-based caching and automatic refresh.
"""

import base64
import json
import os
from datetime import datetime, timedelta

import requests


class OAuth2TokenManager:
    """Manages OAuth2 token lifecycle for CircuIT.

    Handles token retrieval, file-based caching, and automatic refresh with a 5-minute buffer.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str = None,
        cache_file: str = "/tmp/circuit_token_cache.json",
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.cache_file = cache_file

    def _get_cached_token(self):
        """Get token from cache file if valid."""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() < expires_at - timedelta(minutes=5):
                return cache_data["access_token"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def _fetch_new_token(self):
        """Request a new access token from the OAuth2 server."""
        # Create Basic Auth header
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope

        response = requests.post(self.token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        # Cache token to file with secure permissions
        cache_data = {
            "access_token": token_data["access_token"],
            "expires_at": expires_at.isoformat(),
        }

        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        os.chmod(self.cache_file, 0o600)  # rw------- (owner only)

        return token_data["access_token"]

    def get_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        token = self._get_cached_token()
        if token:
            return token
        return self._fetch_new_token()

    def cleanup_token_cache(self):
        """Securely remove token cache file."""
        if os.path.exists(self.cache_file):
            # Overwrite file with zeros before deletion for security
            with open(self.cache_file, "r+b") as f:
                length = f.seek(0, 2)  # Get file size
                f.seek(0)
                f.write(b"\0" * length)  # Overwrite with zeros
            os.remove(self.cache_file)
