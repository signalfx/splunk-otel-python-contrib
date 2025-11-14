#!/usr/bin/env python3
"""Helper to fetch a CircuIT OAuth token for LiteLLM."""

import argparse
import base64
import json
import os
import sys
from typing import Tuple

import requests

DEFAULT_TOKEN_URL = "https://id.cisco.com/oauth2/default/v1/token"


def fetch_token(client_id: str, client_secret: str, token_url: str) -> Tuple[str, int]:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    auth_b64 = base64.b64encode(raw).decode("utf-8")
    payload = "grant_type=client_credentials"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {auth_b64}",
    }
    response = requests.post(token_url, headers=headers, data=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["access_token"], int(data.get("expires_in", 3600))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a CircuIT OAuth token")
    parser.add_argument("--client-id", default=os.getenv("CISCO_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("CISCO_CLIENT_SECRET"))
    parser.add_argument("--token-url", default=os.getenv("CISCO_TOKEN_URL", DEFAULT_TOKEN_URL))
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        parser.error("CISCO_CLIENT_ID and CISCO_CLIENT_SECRET must be provided via env or flags")

    token, expires_in = fetch_token(args.client_id, args.client_secret, args.token_url)
    if args.json:
        json.dump({"access_token": token, "expires_in": expires_in}, sys.stdout)
        sys.stdout.write("\n")
        return
    print(token)


if __name__ == "__main__":
    main()
