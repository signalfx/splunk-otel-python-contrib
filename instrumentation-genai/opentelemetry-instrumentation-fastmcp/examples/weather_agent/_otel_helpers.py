"""Shared OTel/dotenv helpers for the weather_agent example scripts."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(env_file: Path | None = None) -> None:
    """Load *env_file* (defaults to `.env` next to this file) into the process environment.

    Uses ``python-dotenv`` when available; falls back to a simple
    key=value parser so the examples work without extra deps.
    """
    if env_file is None:
        env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv as _load

        _load(env_file, override=False)
    except ImportError:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = value.strip()


def providers_already_configured() -> bool:
    """Return ``True`` when ``opentelemetry-instrument`` has already set up providers.

    Detects whether the current process already has a real ``TracerProvider``
    registered (as opposed to the default ``ProxyTracerProvider`` or
    ``NoOpTracerProvider``).  Used to avoid double-registering providers when
    running under ``opentelemetry-instrument``.
    """
    from opentelemetry import trace

    name = type(trace.get_tracer_provider()).__name__
    return "Proxy" not in name and "NoOp" not in name
