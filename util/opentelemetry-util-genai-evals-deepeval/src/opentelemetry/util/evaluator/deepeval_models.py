# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

"""Plugin registry for Deepeval evaluation models.

This module exposes a lightweight registry that allows additional packages to
register :class:`deepeval.models.base_model.DeepEvalBaseLLM` factories. The
registry is populated from two sources:

* Direct calls to :func:`register_model` (used internally and by tests)
* Python entry points declared under the group
  ``opentelemetry_util_genai_evals.deepeval_models``

Each entry point name becomes the lookup key. The value must be a callable that
returns a ``DeepEvalBaseLLM`` instance when invoked with no arguments. The
registry caches instantiated models to avoid repeated token provisioning while
ensuring a fresh model can be created on demand when required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import metadata
from threading import RLock
from typing import Any

try:  # Deepeval is an optional dependency; only import when available.
    from deepeval.models.base_model import DeepEvalBaseLLM  # type: ignore
except Exception:  # pragma: no cover - dependency missing during tests
    DeepEvalBaseLLM = type("DeepEvalBaseLLM", (), {})  # type: ignore[misc]


_LOGGER = logging.getLogger(__name__)

_ENTRYPOINT_GROUP = "opentelemetry_util_genai_evals.deepeval_models"

_MODEL_FACTORIES: dict[str, Callable[[], DeepEvalBaseLLM | None]] = {}
_MODEL_CACHE: dict[str, DeepEvalBaseLLM] = {}
_ENTRYPOINTS_LOADED = False
_LOCK = RLock()


def _normalized(name: str) -> str:
    return name.strip().lower()


def register_model(
    name: str, factory: Callable[[], DeepEvalBaseLLM | None]
) -> None:
    """Register a Deepeval evaluation model factory.

    Parameters
    ----------
    name:
        The key that end users reference via ``DEEPEVAL_MODEL`` or metric-level
        ``model`` overrides. Normalized to lowercase.
    factory:
        Zero-argument callable returning a ``DeepEvalBaseLLM`` instance (or
        ``None`` to indicate registration failure).
    """

    if not name:
        raise ValueError("Model name must be a non-empty string")
    if not callable(factory):
        raise TypeError("Model factory must be callable")
    key = _normalized(name)
    with _LOCK:
        _MODEL_FACTORIES[key] = factory
        _MODEL_CACHE.pop(key, None)


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    with _LOCK:
        if _ENTRYPOINTS_LOADED:
            return
        eps: list[Any]
        try:
            eps_obj = metadata.entry_points()
            if hasattr(eps_obj, "select"):
                eps = list(eps_obj.select(group=_ENTRYPOINT_GROUP))  # type: ignore[arg-type]
            else:  # pragma: no cover - legacy structure
                eps = list(eps_obj.get(_ENTRYPOINT_GROUP, []))  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to enumerate Deepeval model entry points: %s", exc
            )
            eps = []
        for ep in eps:
            try:
                obj = ep.load()
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Skipping Deepeval model entry point '%s': %s",
                    ep.name,
                    exc,
                )
                continue
            if not callable(obj):
                _LOGGER.warning(
                    "Entry point '%s' does not provide a callable factory",
                    ep.name,
                )
                continue
            try:
                register_model(ep.name, obj)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Failed to register Deepeval model '%s': %s", ep.name, exc
                )
        _ENTRYPOINTS_LOADED = True


def resolve_model(name: str) -> DeepEvalBaseLLM | None:
    """Return an instance of the registered model if available."""

    if not name:
        return None
    _load_entrypoints()
    key = _normalized(name)
    factory: Callable[[], DeepEvalBaseLLM | None] | None
    with _LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        factory = _MODEL_FACTORIES.get(key)
    if factory is None:
        return None
    try:
        instance = factory()
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning(
            "Model factory for '%s' raised an exception: %s", key, exc
        )
        return None
    if instance is None:
        return None
    if not isinstance(instance, DeepEvalBaseLLM):
        _LOGGER.warning(
            "Model factory for '%s' did not return a DeepEvalBaseLLM instance",
            key,
        )
        return None
    with _LOCK:
        _MODEL_CACHE[key] = instance
    return instance


def list_models() -> list[str]:
    """Return the list of currently registered model keys."""

    _load_entrypoints()
    with _LOCK:
        return sorted(_MODEL_FACTORIES)


def clear_models() -> None:
    """Reset the registry (used primarily in tests)."""

    global _ENTRYPOINTS_LOADED
    with _LOCK:
        _MODEL_FACTORIES.clear()
        _MODEL_CACHE.clear()
        _ENTRYPOINTS_LOADED = False


__all__ = [
    "register_model",
    "resolve_model",
    "list_models",
    "clear_models",
]
