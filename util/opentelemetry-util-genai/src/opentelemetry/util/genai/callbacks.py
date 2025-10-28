from __future__ import annotations

from typing import Protocol

from .types import GenAI


class CompletionCallback(Protocol):
    """Protocol implemented by handlers interested in completion events."""

    def on_completion(self, invocation: GenAI) -> None:
        """Handle completion of a GenAI invocation."""


class NoOpCompletionCallback:
    """Completion callback that performs no work."""

    def on_completion(
        self, invocation: GenAI
    ) -> None:  # pragma: no cover - trivial
        return None


__all__ = ["CompletionCallback", "NoOpCompletionCallback"]
