# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from opentelemetry.util.genai.types import EmbeddingInvocation, LLMInvocation

__all__ = ["_InvocationManager"]


@dataclass
class _InvocationState:
    invocation: Union[LLMInvocation, EmbeddingInvocation]
    children: List[str] = field(default_factory=lambda: list())


class _InvocationManager:
    """
    Manages LlamaIndex invocations and their parent/child relationships.

    This replaces the entity registry pattern from TelemetryHandler, as the
    handler is dropping support for entity tracking.
    """

    def __init__(self) -> None:
        # Map from event_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: Dict[str, _InvocationState] = {}

    def add_invocation_state(
        self,
        event_id: str,
        parent_id: Optional[str],
        invocation: Union[LLMInvocation, EmbeddingInvocation],
    ) -> None:
        """Add an invocation to the manager."""
        invocation_state = _InvocationState(invocation=invocation)
        self._invocations[event_id] = invocation_state

        if parent_id is not None and parent_id in self._invocations:
            parent_invocation_state = self._invocations[parent_id]
            parent_invocation_state.children.append(event_id)

    def get_invocation(
        self, event_id: str
    ) -> Optional[Union[LLMInvocation, EmbeddingInvocation]]:
        """Get an invocation by event_id."""
        invocation_state = self._invocations.get(event_id)
        return invocation_state.invocation if invocation_state else None

    def delete_invocation_state(self, event_id: str) -> None:
        """Delete an invocation and all its children from the manager."""
        invocation_state = self._invocations.get(event_id)
        if not invocation_state:
            return
        for child_id in list(invocation_state.children):
            self._invocations.pop(child_id, None)
        self._invocations.pop(event_id, None)
