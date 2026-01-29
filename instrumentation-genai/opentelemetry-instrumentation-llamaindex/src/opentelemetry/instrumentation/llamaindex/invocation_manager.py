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

from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    LLMInvocation,
    RetrievalInvocation,
    Workflow,
    ToolCall,
)

__all__ = ["_InvocationManager"]


@dataclass
class _InvocationState:
    invocation: Union[
        LLMInvocation,
        EmbeddingInvocation,
        RetrievalInvocation,
        Workflow,
        AgentInvocation,
        ToolCall,
    ]
    children: List[str] = field(default_factory=lambda: list())


class _InvocationManager:
    """
    Manages LlamaIndex invocations and workflows with parent/child relationships.

    This replaces the entity registry pattern from TelemetryHandler, as the
    handler is dropping support for entity tracking.
    """

    def __init__(self) -> None:
        # Map from event_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: Dict[str, _InvocationState] = {}
        self._parents: Dict[str, Optional[str]] = {}

    def add_invocation_state(
        self,
        event_id: str,
        parent_id: Optional[str],
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            RetrievalInvocation,
            Workflow,
            AgentInvocation,
            ToolCall,
        ],
    ) -> None:
        """Add an invocation to the manager."""
        invocation_state = _InvocationState(invocation=invocation)
        self._invocations[event_id] = invocation_state
        self._parents[event_id] = parent_id

        if parent_id is not None and parent_id in self._invocations:
            parent_invocation_state = self._invocations[parent_id]
            parent_invocation_state.children.append(event_id)

    def get_invocation(
        self, event_id: str
    ) -> Optional[
        Union[LLMInvocation, EmbeddingInvocation, RetrievalInvocation, Workflow]
    ]:
        """Get an invocation or workflow by event_id."""
        invocation_state = self._invocations.get(event_id)
        return invocation_state.invocation if invocation_state else None

    def get_parent_id(self, event_id: str) -> Optional[str]:
        """Get the parent event_id for a given invocation."""
        return self._parents.get(event_id)

    def delete_invocation_state(self, event_id: str) -> None:
        """Delete an invocation and all its children from the manager."""
        to_visit = [event_id]
        to_delete = []
        while to_visit:
            current_id = to_visit.pop()
            invocation_state = self._invocations.get(current_id)
            if not invocation_state:
                continue
            to_delete.append(current_id)
            to_visit.extend(invocation_state.children)
        for current_id in to_delete:
            self._invocations.pop(current_id, None)
            self._parents.pop(current_id, None)
