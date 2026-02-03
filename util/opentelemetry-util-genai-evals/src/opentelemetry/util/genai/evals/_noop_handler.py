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

"""No-op TelemetryHandler for the evaluator child process."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from opentelemetry.util.genai.types import EvaluationResult, GenAI


class NoOpTelemetryHandler:
    """A no-op TelemetryHandler for the evaluator child process.

    This stub is used in the child process where we don't want to
    emit any telemetry. The child process only runs evaluations and
    sends results back to the parent via IPC.
    """

    def evaluation_results(
        self, obj: GenAI, results: Sequence[EvaluationResult]
    ) -> None:
        """No-op: results are sent back via IPC, not emitted here."""
        pass


__all__ = ["NoOpTelemetryHandler"]
