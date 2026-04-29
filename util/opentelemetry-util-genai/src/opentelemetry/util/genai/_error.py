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

"""Error and evaluation result types for GenAI telemetry.

Extracted from types.py to avoid circular imports between
_invocation.py and types.py.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type


class ErrorClassification(Enum):
    """Classification of an error for status reporting."""

    REAL_ERROR = "error"
    INTERRUPT = "interrupted"
    CANCELLATION = "cancelled"


@dataclass
class Error:
    message: str
    type: Type[BaseException]
    classification: ErrorClassification = ErrorClassification.REAL_ERROR


@dataclass
class EvaluationResult:
    """Represents the outcome of a single evaluation metric.

    Additional fields (e.g., judge model, threshold) can be added without
    breaking callers that rely only on the current contract.
    """

    metric_name: str
    score: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[Error] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    duration_s: Optional[float] = None
    evaluator_name: Optional[str] = None
    evaluation_cost: Optional[float] = None
