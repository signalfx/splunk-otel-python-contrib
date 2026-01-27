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

"""Evaluation manager and registry for :mod:`opentelemetry.util.genai`.

This package is installed alongside the core GenAI utilities when evaluation
features are desired. It provides the completion callback factory consumed by
``TelemetryHandler`` as well as the evaluator registry and environment helpers.
"""

from . import (
    builtins as _builtins,  # noqa: E402,F401  (auto-registration side effects)
)
from .base import Evaluator
from .bootstrap import (
    EvaluatorCompletionCallback,
    create_completion_callback,
    create_evaluation_manager,
)
from .errors import ErrorEvent, ErrorTracker
from .manager import Manager, Sampler
from .registry import get_evaluator, list_evaluators, register_evaluator

__all__ = [
    "Evaluator",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
    "Manager",
    "Sampler",
    "EvaluatorCompletionCallback",
    "create_completion_callback",
    "create_evaluation_manager",
    "ErrorEvent",
    "ErrorTracker",
]
