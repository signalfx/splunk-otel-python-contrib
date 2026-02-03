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
"""Configure DeepEval environment before importing any deepeval modules.

This module MUST be imported before any deepeval modules to ensure the
environment variables are set correctly. Import this module at the very
top of any file that imports from deepeval.

Example:
    from opentelemetry.util.evaluator import _configure_deepeval  # noqa: F401
    from deepeval import evaluate  # Now safe to import
"""

import os

# Disable DeepEval's internal telemetry (Posthog/New Relic) to prevent the evaluator
# from creating its own spans/events in the context of the instrumented application.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

# Prevent DeepEval from creating .deepeval folder (for cloud/read-only environments).
# This triggers a warning message that we suppress during import.
os.environ.setdefault("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
