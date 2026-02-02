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
"""Suppress verbose DeepEval and related library output.

This module configures environment variables and logging levels to reduce
noise from DeepEval, LiteLLM, and OpenTelemetry when running evaluations.

Import this module early (before importing deepeval) to ensure suppressions
take effect:

    from opentelemetry.util.evaluator.suppress_output import suppress_deepeval_output
    suppress_deepeval_output()

Or use the auto-suppression by simply importing this module:

    import opentelemetry.util.evaluator.suppress_output  # noqa: F401
"""

from __future__ import annotations

import logging
import os
import warnings

_suppressed = False


def suppress_deepeval_output() -> None:
    """Suppress verbose output from DeepEval and related libraries.

    This function:
    1. Sets DEEPEVAL_FILE_SYSTEM=READ_ONLY to prevent disk cache warnings
    2. Sets DEEPEVAL_TELEMETRY_OPT_OUT=YES to disable DeepEval telemetry
    3. Suppresses OpenTelemetry LogRecord deprecation warnings
    4. Sets logging levels to WARNING for LiteLLM and DeepEval loggers

    This function is idempotent - calling it multiple times has no effect
    after the first call.
    """
    global _suppressed
    if _suppressed:
        return
    _suppressed = True

    # Set environment variables for DeepEval configuration
    # These must be set before importing deepeval
    os.environ.setdefault("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    # Suppress OpenTelemetry LogRecord deprecation warning
    try:
        from opentelemetry.sdk._logs._internal import LogDeprecatedInitWarning

        warnings.filterwarnings("ignore", category=LogDeprecatedInitWarning)
    except ImportError:
        # Fallback: filter by message if class not available
        warnings.filterwarnings("ignore", message="LogRecord will be removed")

    # Suppress verbose DeepEval and LiteLLM logging
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
    logging.getLogger("deepeval").setLevel(logging.WARNING)
    logging.getLogger("deepeval.evaluate").setLevel(logging.WARNING)


# Auto-suppress on import for convenience
suppress_deepeval_output()


__all__ = ["suppress_deepeval_output"]
