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

import contextlib
import io
import logging
import os
import sys
import warnings

_suppressed = False


def suppress_deepeval_output() -> None:
    """Suppress verbose output from DeepEval and related libraries.

    This function:
    1. Sets DEEPEVAL_TELEMETRY_OPT_OUT=YES to disable DeepEval's internal telemetry
       (Posthog/New Relic). This prevents the evaluator from creating its own
       telemetry in the context of the instrumented application.
    2. Sets DEEPEVAL_FILE_SYSTEM=READ_ONLY to prevent DeepEval from creating
       .deepeval folder, which may not work in cloud/read-only environments.
    3. Suppresses OpenTelemetry LogRecord deprecation warnings.
    4. Sets logging levels to WARNING for LiteLLM and DeepEval loggers.

    Note: The DEEPEVAL_FILE_SYSTEM=READ_ONLY setting triggers a warning from
    DeepEval that is printed during module import. This warning is captured
    and suppressed by redirecting stdout during the initial import.

    This function is idempotent - calling it multiple times has no effect
    after the first call.
    """
    global _suppressed
    if _suppressed:
        return
    _suppressed = True

    # Set environment variables for DeepEval configuration
    # These must be set before importing deepeval

    # Disable DeepEval's internal telemetry (Posthog/New Relic) to prevent
    # the evaluator from emitting its own spans/events in the instrumented app context
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    # Prevent DeepEval from creating .deepeval folder (for cloud/read-only environments)
    # Note: This triggers a warning from DeepEval that we suppress below
    os.environ.setdefault("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")

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


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout.

    Used to capture and discard the "Warning: DeepEval is configured for
    read only environment" message that deepeval prints at import time.
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def import_deepeval_quietly():
    """Import deepeval modules while suppressing their startup warnings.

    This should be called after suppress_deepeval_output() to ensure
    environment variables are set before importing deepeval.
    """
    with suppress_stdout():
        # These imports trigger the "read only environment" warning
        import deepeval.prompt.prompt  # noqa: F401
        import deepeval.test_run.cache  # noqa: F401
        import deepeval.test_run.test_run  # noqa: F401


# Auto-suppress on import for convenience
suppress_deepeval_output()


__all__ = [
    "suppress_deepeval_output",
    "import_deepeval_quietly",
    "suppress_stdout",
]
