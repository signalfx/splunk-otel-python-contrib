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

This module provides utilities to reduce noise from DeepEval, LiteLLM,
and OpenTelemetry when running evaluations.

Key functions:
- suppress_deepeval_output(): Configure logging levels to suppress verbose output
- suppress_stdout(): Context manager to temporarily suppress stdout
- suppress_rich_console(): Context manager to suppress rich console output during evaluation
- import_deepeval_quietly(): Import deepeval modules while suppressing startup warnings
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from typing import Generator

_suppressed = False


def suppress_deepeval_output() -> None:
    """Suppress verbose logging output from DeepEval and related libraries.

    This function:
    1. Suppresses OpenTelemetry LogRecord deprecation warnings.
    2. Sets logging levels to WARNING for LiteLLM and DeepEval loggers.

    Note: Environment variables (DEEPEVAL_TELEMETRY_OPT_OUT, DEEPEVAL_FILE_SYSTEM)
    should be set in the deepeval module before importing deepeval.

    This function is idempotent - calling it multiple times has no effect
    after the first call.
    """
    global _suppressed
    if _suppressed:
        return
    _suppressed = True

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
def suppress_stdout() -> Generator[None, None, None]:
    """Context manager to temporarily suppress stdout.

    Used to capture and discard unwanted print statements from DeepEval.
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


@contextlib.contextmanager
def suppress_rich_console() -> Generator[None, None, None]:
    """Context manager to suppress rich console output during evaluation.

    DeepEval uses rich.console.Console to print evaluation results, progress
    indicators, and promotional messages. This context manager temporarily
    replaces the console's file with a null stream to suppress all output.

    Usage:
        with suppress_rich_console():
            result = evaluate([test_case], metrics)
    """
    try:
        from deepeval.test_run.test_run import console as test_run_console
    except ImportError:
        # If we can't import, just yield without suppression
        yield
        return

    # Save the original file
    original_file = test_run_console.file

    # Replace with a null stream
    test_run_console.file = io.StringIO()
    try:
        yield
    finally:
        # Restore the original file
        test_run_console.file = original_file


def patch_deepeval_console() -> None:
    """Patch DeepEval's rich console to suppress all output permanently.

    This function replaces the console's file with a null stream, effectively
    suppressing all rich console output from DeepEval (evaluation results,
    progress indicators, promotional messages, etc.).

    Call this function once after importing deepeval to suppress all console output.
    """
    try:
        from deepeval.test_run.test_run import console as test_run_console

        test_run_console.file = io.StringIO()
    except ImportError:
        pass

    # Also patch the utils console if it exists
    try:
        from deepeval.utils import console as utils_console

        utils_console.file = io.StringIO()
    except (ImportError, AttributeError):
        pass

    # Patch custom_console used in progress bars
    try:
        from deepeval import utils as deepeval_utils

        if hasattr(deepeval_utils, "custom_console"):
            deepeval_utils.custom_console.file = io.StringIO()
    except (ImportError, AttributeError):
        pass


def import_deepeval_quietly() -> None:
    """Import deepeval modules while suppressing their startup warnings.

    This should be called after environment variables are set to ensure
    they take effect before importing deepeval.
    """
    with suppress_stdout():
        # These imports trigger the "read only environment" warning
        import deepeval.prompt.prompt  # noqa: F401
        import deepeval.test_run.cache  # noqa: F401
        import deepeval.test_run.test_run  # noqa: F401

    # Patch the console to suppress all future output
    patch_deepeval_console()


# Auto-suppress logging on import for convenience
suppress_deepeval_output()


__all__ = [
    "suppress_deepeval_output",
    "import_deepeval_quietly",
    "suppress_stdout",
    "suppress_rich_console",
    "patch_deepeval_console",
]
