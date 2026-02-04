# Copyright The OpenTelemetry Authors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

# Configure DeepEval environment BEFORE importing any deepeval modules
from . import _configure_deepeval  # noqa: F401, isort: skip

# Import deepeval modules while suppressing startup warnings
from .suppress_output import import_deepeval_quietly  # isort: skip

import_deepeval_quietly()

from typing import Any  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402

from opentelemetry.util.genai.evals.normalize import (  # noqa: E402
    normalize_invocation,
)
from opentelemetry.util.genai.types import (  # noqa: E402
    AgentInvocation,
    GenAI,
    LLMInvocation,
)


def build_llm_test_case(invocation: GenAI) -> Any | None:
    if not isinstance(invocation, (LLMInvocation, AgentInvocation)):
        return None
    canonical = normalize_invocation(invocation)
    if not canonical.input_text and not canonical.output_text:
        return None
    name = (
        invocation.request_model
        if isinstance(invocation, LLMInvocation)
        else (invocation.operation or invocation.name)
    )
    return LLMTestCase(
        input=canonical.input_text or "",
        actual_output=canonical.output_text or "",
        context=canonical.context,
        retrieval_context=canonical.retrieval_context,
        additional_metadata=canonical.metadata or None,
        name=name,
    )


__all__ = ["build_llm_test_case"]
