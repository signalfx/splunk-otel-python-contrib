# Copyright Splunk Inc.
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

"""OpenTelemetry instrumentation for AWS Bedrock Runtime."""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

try:
    import botocore.client as botocore_client
except (ImportError, ModuleNotFoundError) as error:
    botocore_client = None
    _BOTOCORE_CLIENT_IMPORT_ERROR = error
else:
    _BOTOCORE_CLIENT_IMPORT_ERROR = None

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import get_telemetry_handler

from .package import _instruments
from .version import __version__
from .wrappers import bedrock_runtime_api_call_wrapper

__all__ = ["BedrockInstrumentor", "__version__"]

_LOGGER = logging.getLogger(__name__)


class BedrockInstrumentor(BaseInstrumentor):
    """Instrument AWS Bedrock Runtime model calls as GenAI LLM invocations."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")
        handler = get_telemetry_handler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        try:
            wrap_function_wrapper(
                "botocore.client",
                "BaseClient._make_api_call",
                bedrock_runtime_api_call_wrapper(handler),
            )
        except (ImportError, ModuleNotFoundError):
            _LOGGER.debug(
                "botocore not importable while instrumenting Bedrock Runtime",
                exc_info=True,
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        if botocore_client is None:
            _LOGGER.debug(
                "botocore not importable while uninstrumenting Bedrock Runtime",
                exc_info=_BOTOCORE_CLIENT_IMPORT_ERROR,
            )
            return

        try:
            unwrap(botocore_client.BaseClient, "_make_api_call")
        except Exception:
            _LOGGER.warning(
                "Failed to uninstrument Bedrock Runtime",
                exc_info=True,
            )
