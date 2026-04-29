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

"""
VertexAI client instrumentation supporting `google-cloud-aiplatform` SDK, it can be enabled by
using ``VertexAIInstrumentor``.

.. _vertexai: https://pypi.org/project/google-cloud-aiplatform/

Usage
-----

.. code:: python

    import vertexai
    from vertexai.generative_models import GenerativeModel
    from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

    VertexAIInstrumentor().instrument()

    vertexai.init()
    model = GenerativeModel("gemini-1.5-flash-002")
    chat_completion = model.generate_content(
        "Write a short poem on OpenTelemetry."
    )

API
---
"""

from __future__ import annotations

from typing import Any, Collection

from wrapt import (
    wrap_function_wrapper,  # type: ignore[reportUnknownVariableType]
)

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.vertexai.package import _instruments
from opentelemetry.instrumentation.vertexai.patch import (
    agenerate_content,
    generate_content,
)
from opentelemetry.instrumentation.vertexai.utils import is_content_enabled
from opentelemetry.util.genai.handler import get_telemetry_handler


class VertexAIInstrumentor(BaseInstrumentor):
    def __init__(self) -> None:
        super().__init__()
        self._methods_to_unwrap: list[tuple[Any, str]] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any):
        """Enable VertexAI instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        meter_provider = kwargs.get("meter_provider")

        handler = get_telemetry_handler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        capture_content = is_content_enabled()

        # This import is very slow, do it lazily in case instrument() is not called
        # pylint: disable=import-outside-toplevel
        from google.cloud.aiplatform_v1.services.prediction_service import (  # noqa: PLC0415
            async_client,
            client,
        )
        from google.cloud.aiplatform_v1beta1.services.prediction_service import (  # noqa: PLC0415
            async_client as async_client_v1beta1,
        )
        from google.cloud.aiplatform_v1beta1.services.prediction_service import (  # noqa: PLC0415
            client as client_v1beta1,
        )

        sync_wrapper = generate_content(capture_content, handler)
        async_wrapper = agenerate_content(capture_content, handler)

        for client_class in (
            client.PredictionServiceClient,
            client_v1beta1.PredictionServiceClient,
        ):
            method_name = client_class.generate_content.__name__  # type: ignore[reportUnknownMemberType]
            wrap_function_wrapper(
                client_class,
                name=method_name,
                wrapper=sync_wrapper,
            )
            self._methods_to_unwrap.append((client_class, method_name))

        for client_class in (
            async_client.PredictionServiceAsyncClient,
            async_client_v1beta1.PredictionServiceAsyncClient,
        ):
            method_name = client_class.generate_content.__name__  # type: ignore[reportUnknownMemberType]
            wrap_function_wrapper(
                client_class,
                name=method_name,
                wrapper=async_wrapper,
            )
            self._methods_to_unwrap.append((client_class, method_name))

    def _uninstrument(self, **kwargs: Any) -> None:
        for client_class, method_name in self._methods_to_unwrap:
            unwrap(client_class, method_name)
