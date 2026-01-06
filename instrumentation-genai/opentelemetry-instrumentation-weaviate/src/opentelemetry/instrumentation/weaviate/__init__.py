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
Weaviate client instrumentation supporting `weaviate-client`, it can be enabled by
using ``WeaviateInstrumentor``.

.. _weaviate-client: https://pypi.org/project/weaviate-client/

Usage
-----

.. code:: python

    import weaviate
    from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

    WeaviateInstrumentor().instrument()

    # Weaviate v4 API
    client = weaviate.connect_to_local()
    # Your Weaviate operations will now be traced

    # Weaviate v3 API (also supported)
    # client = weaviate.Client("http://localhost:8080")

API
---
"""

from typing import Any, Collection, Optional

import weaviate
from wrapt import wrap_function_wrapper  # type: ignore

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.weaviate.config import Config
from opentelemetry.instrumentation.weaviate.version import __version__

# Potentially not needed.
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import Tracer, get_tracer

from .mapping import MAPPING_V3, MAPPING_V4
from .wrapper import (
    _WeaviateConnectionWrapper,
    _WeaviateOperationWrapper,
)

WEAVIATE_V3 = 3
WEAVIATE_V4 = 4

weaviate_version = None
_instruments = ("weaviate-client >= 3.0.0, < 5",)


class WeaviateInstrumentor(BaseInstrumentor):
    """An instrumentor for Weaviate's client library."""

    def __init__(self, exception_logger: Optional[Any] = None) -> None:
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        global weaviate_version
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )

        try:
            major_version = int(weaviate.__version__.split(".")[0])
            if major_version >= 4:
                weaviate_version = WEAVIATE_V4
            else:
                weaviate_version = WEAVIATE_V3
        except (ValueError, IndexError):
            # Default to V3 if version parsing fails
            weaviate_version = WEAVIATE_V3

        self._instrument_client(weaviate_version, tracer)

        # Wrap v4 connection functions to capture connection info
        if weaviate_version == WEAVIATE_V4:
            from .mapping import CONNECTION_WRAPPING

            for conn_wrap in CONNECTION_WRAPPING:
                try:
                    wrap_function_wrapper(
                        module=conn_wrap["module"],
                        name=conn_wrap["name"],
                        wrapper=_WeaviateConnectionWrapper(tracer),
                    )
                except (ImportError, AttributeError):
                    # Connection function might not exist in this version
                    pass

        wrappings = MAPPING_V3 if weaviate_version == WEAVIATE_V3 else MAPPING_V4
        for to_wrap in wrappings:
            name = ".".join([to_wrap["name"], to_wrap["function"]])
            wrap_function_wrapper(
                module=to_wrap["module"],
                name=name,
                wrapper=_WeaviateOperationWrapper(tracer, wrap_properties=to_wrap),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        global weaviate_version
        wrappings = MAPPING_V3 if weaviate_version == WEAVIATE_V3 else MAPPING_V4
        for to_unwrap in wrappings:
            try:
                module = ".".join([to_unwrap["module"], to_unwrap["name"]])
                unwrap(
                    module,
                    to_unwrap["function"],
                )
            except (ImportError, AttributeError, ValueError):
                # Ignore errors when unwrapping - module might not be loaded
                # or function might not be wrapped
                pass

        # unwrap the connection initialization to remove the context variable injection
        try:
            if weaviate_version == WEAVIATE_V3:
                unwrap("weaviate.Client", "__init__")
            elif weaviate_version == WEAVIATE_V4:
                unwrap("weaviate.WeaviateClient", "__init__")
        except (ImportError, AttributeError, ValueError):
            # Ignore errors when unwrapping connection methods
            pass

    def _instrument_client(self, version: int, tracer: Tracer) -> None:
        name = "Client.__init__"
        if version == WEAVIATE_V4:
            name = "WeaviateClient.__init__"

        wrap_function_wrapper(
            module="weaviate",
            name=name,
            wrapper=_WeaviateConnectionWrapper(tracer),
        )


__all__ = [
    "WeaviateInstrumentor",
]
