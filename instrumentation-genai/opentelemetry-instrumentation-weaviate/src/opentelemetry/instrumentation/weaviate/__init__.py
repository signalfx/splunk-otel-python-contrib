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

import json
from contextvars import ContextVar
from typing import Any, Collection, Dict, Optional

import weaviate
from wrapt import wrap_function_wrapper  # type: ignore

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    is_instrumentation_enabled,
    unwrap,
)
from opentelemetry.instrumentation.weaviate.config import Config
from opentelemetry.instrumentation.weaviate.version import __version__

# from opentelemetry.metrics import get_meter
# from opentelemetry._events import get_event_logger
from opentelemetry.semconv.attributes import (
    db_attributes as DbAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)

# Potentially not needed.
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import SpanKind, Tracer, get_tracer

from .mapping import MAPPING_V3, MAPPING_V4, SPAN_NAME_PREFIX
from .utils import (
    extract_collection_name,
    parse_url_to_host_port,
)

WEAVIATE_V3 = 3
WEAVIATE_V4 = 4

weaviate_version = None
_instruments = ("weaviate-client >= 3.0.0, < 5",)


# Context variable for passing connection info within operation call stacks
_connection_host_context: ContextVar[Optional[str]] = ContextVar(
    "weaviate_connection_host", default=None
)
_connection_port_context: ContextVar[Optional[int]] = ContextVar(
    "weaviate_connection_port", default=None
)


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

        self._get_server_details(weaviate_version, tracer)

        wrappings = (
            MAPPING_V3 if weaviate_version == WEAVIATE_V3 else MAPPING_V4
        )
        for to_wrap in wrappings:
            name = ".".join([to_wrap["name"], to_wrap["function"]])
            wrap_function_wrapper(
                module=to_wrap["module"],
                name=name,
                wrapper=_WeaviateTraceInjectionWrapper(
                    tracer, wrap_properties=to_wrap
                ),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        global weaviate_version
        wrappings = (
            MAPPING_V3 if weaviate_version == WEAVIATE_V3 else MAPPING_V4
        )
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

    def _get_server_details(self, version: int, tracer: Tracer) -> None:
        name = "Client.__init__"
        if version == WEAVIATE_V4:
            name = "WeaviateClient.__init__"

        wrap_function_wrapper(
            module="weaviate",
            name=name,
            wrapper=_WeaviateConnectionInjectionWrapper(tracer),
        )


class _WeaviateConnectionInjectionWrapper:
    """
    A wrapper that intercepts calls to weaviate connection methods to inject tracing headers.
    This is used to create spans for Weaviate connection operations.
    """

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)

        # Extract connection details from args/kwargs before calling wrapped function
        connection_host = None
        connection_port = None
        connection_url = None

        # For v3, extract URL from constructor arguments
        # weaviate.Client(url="http://localhost:8080", ...)
        if args and len(args) > 0:
            # First positional argument is typically the URL
            connection_url = args[0]
        elif "url" in kwargs:
            # URL passed as keyword argument
            connection_url = kwargs["url"]

        if connection_url:
            connection_host, connection_port = parse_url_to_host_port(
                connection_url
            )

        return_value = wrapped(*args, **kwargs)

        # For v4, try to extract from instance after creation (fallback)
        if (
            not connection_url
            and hasattr(instance, "_connection")
            and instance._connection is not None
        ):
            connection_url = instance._connection.url
            if connection_url:
                connection_host, connection_port = parse_url_to_host_port(
                    connection_url
                )

        _connection_host_context.set(connection_host)
        _connection_port_context.set(connection_port)
        return return_value


class _WeaviateTraceInjectionWrapper:
    """
    A wrapper that intercepts calls to weaviate to inject tracing headers.
    This is used to create spans for Weaviate operations.
    """

    def __init__(
        self, tracer: Tracer, wrap_properties: Optional[Dict[str, str]] = None
    ) -> None:
        self.tracer = tracer
        self.wrap_properties = wrap_properties or {}

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """
        Wraps the original function to inject tracing headers.
        """
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)

        # Create DB span for all operations
        return self._create_db_span(wrapped, instance, args, kwargs)

    def _create_db_span(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Create a regular DB operation span."""
        name = self.wrap_properties.get(
            "span_name",
            getattr(wrapped, "__name__", "unknown"),
        )
        name = f"{SPAN_NAME_PREFIX}.{name}"
        with self.tracer.start_as_current_span(
            name, kind=SpanKind.CLIENT
        ) as span:
            span.set_attribute(DbAttributes.DB_SYSTEM_NAME, "weaviate")

            # Extract operation name dynamically from the function call
            module_name = self.wrap_properties.get("module", "")
            function_name = self.wrap_properties.get("function", "")
            span.set_attribute(DbAttributes.DB_OPERATION_NAME, function_name)

            # Extract collection name from the operation
            collection_name = extract_collection_name(
                wrapped, instance, args, kwargs, module_name, function_name
            )
            if collection_name:
                # Use a Weaviate-specific collection attribute similar to MongoDB's DB_MONGODB_COLLECTION
                span.set_attribute(
                    "db.weaviate.collection.name", collection_name
                )

            connection_host = _connection_host_context.get()
            connection_port = _connection_port_context.get()
            if connection_host is not None:
                span.set_attribute(
                    ServerAttributes.SERVER_ADDRESS, connection_host
                )
            if connection_port is not None:
                span.set_attribute(
                    ServerAttributes.SERVER_PORT, connection_port
                )

            return_value = wrapped(*args, **kwargs)

            # Extract documents from similarity search operations
            if self._is_similarity_search():
                documents = self._extract_documents_from_response(return_value)
                if documents:
                    span.set_attribute(
                        "db.weaviate.documents.count", len(documents)
                    )
                    # emit the documents as events
                    for doc in documents:
                        # emit the document content as an event
                        query = ""
                        if "query" in kwargs:
                            query = json.dumps(kwargs["query"])
                        attributes = {
                            "db.weaviate.document.content": json.dumps(
                                doc["content"]
                            ),
                        }

                        # Only add non-None values to attributes
                        if doc.get("distance") is not None:
                            attributes["db.weaviate.document.distance"] = doc[
                                "distance"
                            ]
                        if doc.get("certainty") is not None:
                            attributes["db.weaviate.document.certainty"] = doc[
                                "certainty"
                            ]
                        if doc.get("score") is not None:
                            attributes["db.weaviate.document.score"] = doc[
                                "score"
                            ]
                        if query:
                            attributes["db.weaviate.document.query"] = query
                        span.add_event(
                            "weaviate.document", attributes=attributes
                        )

        return return_value

    def _is_similarity_search(self) -> bool:
        """
        Check if this is a similarity search operation.
        """
        module_name = self.wrap_properties.get("module", "")
        function_name = self.wrap_properties.get("function", "")
        return (
            "query" in module_name.lower()
            or "do" in function_name.lower()
            or "near_text" in function_name.lower()
            or "fetch_objects" in function_name.lower()
        )

    def _extract_documents_from_response(
        self, response: Any
    ) -> list[dict[str, Any]]:
        """
        Extract documents from weaviate response.
        """
        documents: list[dict[str, Any]] = []
        try:
            if hasattr(response, "objects"):
                for obj in response.objects:
                    doc: dict[str, Any] = {}
                    if hasattr(obj, "properties"):
                        doc["content"] = obj.properties

                    # Extract similarity scores
                    if hasattr(obj, "metadata") and obj.metadata:
                        metadata = obj.metadata
                        if (
                            hasattr(metadata, "distance")
                            and metadata.distance is not None
                        ):
                            doc["distance"] = metadata.distance
                        if (
                            hasattr(metadata, "certainty")
                            and metadata.certainty is not None
                        ):
                            doc["certainty"] = metadata.certainty
                        if (
                            hasattr(metadata, "score")
                            and metadata.score is not None
                        ):
                            doc["score"] = metadata.score

                    documents.append(doc)
            elif "data" in response:
                # Handle GraphQL responses
                for response_key in response["data"].keys():
                    for collection in response["data"][response_key]:
                        for obj in response["data"][response_key][collection]:
                            doc: dict[str, Any] = {}
                            doc["content"] = dict(obj)
                            del doc["content"]["_additional"]
                            if "_additional" in obj:
                                metadata = obj["_additional"]
                                if (
                                    "distance" in metadata
                                    and metadata["distance"] is not None
                                ):
                                    doc["distance"] = metadata["distance"]
                                if (
                                    "certainty" in metadata
                                    and metadata["certainty"] is not None
                                ):
                                    doc["certainty"] = metadata["certainty"]
                                if (
                                    "score" in metadata
                                    and metadata["score"] is not None
                                ):
                                    doc["score"] = metadata["score"]
                            documents.append(doc)
        except Exception:
            # silently handle extraction errors
            pass
        return documents


__all__ = [
    "WeaviateInstrumentor",
]