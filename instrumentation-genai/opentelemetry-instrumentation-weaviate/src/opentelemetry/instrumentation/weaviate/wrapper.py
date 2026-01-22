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
Wrapper classes for Weaviate instrumentation.
"""

import json
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.instrumentation.weaviate.config import Config
from opentelemetry.metrics import Histogram
from opentelemetry.semconv.attributes import (
    db_attributes as DbAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from .mapping import SPAN_NAME_PREFIX
from .utils import (
    extract_collection_name,
    parse_url_to_host_port,
)


# Context variable for passing connection info within operation call stacks
_connection_host_context: ContextVar[Optional[str]] = ContextVar(
    "weaviate_connection_host", default=None
)
_connection_port_context: ContextVar[Optional[int]] = ContextVar(
    "weaviate_connection_port", default=None
)


class _WeaviateConnectionWrapper:
    """A wrapper that intercepts Weaviate client initialization to capture server connection details."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)

        # Extract connection details from args/kwargs before calling wrapped function
        connection_host = None
        connection_port = None
        connection_url = None

        # Extract URL from constructor arguments
        # weaviate.Client(url="http://localhost:8080", ...) for v3
        # weaviate.WeaviateClient(...) for v4
        if args and len(args) > 0:
            # First positional argument is typically the URL
            connection_url = args[0]
        elif "url" in kwargs:
            # URL passed as keyword argument
            connection_url = kwargs["url"]

        if connection_url:
            connection_host, connection_port = parse_url_to_host_port(connection_url)

        if not connection_url and "connection_params" in kwargs:
            connection_params = kwargs["connection_params"]
            if hasattr(connection_params, "http"):
                connection_host = connection_params.http.host
                connection_port = connection_params.http.port

        # Call the wrapped method to create the client
        return_value = wrapped(*args, **kwargs)

        # For v4, try to extract from the created client after initialization
        # This handles cases like connect_to_local() where URL is set internally
        if not connection_url:
            # For __init__, instance is the object being initialized
            # For connect_to_local(), return_value is the created client
            client_obj = return_value if return_value is not None else instance

            if (
                hasattr(client_obj, "_connection")
                and client_obj._connection is not None
            ):
                if hasattr(client_obj._connection, "url"):
                    connection_url = client_obj._connection.url
                    connection_host, connection_port = parse_url_to_host_port(
                        connection_url
                    )
                elif hasattr(client_obj._connection, "grpc_address"):
                    # Try gRPC address
                    grpc_addr = client_obj._connection.grpc_address
                    if grpc_addr:
                        connection_host = (
                            grpc_addr.split(":")[0] if ":" in grpc_addr else grpc_addr
                        )
                        connection_port = (
                            int(grpc_addr.split(":")[1]) if ":" in grpc_addr else None
                        )

        # Set context after extracting connection info
        _connection_host_context.set(connection_host)
        _connection_port_context.set(connection_port)

        return return_value


class _WeaviateOperationWrapper:
    """A wrapper that intercepts Weaviate client operations to create database spans with Weaviate attributes."""

    def __init__(
        self,
        tracer: Tracer,
        duration_histogram: Histogram,
        wrap_properties: Optional[Dict[str, str]] = None,
        capture_content: bool = False,
    ) -> None:
        self.tracer = tracer
        self.duration_histogram = duration_histogram
        self.wrap_properties = wrap_properties or {}
        self.capture_content = capture_content

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """Wraps the original Weaviate operation to create a tracing span."""
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

        # Extract metadata before starting span (needed for both span attributes and metrics)
        module_name = self.wrap_properties.get("module", "")
        function_name = self.wrap_properties.get("function", "")
        connection_host = _connection_host_context.get()
        connection_port = _connection_port_context.get()
        collection_name = extract_collection_name(
            wrapped, instance, args, kwargs, module_name, function_name
        )

        with self.tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(DbAttributes.DB_SYSTEM_NAME, "weaviate")

            if function_name:
                span.set_attribute(DbAttributes.DB_OPERATION_NAME, function_name)
            if collection_name:
                span.set_attribute(DbAttributes.DB_COLLECTION_NAME, collection_name)

            if connection_host is not None:
                span.set_attribute(ServerAttributes.SERVER_ADDRESS, connection_host)
            if connection_port is not None:
                span.set_attribute(ServerAttributes.SERVER_PORT, connection_port)

            # Record operation duration as a metric
            start_time = time.perf_counter()
            try:
                return_value = wrapped(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Build metric attributes
                metric_attributes: Dict[str, AttributeValue] = {
                    DbAttributes.DB_SYSTEM_NAME: "weaviate",
                }
                if function_name:
                    metric_attributes[DbAttributes.DB_OPERATION_NAME] = function_name
                if collection_name:
                    metric_attributes[DbAttributes.DB_COLLECTION_NAME] = collection_name
                if connection_host is not None:
                    metric_attributes[ServerAttributes.SERVER_ADDRESS] = connection_host
                if connection_port is not None:
                    metric_attributes[ServerAttributes.SERVER_PORT] = connection_port

                # Record the duration metric with span context to link metric to trace
                try:
                    context = trace.set_span_in_context(span)
                except (TypeError, ValueError, AttributeError):
                    context = None

                self.duration_histogram.record(
                    duration_ms, attributes=metric_attributes, context=context
                )

            # Extract documents from similarity search operations
            if self._is_similarity_search():
                documents = self._extract_documents_from_response(return_value)
                if documents:
                    span.set_attribute("db.weaviate.documents.count", len(documents))
                    # emit the documents as events only if content capture is enabled
                    if self.capture_content:
                        for doc in documents:
                            # emit the document content as an event
                            query = ""
                            if "query" in kwargs:
                                query = json.dumps(kwargs["query"])
                            
                            # Serialize content - if it's already a string, use it; otherwise json.dumps
                            content = doc["content"]
                            if isinstance(content, str):
                                content_str = content
                            else:
                                content_str = json.dumps(content)
                            
                            attributes = {
                                "db.weaviate.document.content": content_str,
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
                                attributes["db.weaviate.document.score"] = doc["score"]
                            if query:
                                attributes["db.weaviate.document.query"] = query
                            span.add_event("weaviate.document", attributes=attributes)

        return return_value

    def _is_similarity_search(self) -> bool:
        """Check if the operation is a similarity search or retrieval operation that returns documents."""
        module_name = self.wrap_properties.get("module", "")
        function_name = self.wrap_properties.get("function", "")
        return (
            "do" in function_name.lower()
            or "near_text" in function_name.lower()
            or "near_vector" in function_name.lower()
            or "fetch_object" in function_name.lower()  # Matches both fetch_objects and fetch_object_by_id
            or "graphql_raw_query" in function_name.lower()  # Raw GraphQL queries
        )


    def _extract_documents_from_response(self, response: Any) -> list[dict[str, Any]]:
        """Extract documents from weaviate response."""
        documents: list[dict[str, Any]] = []
        try:
            # Handle single object response (e.g., from fetch_object_by_id)
            if hasattr(response, "properties") and not hasattr(response, "objects"):
                doc: dict[str, Any] = {}
                # Convert properties to dict if it's not already
                if isinstance(response.properties, dict):
                    doc["content"] = response.properties
                else:
                    # For non-dict properties, convert to string representation
                    doc["content"] = str(response.properties)

                # Extract similarity scores from single object
                if hasattr(response, "metadata") and response.metadata:
                    metadata = response.metadata
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
                    if hasattr(metadata, "score") and metadata.score is not None:
                        doc["score"] = metadata.score

                documents.append(doc)
            # Handle collection of objects (e.g., from fetch_objects, near_text, etc.)
            elif hasattr(response, "objects"):
                for obj in response.objects:
                    doc: dict[str, Any] = {}
                    if hasattr(obj, "properties"):
                        # Convert properties to dict if it's not already
                        if isinstance(obj.properties, dict):
                            doc["content"] = obj.properties
                        else:
                            doc["content"] = str(obj.properties)

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
                        if hasattr(metadata, "score") and metadata.score is not None:
                            doc["score"] = metadata.score

                    documents.append(doc)
            elif hasattr(response, "get") and response.get:
                # Handle raw GraphQL query responses (_RawGQLReturn with .get attribute)
                for collection_name, objects in response.get.items():
                    if isinstance(objects, list):
                        for obj in objects:
                            if isinstance(obj, dict):
                                doc: dict[str, Any] = {}
                                doc["content"] = obj
                                # Extract metadata from _additional if present
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
        except Exception as e:
            if Config.exception_logger:
                Config.exception_logger(e)
            pass
        return documents
