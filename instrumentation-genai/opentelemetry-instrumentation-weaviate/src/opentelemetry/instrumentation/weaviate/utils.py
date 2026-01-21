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

import logging
from os import environ
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

from opentelemetry.instrumentation.weaviate.config import Config

# TODO: get semconv for vector databases
# from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI

logger = logging.getLogger(__name__)

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


def is_content_enabled() -> bool:
    """Check if content capture is enabled via environment variable.
    
    Returns:
        True if OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is set to 'true',
        False otherwise.
    """
    capture_content = environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    )
    return capture_content.lower() == "true"


def parse_url_to_host_port(url: str) -> Tuple[Optional[str], Optional[int]]:
    parsed = urlparse(url)
    host: Optional[str] = parsed.hostname
    port: Optional[int] = parsed.port
    return host, port


def extract_collection_name(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
    module_name: str,
    function_name: str,
) -> Optional[str]:
    """
    Extract collection name from Weaviate function calls.

    Args:
        wrapped: The wrapped function
        instance: The instance object (if any)
        args: Function arguments
        kwargs: Function keyword arguments
        module_name: The module name from mapping
        function_name: The function name from mapping

    Returns:
        Collection name if found, None otherwise
    """
    collection_name = None

    try:
        # Weaviate Client V4 stores this in the "request" attribute of the kwargs
        if kwargs and "request" in kwargs and hasattr(kwargs["request"], "collection"):
            collection_name = kwargs["request"].collection

        # V4: Check if instance has _name attribute directly (for Collection objects)
        elif hasattr(instance, "_name"):
            collection_name = instance._name

        # V4: Check if instance has name attribute directly
        elif hasattr(instance, "name"):
            collection_name = instance.name

        # V4: For data/query operations, check if instance has _collection attribute
        elif hasattr(instance, "_collection"):
            if hasattr(instance._collection, "_name"):
                collection_name = instance._collection._name
            elif hasattr(instance._collection, "name"):
                collection_name = instance._collection.name

        # V3: Check for class_name in kwargs (common in v3 operations)
        elif kwargs and "class_name" in kwargs:
            collection_name = kwargs["class_name"]

        # V3: Check for class_name as first positional argument
        elif args and len(args) > 0 and isinstance(args[0], str):
            # For v3 operations like client.query.get(class_name, ...)
            if (
                "query" in module_name
                or "schema" in module_name
                or "data" in module_name
            ):
                collection_name = args[0]

        return collection_name

    except Exception as e:
        # Silently ignore any errors during extraction to avoid breaking the tracing
        if Config.exception_logger:
            Config.exception_logger(e)
        pass

    return None
