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
Session propagation for MCP protocol via OTel Baggage.

Session context (gen_ai.conversation.id, user.id, customer.id) is propagated
via the standard W3C Baggage header format, using the standard OTel Propagation
API (``propagate.inject()`` / ``propagate.extract()``). The MCP ``params._meta``
object serves as the carrier — analogous to HTTP headers.

When baggage propagation is enabled
(``OTEL_INSTRUMENTATION_GENAI_SESSION_PROPAGATION=baggage``), session values
set via ``set_session_context()`` are injected as OTel Baggage entries by the
standard ``propagate.inject()`` call in the transport instrumentor, and
extracted on the server side to restore session context for GenAI operations.

Trace context injection/extraction itself is handled directly by
``TransportInstrumentor`` (in ``transport_instrumentor.py``) which calls
``propagate.inject()`` / ``propagate.extract()`` with the pydantic Meta object
as the carrier. This module provides only the session-specific
``restore_session_from_context`` helper used by the server-side transport
wrapper.
"""

import logging
from typing import Optional

from opentelemetry.context import Context

_LOGGER = logging.getLogger(__name__)


def restore_session_from_context(ctx: Optional[Context] = None) -> None:
    """Restore session context from OTel Baggage in the given context.

    This function reads gen_ai.conversation.id, user.id, and customer.id from
    OTel Baggage and sets them in the local session context (ContextVar),
    making them available to GenAI instrumentation on the server side.

    This is called automatically by the transport instrumentor when
    baggage propagation is enabled.

    Args:
        ctx: OTel context containing baggage. If None, uses current context.
    """
    try:
        from opentelemetry import baggage

        from opentelemetry.util.genai.handler import (
            _is_baggage_propagation_enabled,
            set_session_context,
        )

        if not _is_baggage_propagation_enabled():
            return

        session_id = baggage.get_baggage("gen_ai.conversation.id", ctx)
        user_id = baggage.get_baggage("user.id", ctx)
        customer_id = baggage.get_baggage("customer.id", ctx)

        if session_id or user_id or customer_id:
            # Set local session context without re-propagating via baggage
            # (it's already in the context from extraction)
            set_session_context(
                session_id=session_id,
                user_id=user_id,
                customer_id=customer_id,
                propagate_via_baggage=False,
            )
            _LOGGER.debug(
                f"Restored session from baggage: "
                f"gen_ai.conversation.id={session_id}, user.id={user_id}, "
                f"customer.id={customer_id}"
            )
    except ImportError:
        _LOGGER.debug("opentelemetry.util.genai not available for session restore")
    except Exception as e:
        _LOGGER.debug(f"Failed to restore session from baggage: {e}")
