OpenTelemetry FastMCP Instrumentation
=====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-fastmcp.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-fastmcp/

This library provides automatic instrumentation for `FastMCP <https://github.com/jlowin/fastmcp>`_,
a Python library for building Model Context Protocol (MCP) servers.

Compatibility Matrix
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Instrumentation
     - fastmcp
     - util-genai
     - Notes
   * - 0.1.1
     - 2.x (jlowin/fastmcp)
     - <= 0.1.9
     - PR #147. Wraps ``ToolManager.call_tool``.
   * - 0.2.0
     - >= 3.0.0, < 4
     - >= 0.1.12
     - Wraps ``FastMCP.call_tool``, ``read_resource``, ``render_prompt``. Breaking change from 0.1.x.
   * - 0.2.1
     - >= 3.0.0, < 4
     - >= 0.1.13
     - ``initialize`` MCPOperation replaces ``AgentInvocation`` as session root span on both client and server. MCP session duration metrics now emitted from the ``initialize`` span.

Installation
------------

::

    pip install splunk-otel-instrumentation-fastmcp

This can also be installed with the ``instruments`` extra to automatically install
FastMCP:

::

    pip install 'splunk-otel-instrumentation-fastmcp[instruments]'


Usage
-----

**Programmatic instrumentation:**

.. code-block:: python

    from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor

    FastMCPInstrumentor().instrument()


**Auto-instrumentation:**

The instrumentation is automatically applied when using OpenTelemetry auto-instrumentation:

::

    opentelemetry-instrument python your_mcp_server.py


Environment Variables
---------------------

The following environment variables control the instrumentation behavior:

- ``OTEL_INSTRUMENTATION_GENAI_ENABLE``: Enable/disable instrumentation (default: ``true``)
- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``: Capture tool arguments and results (default: ``false``)
- ``OTEL_INSTRUMENTATION_GENAI_EMITTERS``: Select emitters - span, metric, event (default: ``span``)


What is Instrumented
--------------------

Server-side (v0.2.1 — FastMCP 3.x):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Session lifecycle** — ``Server.run`` is wrapped with an ``initialize`` MCPOperation span that spans the full session and acts as the root span for all server-side operations.
- Tool execution via ``FastMCP.call_tool``
- Resource reads via ``FastMCP.read_resource``
- Prompt rendering via ``FastMCP.render_prompt``

Client-side (v0.2.1):
~~~~~~~~~~~~~~~~~~~~~~

- **Session lifecycle** — ``Client.__aenter__`` / ``__aexit__`` are wrapped with an ``initialize`` MCPOperation span. After a successful connect the span is enriched with ``mcp.protocol.version`` and ``sdot.mcp.server_name`` from the server handshake.
- Tool calls via ``Client.call_tool``
- Tool listings via ``Client.list_tools``
- Resource reads via ``Client.read_resource``
- Prompt rendering via ``Client.get_prompt``

Server-side (v0.2.0 — FastMCP 3.x):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- FastMCP server initialization
- Tool execution via ``FastMCP.call_tool``
- Resource reads via ``FastMCP.read_resource``
- Prompt rendering via ``FastMCP.render_prompt``

Server-side (v0.1.x — FastMCP 2.x):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- FastMCP server initialization
- Tool execution via ``ToolManager.call_tool``

Transport-level:
~~~~~~~~~~~~~~~~

- Automatic trace context propagation via ``_meta`` field
- Works for all MCP transports: stdio, SSE, streamable-http

Trace Context Propagation
-------------------------

The instrumentation automatically propagates W3C TraceContext (traceparent, tracestate)
and baggage between MCP client and server processes. This enables distributed tracing
across process boundaries:

- Client spans and server spans share the same ``trace_id``
- Server tool execution spans are children of client tool call spans
- No code changes required in your MCP server or client

Transport bridge (``transport_instrumentor.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MCP Python SDK v1.x (current stable, up to 1.27.0) does not natively
propagate OpenTelemetry context.  This instrumentation includes a
**transport-layer bridge** (``transport_instrumentor.py``) that:

- **Client side**: wraps ``BaseSession.send_request`` to inject ``traceparent``,
  ``tracestate``, and ``baggage`` into ``params.meta`` (serialized as ``_meta``
  on the wire).
- **Server side**: wraps ``Server._handle_request`` to extract trace context
  from ``request_meta`` and populate an ``MCPRequestContext`` (via
  ``ContextVar``) for the server instrumentor to read transport-level attributes
  like ``jsonrpc.request.id`` and ``network.transport``.

Upstream native support (mcp v2.x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Native OTel support has been merged to the upstream SDK's ``main`` branch,
targeting **v2.x** (not yet released as of Apr 2026):

- `#2298 <https://github.com/modelcontextprotocol/python-sdk/pull/2298>`_
  (merged Mar 31) — propagate ``contextvars.Context`` through anyio streams.
  Supersedes `#1996 <https://github.com/modelcontextprotocol/python-sdk/pull/1996>`_
  (closed).
- `#2381 <https://github.com/modelcontextprotocol/python-sdk/pull/2381>`_
  (merged Mar 31) — native CLIENT + SERVER spans, W3C trace-context
  inject/extract via ``params.meta``, and ``opentelemetry-api`` as a mandatory
  dependency.

Related open/draft PRs that may further extend the native support:

- `#2093 <https://github.com/modelcontextprotocol/python-sdk/pull/2093>`_
  — enhanced inject logic (open).
- `#2133 <https://github.com/modelcontextprotocol/python-sdk/pull/2133>`_
  — enhanced extract logic (draft, depends on #2298).
- `#2132 <https://github.com/modelcontextprotocol/python-sdk/pull/2132>`_
  — richer CLIENT span attributes (draft, depends on #2298).

Migration plan
^^^^^^^^^^^^^^

Once ``mcp >= 2.x`` is released and the minimum supported version is raised:

- ``_send_request_wrapper`` (client-side inject) can be **removed**.
- The trace-context extract/attach portion of ``_server_handle_request_wrapper``
  can be **removed**.  The ``MCPRequestContext`` population
  (``jsonrpc.request.id``, ``network.transport``) should remain because the
  v2.x native spans (per #2381) only surface ``mcp.method.name`` and
  ``jsonrpc.request.id``; ``network.transport`` is not included.
  Re-evaluate as the upstream spans mature.
- ``_extract_carrier_from_meta`` can be **removed**.

A feature-detection guard (similar to ``_has_native_telemetry`` in the server
instrumentor) should be added so the wrappers gracefully become no-ops when
running against ``mcp >= 2.x``, allowing a wider version range.

Telemetry
---------

Spans:
~~~~~~

- ``initialize`` (client + server) — Session root span. Spans the full client/server session lifetime. All MCP operation spans are children.
- ``tools/call {tool_name}`` — Child span for each tool execution. ``SpanKind.CLIENT`` on client side, ``SpanKind.SERVER`` on server side.
- ``tools/list`` — Child span for tool listing.
- ``resources/read {uri}`` — Child span for resource reads.
- ``prompts/get {name}`` — Child span for prompt rendering.

Metrics:
~~~~~~~~

- ``mcp.client.operation.duration`` — Duration of client-side MCP operations (histogram).
- ``mcp.server.operation.duration`` — Duration of server-side MCP operations (histogram).
- ``mcp.client.session.duration`` — Client session duration, emitted when the ``initialize`` span ends (histogram).
- ``mcp.server.session.duration`` — Server session duration, emitted when the ``initialize`` span ends (histogram).
- ``mcp.tool.output.size`` — Size of tool output in bytes (histogram, custom SDOT attribute).

Events:
~~~~~~~

When content capture is enabled (``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``):

- ``gen_ai.tool.message`` — Tool input arguments and output results.


References
----------

- `FastMCP 3.x <https://github.com/gofastmcp/fastmcp>`_ (>= 3.0.0)
- `FastMCP 2.x <https://github.com/jlowin/fastmcp>`_ (<= 2.14.7)
- `Model Context Protocol <https://modelcontextprotocol.io/>`_
- `OpenTelemetry GenAI MCP Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/>`_
- `OpenTelemetry Project <https://opentelemetry.io/>`_
