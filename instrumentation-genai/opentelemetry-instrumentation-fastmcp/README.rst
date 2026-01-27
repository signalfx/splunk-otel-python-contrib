OpenTelemetry FastMCP Instrumentation
=====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/splunk-otel-instrumentation-fastmcp.svg
   :target: https://pypi.org/project/splunk-otel-instrumentation-fastmcp/

This library provides automatic instrumentation for `FastMCP <https://github.com/jlowin/fastmcp>`_,
a Python library for building Model Context Protocol (MCP) servers.

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

Server-side:
~~~~~~~~~~~~

- FastMCP server initialization
- Tool execution via ``ToolManager.call_tool``

Client-side:
~~~~~~~~~~~~

- FastMCP client session lifecycle
- Tool calls and listings

Transport-level:
~~~~~~~~~~~~~~~~

- Automatic trace context propagation via ``_meta`` field
- Works for all MCP transports: stdio, SSE, streamable-http

Trace Context Propagation
-------------------------

The instrumentation automatically propagates W3C TraceContext (traceparent, tracestate)
between MCP client and server processes. This enables distributed tracing across
process boundaries:

- Client spans and server spans share the same ``trace_id``
- Server tool execution spans are children of client tool call spans
- No code changes required in your MCP server or client

Telemetry
---------

Spans:
~~~~~~

- ``mcp.server`` - Parent span for server operations
- ``{tool_name}.tool`` - Child span for each tool execution
- ``mcp.client`` - Parent span for client session

Metrics:
~~~~~~~~

- ``gen_ai.mcp.tool.duration`` - Duration of tool executions (histogram)

Events:
~~~~~~~

When content capture is enabled:

- ``mcp.tool.input`` - Tool arguments
- ``mcp.tool.output`` - Tool results


References
----------

- `FastMCP <https://github.com/jlowin/fastmcp>`_
- `Model Context Protocol <https://modelcontextprotocol.io/>`_
- `OpenTelemetry Project <https://opentelemetry.io/>`_
