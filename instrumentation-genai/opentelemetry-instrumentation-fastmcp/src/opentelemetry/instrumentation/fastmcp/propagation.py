"""
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

MCP trace context propagation.

Trace context (traceparent, tracestate) is propagated between MCP client and
server using the standard OTel Propagation API (``propagate.inject()`` /
``propagate.extract()``), which works over any transport (HTTP, gRPC, stdio).

The injection/extraction is handled directly by ``TransportInstrumentor``
(in ``transport_instrumentor.py``).  This module is intentionally minimal;
it exists only as a namespace placeholder for future propagation helpers.
"""
