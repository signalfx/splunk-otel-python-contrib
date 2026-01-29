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

from opentelemetry.metrics import Histogram, Meter

# Bucket boundaries per OpenTelemetry semantic conventions:
# https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md
_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]

_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]


class Instruments:
    """
    Manages OpenTelemetry metrics instruments for GenAI telemetry.
    """

    def __init__(self, meter: Meter):
        self.operation_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI client operations",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        self.token_usage_histogram: Histogram = meter.create_histogram(
            name="gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
        )
        # Agentic AI metrics
        # Use the same bucket boundaries as operation duration since all duration metrics
        # are measured in seconds and should follow the same distribution patterns
        self.workflow_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.workflow.duration",
            unit="s",
            description="Duration of GenAI workflows",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        self.agent_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.agent.duration",
            unit="s",
            description="Duration of agent operations",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        self.retrieval_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.retrieval.duration",
            unit="s",
            description="Duration of retrieval operations",
        )
        # MCP (Model Context Protocol) metrics
        # Per OTel semconv: https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp
        # Bucket boundaries: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
        self.mcp_client_operation_duration: Histogram = meter.create_histogram(
            name="mcp.client.operation.duration",
            unit="s",
            description="Duration of MCP request or notification as observed on "
            "the sender from the time it was sent until response or ack is received",
        )
        self.mcp_server_operation_duration: Histogram = meter.create_histogram(
            name="mcp.server.operation.duration",
            unit="s",
            description="MCP request or notification duration as observed on "
            "the receiver from the time it was received until result or ack is sent",
        )
        self.mcp_client_session_duration: Histogram = meter.create_histogram(
            name="mcp.client.session.duration",
            unit="s",
            description="Duration of the MCP session as observed on the MCP client",
        )
        self.mcp_server_session_duration: Histogram = meter.create_histogram(
            name="mcp.server.session.duration",
            unit="s",
            description="Duration of the MCP session as observed on the MCP server",
        )
        # Custom metric: Track tool output size (impacts LLM token usage when passed as context)
        self.mcp_tool_output_size: Histogram = meter.create_histogram(
            name="mcp.tool.output.size",
            unit="{byte}",
            description="Size of the tool call output in bytes. "
            "This output typically becomes part of the LLM input context.",
        )
