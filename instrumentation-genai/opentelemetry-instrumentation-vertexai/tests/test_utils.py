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


from google.cloud.aiplatform_v1.types import (
    content,
    tool,
)
from google.cloud.aiplatform_v1beta1.types import (
    content as content_v1beta1,
)

from opentelemetry.instrumentation.vertexai.utils import (
    _map_finish_reason,
    convert_content_to_message_parts,
    extract_tool_definitions,
    get_server_attributes,
)


def test_get_server_attributes() -> None:
    # without port
    assert get_server_attributes("us-central1-aiplatform.googleapis.com") == {
        "server.address": "us-central1-aiplatform.googleapis.com",
        "server.port": 443,
    }

    # with port
    assert get_server_attributes(
        "us-central1-aiplatform.googleapis.com:5432"
    ) == {
        "server.address": "us-central1-aiplatform.googleapis.com",
        "server.port": 5432,
    }


def test_map_finish_reason():
    for Enum in (
        content.Candidate.FinishReason,
        content_v1beta1.Candidate.FinishReason,
    ):
        for finish_reason, expect in [
            # Handled mappings
            (Enum.FINISH_REASON_UNSPECIFIED, "error"),
            (Enum.OTHER, "error"),
            (Enum.STOP, "stop"),
            (Enum.MAX_TOKENS, "length"),
            # Preserve vertex enum value
            (Enum.BLOCKLIST, "BLOCKLIST"),
            (Enum.MALFORMED_FUNCTION_CALL, "MALFORMED_FUNCTION_CALL"),
            (Enum.PROHIBITED_CONTENT, "PROHIBITED_CONTENT"),
            (Enum.RECITATION, "RECITATION"),
            (Enum.SAFETY, "SAFETY"),
            (Enum.SPII, "SPII"),
        ]:
            assert _map_finish_reason(finish_reason) == expect


def test_convert_content_function_call():
    """function_call parts are mapped to ToolCall message parts."""
    c = content.Content(
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": "get_weather",
                        "args": {"location": "New Delhi"},
                    }
                }
            ],
        }
    )
    parts = convert_content_to_message_parts(c)
    assert len(parts) == 1
    tc = parts[0]
    assert tc.type == "tool_call"
    assert tc.name == "get_weather"
    assert tc.arguments == {"location": "New Delhi"}
    assert tc.id == "get_weather_0"


def test_convert_content_mixed_parts():
    """Text, function_call, and function_response parts are all mapped."""
    c = content.Content(
        {
            "role": "model",
            "parts": [
                {"text": "intro"},
                {
                    "function_call": {
                        "name": "search",
                        "args": {"q": "hello"},
                    }
                },
                {
                    "function_response": {
                        "name": "search",
                        "response": {"answer": "world"},
                    }
                },
            ],
        }
    )
    parts = convert_content_to_message_parts(c)
    assert len(parts) == 3
    assert parts[0].type == "text"
    assert parts[0].content == "intro"
    assert parts[1].type == "tool_call"
    assert parts[1].name == "search"
    assert parts[2].type == "tool_call_response"
    assert parts[2].response == {"answer": "world"}


def test_extract_tool_definitions():
    """extract_tool_definitions converts Tool protos to dicts."""
    t = tool.Tool(
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {
                            "loc": {"type_": "STRING"},
                        },
                    },
                },
                {
                    "name": "get_time",
                    "description": "Get time",
                },
            ]
        }
    )
    result = extract_tool_definitions([t])
    assert len(result) == 2
    assert result[0]["name"] == "get_weather"
    assert result[0]["description"] == "Get weather"
    assert "properties" in result[0]["parameters"]
    assert result[1]["name"] == "get_time"
    assert result[1]["description"] == "Get time"


def test_extract_tool_definitions_none():
    """extract_tool_definitions returns empty list for None input."""
    assert extract_tool_definitions(None) == []
