# Copyright Splunk Inc.
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

"""Tests for Bedrock Runtime utility helpers."""

from opentelemetry.instrumentation.bedrock import utils


def test_maybe_parse_json_skips_non_json_strings(monkeypatch):
    def fail_loads(_value):
        raise AssertionError("json.loads should not be called")

    monkeypatch.setattr(utils.json, "loads", fail_loads)

    assert utils.maybe_parse_json("x" * 100_000) == "x" * 100_000


def test_maybe_parse_json_parses_objects_and_arrays():
    assert utils.maybe_parse_json('{"city":"Paris"}') == {"city": "Paris"}
    assert utils.maybe_parse_json('[{"city":"Paris"}]') == [{"city": "Paris"}]
