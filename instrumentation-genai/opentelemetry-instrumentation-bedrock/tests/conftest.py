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

"""Test fixtures for Bedrock Runtime instrumentation."""

import pytest


class StubTelemetryHandler:
    """Minimal telemetry handler for wrapper unit tests."""

    def __init__(self):
        self.started_llm = []
        self.stopped_llm = []
        self.failed_llm = []

    def start_llm(self, invocation):
        self.started_llm.append(invocation)
        return invocation

    def stop_llm(self, invocation):
        self.stopped_llm.append(invocation)
        return invocation

    def fail_llm(self, invocation, error):
        self.failed_llm.append((invocation, error))
        return invocation


class FakeServiceModel:
    def __init__(self, service_name="bedrock-runtime"):
        self.service_name = service_name


class FakeMeta:
    def __init__(self, service_name="bedrock-runtime"):
        self.service_model = FakeServiceModel(service_name)
        self.endpoint_url = "https://bedrock-runtime.us-west-2.amazonaws.com"


class FakeClient:
    def __init__(self, service_name="bedrock-runtime"):
        self.meta = FakeMeta(service_name)


class FakeStream:
    def __init__(self, events):
        self._events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def close(self):
        self.closed = True


@pytest.fixture
def stub_handler():
    return StubTelemetryHandler()


@pytest.fixture
def fake_client():
    return FakeClient()
