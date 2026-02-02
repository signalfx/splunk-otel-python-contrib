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

"""OAuth2 authentication utilities for OpenTelemetry instrumented LLM applications."""

from opentelemetry.util.auth.oauth2_token_manager import OAuth2TokenManager
from opentelemetry.util.auth.version import __version__

__all__ = ["OAuth2TokenManager", "__version__"]

