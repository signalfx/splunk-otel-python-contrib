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

"""Tests for FastMCPInstrumentor main class."""

from unittest.mock import patch, MagicMock

from opentelemetry.instrumentation.fastmcp import FastMCPInstrumentor
from opentelemetry.instrumentation.fastmcp.version import __version__


class TestFastMCPInstrumentor:
    """Tests for FastMCPInstrumentor class."""

    def test_version(self):
        """Test version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies are defined."""
        instrumentor = FastMCPInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert len(deps) > 0
        assert any("fastmcp" in dep for dep in deps)

    def test_init(self):
        """Test instrumentor initialization."""
        instrumentor = FastMCPInstrumentor()
        assert instrumentor._telemetry_handler is None
        assert instrumentor._server_instrumentor is None
        assert instrumentor._client_instrumentor is None

    def test_init_with_exception_logger(self):
        """Test instrumentor initialization with exception logger."""
        logger = MagicMock()
        instrumentor = FastMCPInstrumentor(exception_logger=logger)
        assert instrumentor._exception_logger == logger

    @patch(
        "opentelemetry.instrumentation.fastmcp.instrumentation.get_telemetry_handler"
    )
    @patch("opentelemetry.instrumentation.fastmcp.instrumentation.ServerInstrumentor")
    @patch("opentelemetry.instrumentation.fastmcp.instrumentation.ClientInstrumentor")
    def test_instrument(
        self,
        mock_client_instrumentor_class,
        mock_server_instrumentor_class,
        mock_get_handler,
    ):
        """Test instrumentation is applied correctly."""
        mock_handler = MagicMock()
        mock_get_handler.return_value = mock_handler

        mock_server_inst = MagicMock()
        mock_client_inst = MagicMock()
        mock_server_instrumentor_class.return_value = mock_server_inst
        mock_client_instrumentor_class.return_value = mock_client_inst

        instrumentor = FastMCPInstrumentor()
        instrumentor._instrument(tracer_provider=MagicMock())

        # Verify handler was obtained
        mock_get_handler.assert_called_once()

        # Verify instrumentors were created and instrument() called
        mock_server_instrumentor_class.assert_called_once_with(mock_handler)
        mock_client_instrumentor_class.assert_called_once_with(mock_handler)
        mock_server_inst.instrument.assert_called_once()
        mock_client_inst.instrument.assert_called_once()

    @patch(
        "opentelemetry.instrumentation.fastmcp.instrumentation.is_instrumentation_enabled"
    )
    def test_instrument_disabled(self, mock_is_enabled):
        """Test instrumentation is not applied when disabled."""
        mock_is_enabled.return_value = False

        instrumentor = FastMCPInstrumentor()
        instrumentor._instrument()

        assert instrumentor._telemetry_handler is None
        assert instrumentor._server_instrumentor is None

    @patch(
        "opentelemetry.instrumentation.fastmcp.instrumentation.get_telemetry_handler"
    )
    @patch("opentelemetry.instrumentation.fastmcp.instrumentation.ServerInstrumentor")
    @patch("opentelemetry.instrumentation.fastmcp.instrumentation.ClientInstrumentor")
    def test_uninstrument(
        self,
        mock_client_instrumentor_class,
        mock_server_instrumentor_class,
        mock_get_handler,
    ):
        """Test uninstrumentation."""
        mock_handler = MagicMock()
        mock_get_handler.return_value = mock_handler

        mock_server_inst = MagicMock()
        mock_client_inst = MagicMock()
        mock_server_instrumentor_class.return_value = mock_server_inst
        mock_client_instrumentor_class.return_value = mock_client_inst

        instrumentor = FastMCPInstrumentor()
        instrumentor._instrument()
        instrumentor._uninstrument()

        mock_server_inst.uninstrument.assert_called_once()
        mock_client_inst.uninstrument.assert_called_once()


class TestFastMCPInstrumentorImports:
    """Tests for module imports."""

    def test_public_api(self):
        """Test public API exports."""
        from opentelemetry.instrumentation.fastmcp import (
            FastMCPInstrumentor,
            __version__,
        )

        assert FastMCPInstrumentor is not None
        assert __version__ is not None
