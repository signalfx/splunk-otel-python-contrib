#!/bin/bash
set -e

echo "[INIT] Starting zero-code instrumented CrewAI application"
echo "[INIT] Service: $OTEL_SERVICE_NAME"
echo "[INIT] Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"
echo ""

# Run with opentelemetry-instrument (zero-code instrumentation)
opentelemetry-instrument python3 customer_support.py

EXIT_CODE=$?

# Give time for final telemetry export
echo ""
echo "[FLUSH] Waiting for telemetry export to complete..."
sleep 5

echo "[FLUSH] Telemetry export complete"
echo "[EXIT] Application exited with code: $EXIT_CODE"

exit $EXIT_CODE

