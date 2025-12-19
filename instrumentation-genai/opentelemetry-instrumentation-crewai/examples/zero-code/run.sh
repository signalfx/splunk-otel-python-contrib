#!/bin/bash
set -e

echo "[INIT] Starting zero-code instrumented CrewAI application"
echo "[INIT] Service: $OTEL_SERVICE_NAME"
echo "[INIT] Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"
echo ""

# Run with opentelemetry-instrument (zero-code instrumentation)
# The --force_flush flag ensures spans are exported before exit
opentelemetry-instrument python3 customer_support.py

EXIT_CODE=$?

# Force flush telemetry providers before exit
echo ""
echo "[FLUSH] Force flushing telemetry providers..."
python3 -c "
from opentelemetry import trace, metrics
import time

# Flush traces
tp = trace.get_tracer_provider()
if hasattr(tp, 'force_flush'):
    print('[FLUSH] Flushing traces (timeout=30s)')
    tp.force_flush(timeout_millis=30000)

# Flush metrics  
mp = metrics.get_meter_provider()
if hasattr(mp, 'force_flush'):
    print('[FLUSH] Flushing metrics (timeout=30s)')
    mp.force_flush(timeout_millis=30000)

# Small delay for network buffers
time.sleep(2)
print('[FLUSH] Telemetry flush complete')
"

echo "[EXIT] Application exited with code: $EXIT_CODE"

exit $EXIT_CODE

