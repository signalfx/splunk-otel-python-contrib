#!/bin/bash
set -e

echo "[INIT] Starting CrewAI Financial Assistant"
echo "[INIT] Service: ${OTEL_SERVICE_NAME:-financial-trading-crew}"
echo "[INIT] Endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4317}"
echo "[INIT] Manual Instrumentation: ${OTEL_MANUAL_INSTRUMENTATION:-true}"
echo ""

# Determine instrumentation mode
if [ "${OTEL_MANUAL_INSTRUMENTATION:-true}" = "true" ]; then
    echo "[INIT] Running with manual instrumentation"
    python3 financial_assistant.py
    EXIT_CODE=$?
else
    echo "[INIT] Running with zero-code instrumentation"
    opentelemetry-instrument python3 financial_assistant.py
    EXIT_CODE=$?
    
    # Force flush telemetry providers for zero-code mode
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
fi

echo "[EXIT] Application exited with code: $EXIT_CODE"
exit $EXIT_CODE

