#!/bin/bash
################################################################################
# OpenTelemetry Collector Startup Script
# 
# Purpose: Start the OTel Collector for macOS ARM64 (M1/M2/M3)
# 
# Usage: ./scripts/start_collector.sh
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COLLECTOR_BIN="$PROJECT_ROOT/bin/otelcol_darwin_arm64"
COLLECTOR_CONFIG="$PROJECT_ROOT/bin/otel-config-ai.yaml"
COLLECTOR_PID_FILE="/tmp/otelcol.pid"

echo "=========================================="
echo "🚀 Starting OpenTelemetry Collector"
echo "=========================================="

# Check if collector binary exists
if [ ! -f "$COLLECTOR_BIN" ]; then
    echo "❌ ERROR: Collector binary not found at: $COLLECTOR_BIN"
    echo ""
    echo "Please ensure the collector binary is present:"
    echo "  - Path: bin/otelcol_darwin_arm64"
    echo "  - Download from: https://github.com/open-telemetry/opentelemetry-collector-releases"
    exit 1
fi

# Check if config file exists
if [ ! -f "$COLLECTOR_CONFIG" ]; then
    echo "❌ ERROR: Collector config not found at: $COLLECTOR_CONFIG"
    echo ""
    echo "Please ensure the config file is present:"
    echo "  - Path: bin/otel-config-ai.yaml"
    exit 1
fi

# Make collector executable
chmod +x "$COLLECTOR_BIN"

# Check if collector is already running
if [ -f "$COLLECTOR_PID_FILE" ]; then
    OLD_PID=$(cat "$COLLECTOR_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Collector already running with PID: $OLD_PID"
        echo ""
        echo "Options:"
        echo "  1. Stop existing: ./scripts/stop_collector.sh"
        echo "  2. View logs: tail -f /tmp/otelcol.log"
        echo "  3. Check status: ps -p $OLD_PID"
        exit 0
    else
        echo "ℹ️  Removing stale PID file"
        rm -f "$COLLECTOR_PID_FILE"
    fi
fi

# Verify environment variables are set
if [ -z "$SPLUNK_ACCESS_TOKEN" ] || [ -z "$SPLUNK_REALM" ]; then
    echo "❌ ERROR: Required environment variables not set"
    echo ""
    echo "Please run: source scripts/setup_environment.sh"
    exit 1
fi

echo "✅ Environment variables configured"
echo "   - SPLUNK_REALM: $SPLUNK_REALM"
echo "   - SPLUNK_API_URL: $SPLUNK_API_URL"
echo "   - SPLUNK_INGEST_URL: $SPLUNK_INGEST_URL"
echo ""

# Start collector in background
echo "🔄 Starting collector..."
echo "   Config: $COLLECTOR_CONFIG"
nohup "$COLLECTOR_BIN" --config="$COLLECTOR_CONFIG" > /tmp/otelcol.log 2>&1 &
COLLECTOR_PID=$!

# Save PID
echo "$COLLECTOR_PID" > "$COLLECTOR_PID_FILE"

# Wait a moment and verify it's running
sleep 2

if ps -p "$COLLECTOR_PID" > /dev/null 2>&1; then
    echo "✅ Collector started successfully!"
    echo "   - PID: $COLLECTOR_PID"
    echo "   - Logs: /tmp/otelcol.log"
    echo "   - OTLP gRPC: localhost:4317"
    echo "   - OTLP HTTP: localhost:4318"
    echo ""
    echo "Monitor logs: tail -f /tmp/otelcol.log"
    echo "Stop collector: ./scripts/stop_collector.sh"
else
    echo "❌ ERROR: Collector failed to start"
    echo ""
    echo "Check logs: cat /tmp/otelcol.log"
    rm -f "$COLLECTOR_PID_FILE"
    exit 1
fi

echo "=========================================="
echo "✅ Collector ready to receive telemetry"
echo "=========================================="
