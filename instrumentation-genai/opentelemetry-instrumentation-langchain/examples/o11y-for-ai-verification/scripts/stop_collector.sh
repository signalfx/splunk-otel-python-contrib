#!/bin/bash
################################################################################
# OpenTelemetry Collector Stop Script
################################################################################

set -e

COLLECTOR_PID_FILE="/tmp/otelcol.pid"

echo "=========================================="
echo "🛑 Stopping OpenTelemetry Collector"
echo "=========================================="

if [ ! -f "$COLLECTOR_PID_FILE" ]; then
    echo "ℹ️  No PID file found - collector may not be running"
    exit 0
fi

PID=$(cat "$COLLECTOR_PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "🔄 Stopping collector (PID: $PID)..."
    kill "$PID"
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  Collector didn't stop gracefully, forcing..."
        kill -9 "$PID"
    fi
    
    echo "✅ Collector stopped"
else
    echo "ℹ️  Collector not running (stale PID file)"
fi

rm -f "$COLLECTOR_PID_FILE"
echo "=========================================="
