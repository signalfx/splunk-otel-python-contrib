#!/bin/sh
# =============================================================================
# startup.sh — Azure App Service entry point for FastAPI + Gunicorn + OTel
#
# Why this file exists:
# App Service runs startup commands with /bin/sh (not bash).
# Oryx builds a virtual environment called "antenv" and sets PYTHONPATH to
# its site-packages, but does NOT add antenv/bin to PATH.
# This means `opentelemetry-instrument` and `gunicorn` are not found without
# first activating the virtual environment.
#
# This script:
# 1. Activates antenv using POSIX-compatible `.` (not bash `source`)
# 2. Runs opentelemetry-instrument gunicorn with Uvicorn workers
#
# Startup command to set in App Service → Configuration → General settings:
#   sh startup.sh
# =============================================================================

set -e

echo "[startup] Python: $(python3 --version 2>&1)"
echo "[startup] Working directory: $(pwd)"

# Activate the Oryx-built virtual environment.
# antenv is always co-located with app.py after Oryx extracts the zip.
if [ -f "antenv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . antenv/bin/activate
    echo "[startup] Activated antenv at $(pwd)/antenv"
else
    echo "[startup] WARNING: antenv/bin/activate not found — falling back to system Python"
    echo "[startup] Contents of current directory:"
    ls -la
fi

# App Service sets PORT; fall back to 8000 (App Service default for custom apps).
APP_PORT="${PORT:-8000}"
echo "[startup] Launching on port ${APP_PORT}"
echo "[startup] Service: ${OTEL_SERVICE_NAME:-multi-agent-travel-planner-azure}"
echo "[startup] OTLP endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT:-not set}"
echo "[startup] Environment: ${OTEL_RESOURCE_ATTRIBUTES:-not set}"

# OTel is initialised programmatically inside app.py via initialize() with a
# sys.modules guard (see the top of app.py for the full explanation).
# This runs post-fork in each Gunicorn worker, giving every worker its own
# fresh PeriodicExportingMetricReader thread — fixing the silent metric drop
# caused by the opentelemetry-instrument wrapper + --preload pattern.
#
# UvicornWorker: required to serve the FastAPI ASGI application.
# --timeout 301: slightly above the default 300s to allow the LLM pipeline
# to complete before Gunicorn kills a slow worker.
exec gunicorn \
    -w 1 \
    -k uvicorn.workers.UvicornWorker \
    app:app \
    --access-logfile "-" \
    --timeout 301 \
    --bind "0.0.0.0:${APP_PORT}"
