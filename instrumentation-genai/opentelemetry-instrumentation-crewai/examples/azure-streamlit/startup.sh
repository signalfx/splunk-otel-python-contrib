#!/bin/sh
# =============================================================================
# startup.sh — Azure App Service entry point for Streamlit + OpenTelemetry
#
# Why this file exists:
#   App Service runs startup commands with /bin/sh (not bash).
#   Oryx builds a virtual environment called "antenv" and sets PYTHONPATH to
#   its site-packages, but does NOT add antenv/bin to PATH.
#   This means `opentelemetry-instrument` and `streamlit` are not found.
#
#   This script:
#     1. Activates antenv using POSIX-compatible `.` (not bash `source`)
#     2. Runs opentelemetry-instrument streamlit with the correct port
#
# Note on WEB_CONCURRENCY=1 (App Service setting):
#   Because this script provides a custom startup command, App Service bypasses
#   Gunicorn entirely and Streamlit runs under its own Tornado web server.
#   WEB_CONCURRENCY is a Gunicorn setting and has no effect here.
#   The actual fix for split traces was the sys.modules guard in app.py that
#   ensures initialize() is called only once per process, regardless of how
#   many times Streamlit reruns the script on user interactions.
#
# Startup command to set in App Service → Configuration → General settings:
#   sh startup.sh
# =============================================================================

set -e

echo "[startup] Python: $(python3 --version 2>&1)"
echo "[startup] Working directory: $(pwd)"

# Activate the Oryx-built virtual environment.
# antenv is always in the same directory as app.py after Oryx extraction.
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
echo "[startup] Service: ${OTEL_SERVICE_NAME:-crewai-customer-support-ui}"
echo "[startup] OTLP endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT:-not set}"

# OTel is initialised programmatically inside app.py (after the pysqlite3
# monkey-patch) so opentelemetry-instrument is not used here.
exec streamlit run app.py \
    --server.port "${APP_PORT}" \
    --server.headless true \
    --server.address 0.0.0.0
