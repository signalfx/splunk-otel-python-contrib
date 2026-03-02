"""
Streamlit UI for the CrewAI Customer Support example (Azure OpenAI).

Launch with zero-code telemetry (recommended):
    bash run.sh          (local)
    sh startup.sh        (Azure App Service)

Launch without telemetry (development only):
    streamlit run app.py

Why programmatic OTel initialisation (not opentelemetry-instrument):
  On Azure App Service the system sqlite3 is < 3.35.0. chromadb (pulled in
  by crewai-tools) raises RuntimeError at import time. When
  opentelemetry-instrument wraps the process it tries to load the CrewAI
  instrumentor *before* app.py runs, so the pysqlite3 monkey-patch comes
  too late and the CrewAI patch silently fails — leaving each agent without
  a parent span, which produces split traces.

  By calling initialize() here we control the order:
    1. pysqlite3 monkey-patch   ← sqlite3 replaced before any crewai import
    2. initialize()             ← CrewAI instrumentor now loads successfully
    3. streamlit / app imports  ← all instrumented correctly

Why the sys.modules guard:
  Streamlit re-executes app.py on every user interaction (button click,
  form submit). Without a guard, initialize() would be called on every
  rerun, re-applying instrumentor patches on top of existing ones and
  producing N× duplicate spans for each crew/LLM call.
  sys.modules persists for the lifetime of the process (unlike module-level
  variables which are reset each rerun), making it the right place for a
  once-per-process flag.
"""

# Step 1 — patch sqlite3 FIRST, before any crewai/chromadb import.
# Azure App Service ships sqlite3 < 3.35.0; pysqlite3-binary bundles 3.45+.
# This must happen before initialize() below, which imports crewai.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # macOS / modern Linux already has a recent enough sqlite3

import os

# Step 2 — initialise OTel exactly once per process.
# Streamlit reruns this script on every interaction, so we use sys.modules
# as a process-level flag that survives reruns (module-level vars do not).
_OTEL_INIT_KEY = "__crewai_otel_initialized__"
if _OTEL_INIT_KEY not in sys.modules:
    sys.modules[_OTEL_INIT_KEY] = True  # set flag before initialize() in case of error
    try:
        from opentelemetry.instrumentation.auto_instrumentation import initialize

        initialize()
    except Exception:
        pass  # SDK not installed — safe to ignore

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Load .env file so credentials are available when running locally.
# In production, set env vars directly in your shell or container.
load_dotenv()

from customer_support_azure import run_crew  # noqa: E402 — import after load_dotenv

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="CrewAI Customer Support",
    page_icon="🤝",
    layout="centered",
)

# =============================================================================
# Header
# =============================================================================

st.title("CrewAI Customer Support")
st.caption(
    "Powered by **Azure OpenAI** · Two-agent crew (Support Rep + QA Specialist) "
    "· Memory enabled · Instrumented with OpenTelemetry"
)

st.divider()

# =============================================================================
# Sidebar — connection info
# =============================================================================

with st.sidebar:
    st.header("Configuration")

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "not set")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "not set")
    embedding = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "not set")
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "not set")
    service_name = os.environ.get("OTEL_SERVICE_NAME", "not set")

    st.markdown(f"**Chat deployment:** `{deployment}`")
    st.markdown(f"**Embedding deployment:** `{embedding}`")
    st.markdown(f"**Endpoint:** `{endpoint}`")

    st.divider()
    st.subheader("Telemetry")
    st.markdown(f"**Service name:** `{service_name}`")
    st.markdown(f"**OTLP endpoint:** `{otel_endpoint}`")

    st.divider()
    st.info(
        "Traces from this UI, CrewAI agent steps, and Azure OpenAI calls "
        "are all exported to the configured OTLP endpoint when launched via `run.sh`."
    )

# =============================================================================
# Input form
# =============================================================================

with st.form("support_form"):
    customer = st.text_input(
        "Customer Name",
        placeholder="e.g. Acme Corp",
        help="The company or customer name passed to the crew agents.",
    )

    person = st.text_input(
        "Your Name",
        value="User",
        help="Your name — shown in the agent task context.",
    )

    inquiry = st.text_area(
        "Docs Query / Inquiry",
        placeholder=(
            "e.g. How do I add memory to my CrewAI crew? "
            "What embedding providers are supported?"
        ),
        height=150,
        help="The support question or docs query the crew will research and answer.",
    )

    submitted = st.form_submit_button(
        "Submit", use_container_width=True, type="primary"
    )

# =============================================================================
# Execution
# =============================================================================

if submitted:
    if not customer.strip():
        st.error("Please enter a customer name.")
    elif not inquiry.strip():
        st.error("Please enter a query or inquiry.")
    else:
        st.divider()
        st.subheader("Running crew…")

        status_box = st.empty()
        status_box.info(
            f"**Customer:** {customer}  \n"
            f"**Person:** {person}  \n"
            f"**Inquiry:** {inquiry[:200]}{'…' if len(inquiry) > 200 else ''}"
        )

        with st.spinner("Agents are working on your inquiry — this may take a minute…"):
            try:
                response = run_crew(
                    customer=customer.strip(),
                    inquiry=inquiry.strip(),
                    person=person.strip() or "User",
                )
                status_box.empty()
                st.success("Crew completed successfully.")
                st.divider()
                st.subheader("Response")
                st.markdown(response)

            except EnvironmentError as env_err:
                status_box.empty()
                st.error(f"**Configuration error:** {env_err}")
                st.info(
                    "Copy `.env.example` to `.env` in this directory, "
                    "fill in your Azure OpenAI credentials, then restart the app."
                )
            except Exception as exc:
                status_box.empty()
                st.error(f"**Crew execution failed:** {exc}")
                with st.expander("Traceback"):
                    import traceback

                    st.code(traceback.format_exc(), language="python")
