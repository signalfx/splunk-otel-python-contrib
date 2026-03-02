"""
CrewAI Customer Support — Azure OpenAI variant.

This module provides the crew logic used by the Streamlit UI (app.py).
It is also runnable directly from the command line for quick testing.

Key differences from the parent customer_support.py:
  - Auth: Azure API key (AZURE_OPENAI_API_KEY) instead of OAuth2
  - Model prefix: azure/<deployment> instead of openai/<model>
  - memory=True with Azure OpenAI embedder
  - No manual telemetry setup — always uses zero-code instrumentation via
    ``opentelemetry-instrument streamlit run app.py`` (see run.sh)
  - Exposes run_crew() for Streamlit to call

Instrumentation:
  Zero-code mode only:  opentelemetry-instrument streamlit run app.py
  (See run.sh for the full launch command with OTEL env vars.)

Environment Variables:
  AZURE_OPENAI_API_KEY:             Azure OpenAI API key
  AZURE_OPENAI_ENDPOINT:            e.g. https://<resource>.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT:          Chat completion deployment name (e.g. gpt-4)
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Embedding deployment (e.g. text-embedding-ada-002)
  AZURE_OPENAI_API_VERSION:         API version (default: 2024-02-15-preview)
  CREWAI_DISABLE_TELEMETRY:         Disable CrewAI's own telemetry (default: true)
  OTEL_CONSOLE_OUTPUT:              Print spans to console as well (default: false)
"""

import os
import sys
from opentelemetry import trace as otel_trace

# =============================================================================
# Environment Defaults
# =============================================================================

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event,splunk")

# =============================================================================
# Constants
# =============================================================================

COMPANY_URL = "https://crewai.com"
DOCS_URL = "https://docs.crewai.com/"


# =============================================================================
# Utility Helpers
# =============================================================================


def _log(tag: str, message: str, stderr: bool = False) -> None:
    """Consistent log output."""
    out = sys.stderr if stderr else sys.stdout
    print(f"[{tag}] {message}", file=out, flush=True)


def _require_env(key: str) -> str:
    """Return env var value, raising clearly if missing."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your Azure credentials."
        )
    return value


# =============================================================================
# LLM Configuration
# =============================================================================


def create_llm():
    """Create a CrewAI LLM backed by Azure OpenAI (gpt-4.1-nano).

    CrewAI 1.6+ uses the azure-ai-inference native provider when the model
    has an ``azure/`` prefix.  That provider expects:
      - ``endpoint``: full deployment URL ending at the deployment name
      - ``api_key``:  Azure API key
      - ``api_version``: Azure REST API version

    For cognitiveservices.azure.com resources the SDK constructs:
        {endpoint}/chat/completions?api-version={api_version}
    which matches the Azure OpenAI REST API path.
    """
    from crewai import LLM

    deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    base = _require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    # Build the full deployment endpoint that the azure-ai-inference SDK expects.
    # ChatCompletionsClient appends /chat/completions to this URL.
    endpoint = f"{base}/openai/deployments/{deployment}"

    _log("LLM", f"Azure deployment: {deployment}  endpoint: {endpoint}  api_version: {api_version}")
    return LLM(
        model=f"azure/{deployment}",
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        temperature=0.7,
    )


# =============================================================================
# Crew Components
# =============================================================================


def create_tools():
    """Create the ScrapeWebsiteTool pointed at CrewAI docs."""
    from crewai_tools import ScrapeWebsiteTool

    tool = ScrapeWebsiteTool(website_url=DOCS_URL)
    _log("TOOLS", f"ScrapeWebsiteTool initialised for {DOCS_URL}")
    return [tool]


def create_agents(llm, tools: list):
    """Create the two-agent crew: Support Rep + QA Specialist."""
    from crewai import Agent

    common = {"llm": llm, "verbose": False, "cache": False}

    support_agent = Agent(
        role="Senior Support Representative",
        goal="Be the most friendly and helpful support representative in your team",
        backstory=(
            f"You work at crewAI ({COMPANY_URL}) and are now working on providing "
            "support to {customer}, a super important customer for your company. "
            "You need to make sure that you provide the best support! "
            "Make sure to provide full complete answers, and make no assumptions. "
            "Use the documentation scraper tool to look up accurate information from "
            "the official CrewAI docs when needed."
        ),
        allow_delegation=False,
        tools=tools,
        **common,
    )

    qa_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Get recognition for providing the best support quality assurance in your team",
        backstory=(
            f"You work at crewAI ({COMPANY_URL}) and are now working with your team "
            "on a request from {customer} ensuring that the support representative is "
            "providing the best support possible. "
            "You need to make sure that the support representative is providing full "
            "complete answers, and make no assumptions."
        ),
        **common,
    )

    return support_agent, qa_agent


def create_tasks(support_agent, qa_agent):
    """Create the inquiry resolution and QA review tasks."""
    from crewai import Task

    inquiry_resolution = Task(
        description=(
            "{customer} just reached out with a super important ask:\n"
            "{inquiry}\n\n"
            "{person} from {customer} is the one that reached out. "
            "Make sure to use everything you know to provide the best support possible. "
            "You must strive to provide a complete and accurate response to the customer's inquiry."
        ),
        expected_output=(
            "A detailed, informative response to the customer's inquiry that addresses "
            "all aspects of their question. "
            "The response should include references to everything you used to find the answer, "
            "including external data or solutions. "
            "Ensure the answer is complete, leaving no questions unanswered, "
            "and maintain a helpful and friendly tone throughout."
        ),
        agent=support_agent,
    )

    quality_assurance_review = Task(
        description=(
            "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
            "Ensure that the answer is comprehensive, accurate, and adheres to the "
            "high-quality standards expected for customer support. "
            "Verify that all parts of the customer's inquiry have been addressed "
            "thoroughly, with a helpful and friendly tone. "
            "Check for references and sources used to find the information, "
            "ensuring the response is well-supported and leaves no questions unanswered."
        ),
        expected_output=(
            "A final, detailed, and informative response ready to be sent to the customer. "
            "This response should fully address the customer's inquiry, incorporating all "
            "relevant feedback and improvements. "
            "Don't be too formal, we are a chill and cool company "
            "but maintain a professional and friendly tone throughout."
        ),
        agent=qa_agent,
    )

    return inquiry_resolution, quality_assurance_review


def create_crew(agents: list, tasks: list):
    """
    Create the crew with memory enabled and Azure OpenAI embedder.

    memory=True requires the embedder to be configured explicitly when using
    Azure OpenAI so that CrewAI's memory store can compute embeddings via the
    same Azure resource.
    """
    from crewai import Crew

    embedding_deployment = _require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    api_base = _require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    # Embeddings use a separate API version from chat completions
    embedding_api_version = os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"]

    _log(
        "CREW",
        f"Memory enabled — Azure embedder: {embedding_deployment} "
        f"(api_version={embedding_api_version})",
    )

    return Crew(
        name="customer_support_crew",
        agents=agents,
        tasks=tasks,
        verbose=False,
        memory=True,
        embedder={
            "provider": "azure",           # crewai 1.6+ uses "azure" (not "azure_openai")
            "config": {
                "deployment_id": embedding_deployment,  # required field in crewai 1.6+
                "model_name": embedding_deployment,     # optional display name
                "api_key": api_key,
                "api_base": api_base,
                "api_version": embedding_api_version,
            },
        },
    )


# =============================================================================
# Public API — called by app.py
# =============================================================================


def run_crew(customer: str, inquiry: str, person: str = "User") -> str:
    """
    Execute the customer support crew and return the final response text.

    Args:
        customer: Company / customer name (e.g. "Acme Corp").
        inquiry:  The docs query or support question.
        person:   Name of the person reaching out (shown in task context).

    Returns:
        The QA-reviewed response string from the crew.

    Raises:
        EnvironmentError: If required Azure env vars are not set.
        Exception: Re-raised from crew execution on failure.
    """
    _log("START", f"Customer: {customer} | Person: {person}")
    _log("START", f"Inquiry: {inquiry[:120]}{'...' if len(inquiry) > 120 else ''}")

    llm = create_llm()
    tools = create_tools()
    support_agent, qa_agent = create_agents(llm, tools)
    inquiry_task, qa_task = create_tasks(support_agent, qa_agent)
    crew = create_crew([support_agent, qa_agent], [inquiry_task, qa_task])

    inputs = {
        "customer": customer,
        "person": person,
        "inquiry": inquiry,
    }

    # Establish a root span before crew.kickoff() spawns agent threads.
    # This ensures all agent spans share one trace regardless of whether
    # Gunicorn's pre-fork model has reset the process-level context.
    tracer = otel_trace.get_tracer("crewai-customer-support")
    with tracer.start_as_current_span(
        "crewai.crew.kickoff",
        attributes={
            "crewai.crew.name": "customer_support_crew",
            "customer.name": customer,
            "customer.person": person,
        },
    ):
        _log("CREW", "Kicking off crew...")
        result = crew.kickoff(inputs=inputs)
        _log("CREW", "Crew execution completed successfully")

    return str(result)


# =============================================================================
# CLI entry point (for quick testing without the UI)
# =============================================================================


def _main() -> int:
    """Run the crew from the command line with sample inputs."""
    try:
        response = run_crew(
            customer="Splunk Observability for AI",
            person="Aditya Mehra",
            inquiry="I need help with setting up a Crew and kicking it off, specifically how can I add memory to my crew? Can you provide guidance?",
        )
        print("\n" + "=" * 80)
        print("CREW RESPONSE")
        print("=" * 80)
        print(response)
        return 0
    except Exception as exc:
        _log("ERROR", str(exc), stderr=True)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(_main())


# =============================================================================
# Expected Trace Structure (zero-code mode)
# =============================================================================
# gen_ai.workflow (customer_support_crew)
# ├── gen_ai.step (Inquiry Resolution)
# │   └── invoke_agent (Senior Support Representative)
# │       ├── chat (Azure OpenAI)        ← OpenAIInstrumentor v2
# │       └── tool (Read website content)
# └── gen_ai.step (Quality Assurance Review)
#     └── invoke_agent (QA Specialist)
#         └── chat (Azure OpenAI)        ← OpenAIInstrumentor v2
# =============================================================================
