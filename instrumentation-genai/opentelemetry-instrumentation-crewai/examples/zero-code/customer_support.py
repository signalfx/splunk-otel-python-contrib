"""
CrewAI Customer Support Example with OAuth2 LLM Integration (Zero-Code Instrumentation).

This example demonstrates:
- Using a custom LLM endpoint via LiteLLM with OAuth2 authentication
- Zero-code OpenTelemetry instrumentation for CrewAI

Run with:
    opentelemetry-instrument python customer_support.py

Environment Variables:
    LLM_CLIENT_ID: Your OAuth2 client ID
    LLM_CLIENT_SECRET: Your OAuth2 client secret
    LLM_TOKEN_URL: OAuth2 token endpoint
    LLM_BASE_URL: LLM endpoint base URL
    LLM_APP_KEY: Your app key (optional)
"""

import os
import json

# Set environment before any other imports
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"

# Now import CrewAI and utilities
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from util import OAuth2TokenManager


# =============================================================================
# LLM Configuration - OAuth2 Provider
# =============================================================================

# Optional app key for request tracking
LLM_APP_KEY = os.environ.get("LLM_APP_KEY") or os.environ.get("CISCO_APP_KEY")

# Initialize token manager (uses LLM_CLIENT_ID, LLM_CLIENT_SECRET env vars)
token_manager = OAuth2TokenManager()


def get_llm():
    """Create LLM instance with fresh OAuth2 token."""
    token = token_manager.get_token()

    # Configure LiteLLM with OAuth2 token in headers
    llm_kwargs = {
        "model": "openai/gpt-4o-mini",
        "base_url": OAuth2TokenManager.get_llm_base_url("gpt-4o-mini"),
        "api_key": "placeholder",  # Required by LiteLLM but we use api-key header
        "extra_headers": {
            "api-key": token,  # OAuth token in api-key header
        },
        "temperature": 0.7,
    }

    # Pass app key in user field if provided (required by some providers)
    if LLM_APP_KEY:
        llm_kwargs["user"] = json.dumps({"appkey": LLM_APP_KEY})

    return LLM(**llm_kwargs)


llm = get_llm()


# =============================================================================
# CrewAI Agents
# =============================================================================

support_agent = Agent(
    role="Senior Support Representative",
    goal="Be the most friendly and helpful support representative in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and are now working on providing "
        "support to {customer}, a super important customer for your company. "
        "You need to make sure that you provide the best support! "
        "Make sure to provide full complete answers, and make no assumptions."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
    cache=False,  # Disable agent caching to avoid embedding calls
)

support_quality_assurance_agent = Agent(
    role="Support Quality Assurance Specialist",
    goal="Get recognition for providing the best support quality assurance in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and are now working with your team "
        "on a request from {customer} ensuring that the support representative is "
        "providing the best support possible. "
        "You need to make sure that the support representative is providing full "
        "complete answers, and make no assumptions."
    ),
    llm=llm,
    verbose=False,
    cache=False,  # Disable agent caching to avoid embedding calls
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/en/concepts/crews"
)


# =============================================================================
# Tasks
# =============================================================================

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
    tools=[docs_scrape_tool],
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
    agent=support_quality_assurance_agent,
)


# =============================================================================
# Crew
# =============================================================================

crew = Crew(
    agents=[support_agent, support_quality_assurance_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
    verbose=False,
    memory=False,  # Disable memory to avoid OpenAI embedding calls
)

inputs = {
    "customer": "Splunk Olly for AI",
    "person": "Aditya Mehra",
    "inquiry": "I need help with setting up a Crew "
    "and kicking it off, specifically "
    "how can I add memory to my crew? "
    "Can you provide guidance?",
}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Refresh token and recreate LLM with fresh token
    fresh_token = token_manager.get_token()
    print(f"[AUTH] Token obtained (length: {len(fresh_token)})")

    # Recreate LLM with fresh token in headers
    llm = get_llm()

    # Update agents with fresh LLM
    support_agent.llm = llm
    support_quality_assurance_agent.llm = llm

    result = crew.kickoff(inputs=inputs)
    print("\n[SUCCESS] Crew execution completed")
    print(result)
