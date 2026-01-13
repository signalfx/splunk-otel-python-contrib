"""CrewAI customer support app (shared by manual + zero-code examples)."""

from __future__ import annotations

import json
import os
from typing import Any, Tuple

from crewai import Agent, Crew, LLM, Task
from crewai_tools import ScrapeWebsiteTool

from opentelemetry.util.oauth2_token_manager import OAuth2TokenManager


DEFAULT_INPUTS: dict[str, Any] = {
    "customer": "Splunk Olly for AI",
    "person": "Aditya Mehra",
    "inquiry": "I need help with setting up a Crew and kicking it off, specifically "
    "how can I add memory to my crew? Can you provide guidance?",
}


def create_cisco_llm(
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[LLM, OAuth2TokenManager, str]:
    """Create a Cisco Chat AI LLM instance with a fresh OAuth2 token."""

    app_key = os.environ.get("OAUTH2_APP_KEY")
    token_manager = OAuth2TokenManager()
    token = token_manager.get_token()

    llm = LLM(
        model=f"openai/{model}",
        base_url=OAuth2TokenManager.get_llm_base_url(model),
        api_key="placeholder",  # Required by LiteLLM but Cisco uses api-key header
        extra_headers={"api-key": token},
        user=json.dumps({"appkey": app_key}) if app_key else None,
        temperature=temperature,
    )
    return llm, token_manager, token


def build_customer_support_crew(
    llm: LLM,
) -> Tuple[Crew, Agent, Agent]:
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

    crew = Crew(
        agents=[support_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, quality_assurance_review],
        verbose=False,
        memory=False,  # Disable memory to avoid OpenAI embedding calls
    )
    return crew, support_agent, support_quality_assurance_agent


__all__ = [
    "DEFAULT_INPUTS",
    "build_customer_support_crew",
    "create_cisco_llm",
]
