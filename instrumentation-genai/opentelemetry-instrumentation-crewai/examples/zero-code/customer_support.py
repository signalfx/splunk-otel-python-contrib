"""CrewAI Customer Support Example (Zero-Code Instrumentation).

Run with:
    opentelemetry-instrument python customer_support.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set environment before any other imports (zero-code convenience defaults).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")

# Import shared app logic
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from customer_support_app import (  # noqa: E402
    DEFAULT_INPUTS,
    build_customer_support_crew,
    create_cisco_llm,
)


if __name__ == "__main__":
    llm, _token_manager, token = create_cisco_llm()
    print(f"[AUTH] Token obtained (length: {len(token)})")

    crew, support_agent, qa_agent = build_customer_support_crew(llm)

    # Recreate the LLM with a fresh token immediately before kickoff.
    llm, _token_manager, _ = create_cisco_llm()
    support_agent.llm = llm
    qa_agent.llm = llm

    result = crew.kickoff(inputs=DEFAULT_INPUTS)
    print("\n[SUCCESS] Crew execution completed")
    print(result)

