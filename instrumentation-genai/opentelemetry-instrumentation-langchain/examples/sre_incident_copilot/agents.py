"""Agent implementations for SRE Incident Copilot."""

import base64
import json
import os
import time
from typing import Dict

import requests
from langchain.agents import create_agent as create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import Config
from tools import (
    investigation_agent_mcp,
    logs_search,
    metrics_query,
    notifier,
    runbook_search,
    service_catalog_lookup,
    task_writer,
    trace_query,
)


def _get_oauth_token(config: Config) -> str:
    """Get OAuth2 token for Cisco endpoint using Basic Auth."""
    if not config.oauth_token_url:
        return config.openai_api_key
    
    # Create Basic Auth header
    credentials = f"{config.oauth_client_id}:{config.oauth_client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {"grant_type": "client_credentials"}
    
    response = requests.post(config.oauth_token_url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def _create_llm(agent_name: str, temperature: float, config: Config) -> ChatOpenAI:
    """Create an LLM instance with metadata for tracing."""
    model = config.openai_model
    
    # Prepare kwargs for ChatOpenAI
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "tags": [f"agent:{agent_name}", "sre-incident-copilot"],
        "metadata": {
            "agent_name": agent_name,
            "agent_type": agent_name,
            "model": model,
            "temperature": temperature,
        },
    }
    
    # Add custom base URL if configured
    if config.openai_base_url:
        llm_kwargs["base_url"] = config.openai_base_url
    
    # Handle OAuth2 authentication for Cisco
    if config.oauth_token_url:
        token = _get_oauth_token(config)
        # Cisco expects the token in 'api-key' header
        llm_kwargs["api_key"] = token
        llm_kwargs["default_headers"] = {"api-key": token}
        # App key goes in the 'user' field of each request
        if config.oauth_app_key:
            llm_kwargs["model_kwargs"] = {"user": f'{{"appkey":"{config.oauth_app_key}"}}'}
    else:
        llm_kwargs["api_key"] = config.openai_api_key
    
    return ChatOpenAI(**llm_kwargs)


def triage_agent(state: Dict, config: Config) -> Dict:
    """Triage agent: normalize alert, identify service, choose investigation plan."""
    llm = _create_llm("triage", temperature=0.2, config=config)

    triage_tools = [service_catalog_lookup, runbook_search, investigation_agent_mcp]
    agent = create_react_agent(llm, tools=triage_tools).with_config(
        {
            "run_name": "triage_agent",
            "tags": ["agent", "agent:triage"],
            "metadata": {
                "agent_name": "triage",
                "session_id": state.get("session_id", ""),
            },
        }
    )

    alert_id = state.get("alert_id")
    scenario_id = state.get("scenario_id")
    service_id = state.get("service_id")  # Get from alert/initial state
    incident_context = state.get("incident_context", {})

    prompt = f"""You are a Triage Agent for SRE incidents. Your task is to:
1. Understand the alert context
2. Identify the affected service and its dependencies
3. Create an investigation checklist
4. REQUIRED: Call investigation_agent_mcp to perform investigation using the checklist you created

Alert ID: {alert_id}
Scenario ID: {scenario_id}
Service ID: {service_id}
Alert Title: {incident_context.get('title', 'N/A')}

IMPORTANT: You MUST call investigation_agent_mcp after creating the investigation checklist.
Steps:
1. Use service_catalog_lookup with service_id="{service_id}" to get service details
2. Optionally use runbook_search if needed
3. Create an investigation_checklist (list of investigation steps)
4. Call investigation_agent_mcp with:
   - service_id="{service_id}"
   - investigation_checklist: JSON string of your checklist
   - scenario_id="{scenario_id}" (if available)

The investigation_agent_mcp will return hypotheses and evidence that you should include in your output.

Output a JSON object with:
- service_id: The affected service (must match the service_id from the alert: {service_id})
- service_info: Key service details
- investigation_checklist: List of investigation steps (focus on metrics, logs, and traces to query)
- relevant_runbooks: Optional list of runbook names if you searched for them
- investigation_result: Investigation results from investigation_agent_mcp (REQUIRED - must call the tool)
"""

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_message = result["messages"][-1]

    # Extract investigation results from tool calls (investigation_agent_mcp)
    # Look through all messages for ToolMessage from investigation_agent_mcp
    investigation_result_data = None
    for msg in result["messages"]:
        if hasattr(msg, "name") and msg.name == "investigation_agent_mcp":
            if hasattr(msg, "content"):
                try:
                    investigation_result_data = json.loads(msg.content)
                    break
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    # Parse the response
    content = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )

    # Try to extract JSON from the response
    try:
        # Look for JSON in the response
        if "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            triage_result = json.loads(content[json_start:json_end])
        else:
            triage_result = {"raw_response": content}
    except (json.JSONDecodeError, TypeError, ValueError):
        triage_result = {"raw_response": content}

    state["triage_result"] = triage_result
    # Ensure service_id is set - prefer from triage result, but fall back to original
    extracted_service_id = triage_result.get("service_id")
    if extracted_service_id and extracted_service_id != "unknown":
        state["service_id"] = extracted_service_id
    elif not state.get("service_id") or state.get("service_id") == "unknown":
        # If still unknown, try to get from alert context
        incident_context = state.get("incident_context", {})
        state["service_id"] = incident_context.get("service_id") or state.get(
            "service_id", "unknown"
        )

    # Extract investigation results from investigation_agent_mcp tool call
    if investigation_result_data:
        if investigation_result_data.get("status") == "success":
            # Extract hypotheses and evidence from investigation result
            if "hypotheses" in investigation_result_data:
                state["hypotheses"] = investigation_result_data["hypotheses"]
            if "investigation_result" in investigation_result_data:
                inv_result = investigation_result_data["investigation_result"]
                if isinstance(inv_result, dict):
                    state["investigation_result"] = {
                        "evidence_summary": inv_result.get("evidence_summary", ""),
                        "evidence_types": inv_result.get("evidence_types", []),
                    }
            # Also check direct fields
            if "evidence_summary" in investigation_result_data:
                if "investigation_result" not in state:
                    state["investigation_result"] = {}
                state["investigation_result"]["evidence_summary"] = (
                    investigation_result_data["evidence_summary"]
                )
            if "evidence_types" in investigation_result_data:
                if "investigation_result" not in state:
                    state["investigation_result"] = {}
                state["investigation_result"]["evidence_types"] = (
                    investigation_result_data["evidence_types"]
                )
            # Extract confidence_score
            if "confidence_score" in investigation_result_data:
                state["confidence_score"] = investigation_result_data[
                    "confidence_score"
                ]
            elif "investigation_result" in investigation_result_data:
                inv_result = investigation_result_data["investigation_result"]
                if isinstance(inv_result, dict) and "confidence_score" in inv_result:
                    state["confidence_score"] = inv_result["confidence_score"]

    state["current_agent"] = (
        "action_planner"  # Next agent is action_planner (investigation done as tool)
    )

    return state


def investigation_agent(state: Dict, config: Config) -> Dict:
    """Investigation agent: query metrics/logs/traces, assemble evidence, propose hypotheses."""
    llm = _create_llm("investigation", temperature=0.3, config=config)

    service_id = state.get("service_id")
    scenario_id = state.get("scenario_id")

    # Set scenario_id in environment so tools can access it
    original_scenario = os.environ.get("CURRENT_SCENARIO_ID")
    if scenario_id:
        os.environ["CURRENT_SCENARIO_ID"] = scenario_id

    investigation_tools = [metrics_query, logs_search, trace_query]
    agent = create_react_agent(llm, tools=investigation_tools).with_config(
        {
            "run_name": "investigation_agent",
            "tags": ["agent", "agent:investigation"],
            "metadata": {
                "agent_name": "investigation",
                "session_id": state.get("session_id", ""),
            },
        }
    )

    triage_result = state.get("triage_result", {})
    incident_context = state.get("incident_context", {})
    alert_type = incident_context.get("alert_type", "")

    prompt = f"""You are an Investigation Agent for SRE incidents. Your task is to:
1. Query metrics, logs, and traces for the affected service
2. Collect evidence from multiple sources
3. Propose hypotheses about the root cause with confidence scores
4. Cite specific evidence (query results, log entries, trace IDs)

Service ID: {service_id}
Scenario ID: {scenario_id}
Alert Type: {alert_type}

Investigation checklist: {json.dumps(triage_result.get('investigation_checklist', []))}

IMPORTANT: Query ALL relevant metrics based on the alert type. 
The metrics_query tool description contains guidance on which metrics to query for different alert types.
Review the tool description and query the appropriate metrics for this alert type.

Use metrics_query to check error rates, latency, cache metrics, and other key metrics. Pass scenario_id={scenario_id} when calling.
Use logs_search to find error messages and warnings. Pass scenario_id={scenario_id} when calling.
Use trace_query to examine request flows and identify bottlenecks. Pass scenario_id={scenario_id} when calling.

For each hypothesis, provide:
- hypothesis: Description of the root cause
- confidence: Confidence score (0.0 to 1.0)
- evidence: List of evidence pieces with citations
- evidence_types: Types of evidence (metrics, logs, traces)

CRITICAL: You MUST output ONLY valid JSON. Do not include markdown formatting, explanations, or any text outside the JSON object.

Output a JSON object with this exact structure:
{{
  "hypotheses": [
    {{
      "hypothesis": "Description of root cause",
      "confidence": 0.85,
      "evidence": ["Evidence 1", "Evidence 2"],
      "evidence_types": ["metrics", "logs", "traces"]
    }}
  ],
  "evidence_summary": {{
    "metrics": {{}},
    "logs": [],
    "traces": []
  }}
}}
"""

    try:
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        final_message = result["messages"][-1]

        content = (
            final_message.content
            if isinstance(final_message, BaseMessage)
            else str(final_message)
        )

        # Try to extract JSON from the response
        investigation_result = None

        # First, try to find JSON block (may be wrapped in markdown code blocks)
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",  # JSON in markdown code block
            r"```\s*(\{.*?\})\s*```",  # JSON in generic code block
            r"(\{.*\})",  # Any JSON object
        ]

        import re

        for pattern in json_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    investigation_result = json.loads(match.group(1))
                    break
                except json.JSONDecodeError:
                    continue

        # If no JSON found, try to parse the entire content
        if investigation_result is None:
            try:
                if "{" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    investigation_result = json.loads(content[json_start:json_end])
            except json.JSONDecodeError:
                pass

        # If still no JSON, try to extract hypotheses from markdown structure
        if investigation_result is None or not investigation_result.get("hypotheses"):
            # Try to parse markdown-formatted hypotheses
            # Format: "1. **Hypothesis 1: Title**\n   - **Description**: ...\n   - **Confidence**: 0.9"
            hypotheses = []

            # Split content by hypothesis markers
            hyp_sections = re.split(
                r"(?:\d+\.\s*\*\*Hypothesis\s*\d+:|### Hypothesis)",
                content,
                flags=re.IGNORECASE,
            )

            for section in hyp_sections[
                1:
            ]:  # Skip first section (before first hypothesis)
                hyp_data = {}

                # Extract hypothesis title/description
                title_match = re.search(r"\*\*(.*?)\*\*", section)
                if title_match:
                    hyp_data["title"] = title_match.group(1).strip()

                # Extract description
                desc_match = re.search(
                    r"(?:Description|description)[:\-]?\s*(.*?)(?:\n\s*[-*]|Confidence|Evidence|$)",
                    section,
                    re.DOTALL | re.IGNORECASE,
                )
                if desc_match:
                    desc_text = desc_match.group(1).strip()
                    # Clean up markdown formatting
                    desc_text = re.sub(r"\*\*", "", desc_text)
                    desc_text = re.sub(r"`", "", desc_text)
                    hyp_data["description"] = desc_text

                # Extract confidence
                conf_match = re.search(
                    r"(?:Confidence|confidence)[:\-]?\s*([\d.]+)",
                    section,
                    re.IGNORECASE,
                )
                if conf_match:
                    hyp_data["confidence"] = float(conf_match.group(1))
                else:
                    hyp_data["confidence"] = 0.5

                # Extract evidence
                evidence = []
                evidence_types = []

                # Look for evidence section
                evidence_section = re.search(
                    r"(?:Evidence|evidence)[:\-]?\s*(.*?)(?:Evidence Types|evidence_types|$)",
                    section,
                    re.DOTALL | re.IGNORECASE,
                )
                if evidence_section:
                    evidence_text = evidence_section.group(1)
                    # Extract evidence items (lines starting with - or *)
                    evidence_items = re.findall(
                        r"[-*]\s*\*\*(.*?)\*\*[:\-]?\s*(.*?)(?:\n|$)",
                        evidence_text,
                        re.DOTALL,
                    )
                    for item_type, item_desc in evidence_items:
                        evidence.append(f"{item_type.strip()}: {item_desc.strip()}")
                        if item_type.lower() in ["metric", "metrics"]:
                            evidence_types.append("metrics")
                        elif item_type.lower() in ["log", "logs"]:
                            evidence_types.append("logs")
                        elif item_type.lower() in ["trace", "traces"]:
                            evidence_types.append("traces")

                # Extract evidence types explicitly listed
                ev_types_match = re.search(
                    r"(?:Evidence Types|evidence_types)[:\-]?\s*\[?(.*?)\]?",
                    section,
                    re.IGNORECASE,
                )
                if ev_types_match:
                    types_text = ev_types_match.group(1)
                    types_list = re.findall(r"(\w+)", types_text)
                    evidence_types = list(set(evidence_types + types_list))

                # Build hypothesis text
                if hyp_data.get("title") and hyp_data.get("description"):
                    hypothesis_text = f"{hyp_data['title']}: {hyp_data['description']}"
                elif hyp_data.get("title"):
                    hypothesis_text = hyp_data["title"]
                elif hyp_data.get("description"):
                    hypothesis_text = hyp_data["description"]
                else:
                    continue  # Skip if no useful content

                hypotheses.append(
                    {
                        "hypothesis": hypothesis_text[:500],  # Limit length
                        "confidence": hyp_data.get("confidence", 0.5),
                        "evidence": evidence[:5],  # Limit to 5 evidence items
                        "evidence_types": list(set(evidence_types))
                        if evidence_types
                        else ["metrics", "logs", "traces"],
                    }
                )

            # If we found hypotheses in markdown, use them
            if hypotheses:
                investigation_result = {
                    "hypotheses": hypotheses,
                    "evidence_summary": {},
                }
                print(f"✅ Extracted {len(hypotheses)} hypotheses from markdown format")
            else:
                investigation_result = {"raw_response": content, "hypotheses": []}
                print(
                    f"⚠️  Warning: No hypotheses found in investigation result. Raw response: {content[:500]}..."
                )

        state["investigation_result"] = investigation_result
        state["hypotheses"] = investigation_result.get("hypotheses", [])

        # Calculate overall confidence
        if state["hypotheses"]:
            top_hypothesis = state["hypotheses"][0]
            state["confidence_score"] = top_hypothesis.get("confidence", 0.5)
        else:
            state["confidence_score"] = 0.3

        state["current_agent"] = "action_planner"
    except Exception as e:
        print(f"Error in investigation agent: {e}")
        import traceback

        traceback.print_exc()
        # Set defaults so workflow can continue
        state["investigation_result"] = {"error": str(e), "hypotheses": []}
        state["hypotheses"] = []
        state["confidence_score"] = 0.3
        state["current_agent"] = "action_planner"
    finally:
        # Restore original scenario_id
        if original_scenario:
            os.environ["CURRENT_SCENARIO_ID"] = original_scenario
        elif "CURRENT_SCENARIO_ID" in os.environ:
            del os.environ["CURRENT_SCENARIO_ID"]

    return state


def action_planner_agent(state: Dict, config: Config) -> Dict:
    """Action planner agent: translate best hypothesis into steps and tasks."""
    llm = _create_llm("action_planner", temperature=0.4, config=config)

    planner_tools = [task_writer, notifier, runbook_search]
    agent = create_react_agent(llm, tools=planner_tools).with_config(
        {
            "run_name": "action_planner_agent",
            "tags": ["agent", "agent:action_planner"],
            "metadata": {
                "agent_name": "action_planner",
                "session_id": state.get("session_id", ""),
            },
        }
    )

    service_id = state.get("service_id")
    hypotheses = state.get("hypotheses", [])
    confidence_score = state.get("confidence_score", 0.5)

    prompt = f"""You are an Action Planner Agent for SRE incidents. Your task is to:
1. Select the best hypothesis (highest confidence)
2. Create a mitigation plan with ranked steps
3. Create tasks/tickets for the actions
4. Draft a communication message for the team
5. Reference relevant runbook sections

Service ID: {service_id}
Top Hypothesis: {json.dumps(hypotheses[0] if hypotheses else {})}
Confidence Score: {confidence_score}

IMPORTANT: You MUST use the runbook_search tool to find relevant runbook sections before creating the mitigation plan.
- Call runbook_search with a query based on the top hypothesis (e.g., "cache miss storm", "database connection pool", etc.)
- Use the runbook sections found to inform your mitigation plan
- Include citations in runbook_references field

Then use task_writer to create tasks for each action item.
Then use notifier to draft a team communication.

Output a JSON object with:
- selected_hypothesis: The chosen hypothesis
- mitigation_plan: List of ranked mitigation steps (must reference runbook sections)
- rollback_plan: Rollback steps if needed
- tasks: List of task objects to create. Each task MUST have:
  - title: Task title (REQUIRED)
  - description: Task description (REQUIRED)
  - priority: Task priority (optional: low, medium, high, critical, defaults to medium)
  - assignee: Optional assignee email
  Do NOT include id or url fields - those are generated automatically
- communication_draft: Draft message for team
- runbook_references: List of citations to runbook sections used (REQUIRED - format: ["runbook:name#chunk-0", ...])
"""

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_message = result["messages"][-1]

    content = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )

    try:
        if "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            action_plan = json.loads(content[json_start:json_end])
        else:
            action_plan = {"raw_response": content}
    except (json.JSONDecodeError, TypeError, ValueError):
        action_plan = {"raw_response": content}

    state["action_plan"] = action_plan

    # Create tasks if specified
    tasks_created = []
    if "tasks" in action_plan:
        for task_spec in action_plan.get("tasks", []):
            try:
                # Ensure required fields are present
                if not isinstance(task_spec, dict):
                    continue
                # Remove any fields that shouldn't be passed to task_writer (id, url are outputs, not inputs)
                clean_task_spec = {
                    "title": task_spec.get("title", ""),
                    "description": task_spec.get("description", ""),
                    "priority": task_spec.get("priority", "medium"),
                }
                if "assignee" in task_spec:
                    clean_task_spec["assignee"] = task_spec["assignee"]

                # Skip if required fields are missing
                if not clean_task_spec.get("title") or not clean_task_spec.get(
                    "description"
                ):
                    print(
                        "⚠️  Skipping task creation: missing required fields (title or description)"
                    )
                    continue

                task_result = task_writer.invoke(clean_task_spec)
                tasks_created.append(json.loads(task_result))
            except Exception as e:
                print(f"⚠️  Error creating task: {e}")
                import traceback

                traceback.print_exc()

    state["tickets_created"] = tasks_created
    state["current_agent"] = "quality_gate"

    return state


def quality_gate_agent(state: Dict, config: Config) -> Dict:
    """Quality gate agent: enforce safety rails, validate outputs, compute eval metrics."""
    llm = _create_llm("quality_gate", temperature=0.2, config=config)

    agent = create_react_agent(llm, tools=[]).with_config(
        {
            "run_name": "quality_gate_agent",
            "tags": ["agent", "agent:quality_gate"],
            "metadata": {
                "agent_name": "quality_gate",
                "session_id": state.get("session_id", ""),
            },
        }
    )

    confidence_score = state.get("confidence_score", 0.5)
    hypotheses = state.get("hypotheses", [])
    action_plan = state.get("action_plan", {})

    # Get alert info for requires_approval check
    alert_id = state.get("alert_id")
    from data_loader import DataLoader

    data_loader = DataLoader(data_dir=config.data_dir)
    alert = data_loader.get_alert(alert_id) if alert_id else None
    requires_approval = alert.get("requires_approval", False) if alert else False

    # Count evidence
    evidence_count = 0
    if hypotheses:
        for hyp in hypotheses:
            evidence_count += len(hyp.get("evidence", []))

    # Check thresholds
    confidence_threshold = config.confidence_threshold
    evidence_threshold = config.evidence_count_threshold

    approval_note = (
        "\nIMPORTANT: This alert requires approval before actions can be executed. Set approval_requested=true if actions need approval."
        if requires_approval
        else ""
    )

    prompt = f"""You are a Quality Gate Agent for SRE incidents. Your task is to:
1. Validate the investigation and action plan quality
2. Check if confidence and evidence meet thresholds
3. Determine if writebacks (tasks, notifications) should be allowed
4. Compute evaluation metrics

Confidence Score: {confidence_score}
Confidence Threshold: {confidence_threshold}
Evidence Count: {evidence_count}
Evidence Threshold: {evidence_threshold}
Requires Approval: {requires_approval}{approval_note}

Hypotheses: {len(hypotheses)}
Action Plan: {json.dumps(action_plan.get('mitigation_plan', []))}

Output a JSON object with:
- validation_passed: Boolean indicating if quality checks passed
- confidence_check: Whether confidence meets threshold
- evidence_check: Whether evidence count meets threshold
- action_safety: Whether actions are safe to execute
- writeback_allowed: Boolean indicating if tasks/notifications can be created (must be false if requires_approval=true and approval not requested)
- approval_requested: Boolean indicating if approval was requested (REQUIRED if requires_approval=true)
- eval_metrics: Object with metrics like hypothesis_accuracy, evidence_sufficiency, action_safety
- recommendations: List of recommendations or required approvals
"""

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_message = result["messages"][-1]

    content = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )

    try:
        if "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            quality_result = json.loads(content[json_start:json_end])
        else:
            quality_result = {
                "validation_passed": confidence_score >= confidence_threshold
                and evidence_count >= evidence_threshold,
                "writeback_allowed": False,
            }
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        quality_result = {
            "validation_passed": confidence_score >= confidence_threshold
            and evidence_count >= evidence_threshold,
            "writeback_allowed": False,
        }

    # Override with deterministic checks
    quality_result["confidence_check"] = confidence_score >= confidence_threshold
    quality_result["evidence_check"] = evidence_count >= evidence_threshold

    # Handle approval requirement
    if requires_approval:
        # If approval is required but not requested, set it
        if "approval_requested" not in quality_result:
            quality_result["approval_requested"] = False
        # Writeback only allowed if approval was requested
        if not quality_result.get("approval_requested", False):
            quality_result["writeback_allowed"] = False
    else:
        # No approval required, default to False if not set
        if "approval_requested" not in quality_result:
            quality_result["approval_requested"] = False

    # Set writeback_allowed based on checks and approval
    if "writeback_allowed" not in quality_result:
        quality_result["writeback_allowed"] = (
            quality_result.get("confidence_check", False)
            and quality_result.get("evidence_check", False)
            and quality_result.get("action_safety", True)
            and (
                not requires_approval or quality_result.get("approval_requested", False)
            )
        )

    state["quality_gate_result"] = quality_result
    state["eval_metrics"] = quality_result.get("eval_metrics", {})
    state["current_agent"] = "completed"

    return state
