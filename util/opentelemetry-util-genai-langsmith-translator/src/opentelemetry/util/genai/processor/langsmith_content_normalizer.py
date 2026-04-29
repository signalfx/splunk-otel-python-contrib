"""
Normalize Langsmith entity input/output blobs into GenAI message schema.

Langsmith uses its own tracing format with specific attribute patterns.
This module handles the conversion from Langsmith format to the standardized
GenAI semantic convention format.

Reference: https://docs.smith.langchain.com/
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Internal sizing caps (kept private to module, not exposed via env)
INPUT_MAX = 100
OUTPUT_MAX = 100
MSG_CONTENT_MAX = 16000
PROMPT_TEMPLATE_MAX = 4096

# Regex patterns for extracting content from Python repr() strings
# Matches: HumanMessage(content="...", ...) or HumanMessage(content='...', ...)
_MESSAGE_REPR_PATTERN = re.compile(
    r"(Human|AI|System|Tool|Function)Message\("
    r"content=['\"](.+?)['\"]"
    r"(?:,.*?)?\)",
    re.DOTALL,
)

# More robust pattern that handles escaped quotes and longer content
_MESSAGE_CONTENT_PATTERN = re.compile(
    r"(Human|AI|System|Tool|Function)Message\(content=['\"]([^'\"]*(?:\\.[^'\"]*)*)['\"]",
    re.DOTALL,
)


def maybe_truncate_template(value: Any) -> Any:
    """Truncate a prompt template if it exceeds the maximum length."""
    if not isinstance(value, str) or len(value) <= PROMPT_TEMPLATE_MAX:
        return value
    return value[:PROMPT_TEMPLATE_MAX] + "…(truncated)"


def _coerce_text_part(content: Any) -> Dict[str, Any]:
    """Convert content to a standardized text part dict."""
    if not isinstance(content, str):
        try:
            content = json.dumps(content)[:MSG_CONTENT_MAX]
        except Exception:
            content = str(content)[:MSG_CONTENT_MAX]
    else:
        content = content[:MSG_CONTENT_MAX]
    return {"type": "text", "content": content}


def _extract_from_python_repr(text: str) -> List[Dict[str, Any]]:
    """
    Extract message content from Python repr() strings.

    Handles strings like:
    - "{'messages': [HumanMessage(content=\"What's my balance?\")]}"
    - "[HumanMessage(content='hello'), AIMessage(content='hi')]"

    Returns list of extracted messages with role and content.
    """
    extracted = []

    # Try the more robust pattern first
    matches = _MESSAGE_CONTENT_PATTERN.findall(text)

    for msg_type, content in matches:
        # Unescape the content
        content = (
            content.replace("\\'", "'")
            .replace('\\"', '"')
            .replace("\\n", "\n")
        )

        # Map message type to role
        role_map = {
            "Human": "user",
            "AI": "assistant",
            "System": "system",
            "Tool": "tool",
            "Function": "function",
        }
        role = role_map.get(msg_type, "user")
        extracted.append({"role": role, "content": content})

    return extracted


def _try_parse_json_safe(text: str) -> Optional[Any]:
    """Try to parse JSON, returning None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_from_generations(
    content_val: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract messages from LangChain LLMResult 'generations' format.

    Format: {"generations": [[{"text": "...", "message": {"lc": 1, ...}}]], ...}

    Returns list of extracted messages with their content and role.
    """
    extracted = []

    if "generations" not in content_val or not isinstance(
        content_val["generations"], list
    ):
        return []

    for gen_batch in content_val["generations"]:
        if not isinstance(gen_batch, list):
            continue
        for gen in gen_batch:
            if not isinstance(gen, dict):
                continue

            content = ""
            finish_reason = "stop"
            tool_calls = None

            # Try to extract from "message" if present (LangChain format)
            message = gen.get("message")
            if isinstance(message, dict):
                # Check for LangChain serialized message
                if message.get("lc") == 1 and "kwargs" in message:
                    kwargs = message["kwargs"]
                    content = kwargs.get("content", "")
                    # Extract tool_calls if present
                    tool_calls = kwargs.get("tool_calls")
                    # Extract finish_reason from response_metadata
                    resp_meta = kwargs.get("response_metadata", {})
                    if isinstance(resp_meta, dict):
                        finish_reason = resp_meta.get("finish_reason", "stop")
                else:
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls")
            else:
                # Fallback to "text" field
                content = gen.get("text", "")

            # Get finish_reason from generation_info
            gen_info = gen.get("generation_info", {})
            if isinstance(gen_info, dict):
                finish_reason = gen_info.get("finish_reason", finish_reason)

            # Build the message dict
            msg_dict: Dict[str, Any] = {
                "content": content,
                "role": "assistant",
                "finish_reason": finish_reason,
            }

            # Include tool_calls if present
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls

            extracted.append(msg_dict)

    return extracted


def _extract_langchain_messages(content_val: Any) -> List[Dict[str, Any]]:
    """
    Extract actual message content from nested LangChain message objects.

    Langsmith traces LangChain applications, which use formats like:
    - {"messages": [{"lc": 1, "kwargs": {"content": "text", "type": "human"}}]}
    - {"outputs": {"messages": [{"lc": 1, "kwargs": {"content": "text"}}]}}
    - {"inputs": {"messages": [{"lc": 1, "kwargs": {"content": "text"}}]}}
    - {"args": [{"messages": [{"lc": 1, "kwargs": {"content": "text"}}]}]}
    - Nested arrays: [[{"lc": 1, ...}]]
    - Python repr() strings: "{'messages': [HumanMessage(content=\"...\")]}"

    Returns list of extracted messages with their content and role.
    """
    extracted = []

    try:
        # Parse if it's a JSON string
        if isinstance(content_val, str):
            # CRITICAL: Check for Python repr() format FIRST
            # This handles strings like: "{'messages': [HumanMessage(content=\"What's my balance?\")]}"
            if "Message(" in content_val:
                repr_msgs = _extract_from_python_repr(content_val)
                if repr_msgs:
                    return repr_msgs

            # Try JSON parsing
            parsed = _try_parse_json_safe(content_val)
            if parsed is not None:
                content_val = parsed
            else:
                return []  # Not JSON and not repr, let caller handle it

        if isinstance(content_val, list):
            # Handle nested arrays like [[{"lc": 1, ...}]]
            if len(content_val) > 0 and isinstance(content_val[0], list):
                # Flatten the first level
                content_val = content_val[0]

            # Process as message list
            for msg in content_val:
                if not isinstance(msg, dict):
                    continue
                msg_result = _extract_single_langchain_message(msg)
                if msg_result:
                    extracted.append(msg_result)
            return extracted

        if not isinstance(content_val, dict):
            return []

        # Check for "outputs" wrapper (common in workflow outputs)
        if "outputs" in content_val and isinstance(
            content_val["outputs"], dict
        ):
            content_val = content_val["outputs"]

        # Check for "inputs" wrapper (common in workflow inputs)
        if "inputs" in content_val and isinstance(content_val["inputs"], dict):
            content_val = content_val["inputs"]

        # Check for "args" wrapper (LangGraph format)
        if "args" in content_val and isinstance(content_val["args"], list):
            if len(content_val["args"]) > 0 and isinstance(
                content_val["args"][0], dict
            ):
                content_val = content_val["args"][0]

        # Check for LangChain LLMResult "generations" format FIRST
        # Format: {"generations": [[{"text": "...", "message": {"lc": 1, ...}}]], ...}
        if "generations" in content_val and isinstance(
            content_val["generations"], list
        ):
            gen_msgs = _extract_from_generations(content_val)
            if gen_msgs:
                return gen_msgs

        # Look for "messages" array
        messages = content_val.get("messages", [])

        # Handle nested array in messages
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], list):
                # Flatten nested: [[msg1, msg2]] -> [msg1, msg2]
                messages = messages[0]

        if not isinstance(messages, list):
            return []

        # Extract content from each LangChain message
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_result = _extract_single_langchain_message(msg)
            if msg_result:
                extracted.append(msg_result)

        return extracted

    except Exception:
        return []


def _extract_single_langchain_message(msg: dict) -> Optional[Dict[str, Any]]:
    """
    Extract content and role from a single LangChain message dict.

    Handles formats:
    - LangChain serialization: {"lc": 1, "kwargs": {"content": "...", "type": "human"}}
    - Standard message: {"content": "...", "type": "human"}
    - OpenAI format: {"role": "user", "content": "..."}
    """
    # Check if this is a LangChain serialized message (has "lc": 1 and "kwargs")
    if msg.get("lc") == 1 and "kwargs" in msg:
        kwargs = msg["kwargs"]
        if isinstance(kwargs, dict):
            msg_content = kwargs.get("content")
            msg_type = kwargs.get("type", "unknown")

            if msg_content is not None:
                # Map LangChain types to roles
                role = _map_langchain_type_to_role(msg_type)
                return {"content": msg_content, "role": role}

    # Check for standard LangChain message format (from entity output)
    # Format: {"content": "...", "type": "human", "additional_kwargs": {...}}
    if "type" in msg and "content" in msg:
        msg_type = msg.get("type", "unknown")
        msg_content = msg.get("content")

        if msg_content is not None:
            role = _map_langchain_type_to_role(msg_type)

            # Handle tool messages specially
            if msg_type == "tool":
                return {
                    "content": msg_content,
                    "role": "tool",
                    "name": msg.get("name"),
                    "tool_call_id": msg.get("tool_call_id"),
                }

            return {"content": msg_content, "role": role}

    # Check for OpenAI-style format
    if "role" in msg and "content" in msg:
        return {
            "content": msg.get("content", ""),
            "role": msg.get("role", "user"),
        }

    return None


def _map_langchain_type_to_role(msg_type: str) -> str:
    """Map LangChain message type to standard role."""
    type_to_role = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "function",
    }
    return type_to_role.get(msg_type, "user")


def _extract_langsmith_run_io(content_val: Any) -> List[Dict[str, Any]]:
    """
    Extract input/output from Langsmith run serialization format.

    Langsmith uses formats like:
    - {"input": "text", "output": "response"}
    - {"prompt": "text", "completion": "response"}
    - {"question": "text", "answer": "response"}
    - Direct message arrays

    Returns list of extracted messages with their content and role.
    """
    extracted = []

    try:
        # Parse if it's a JSON string
        if isinstance(content_val, str):
            try:
                content_val = json.loads(content_val)
            except Exception:
                return []  # Not JSON, let caller handle it

        if not isinstance(content_val, dict):
            return []

        # Check for common Langsmith input/output patterns
        input_keys = [
            "input",
            "prompt",
            "question",
            "query",
            "text",
            "user_input",
        ]
        output_keys = [
            "output",
            "completion",
            "answer",
            "response",
            "result",
            "assistant_output",
        ]

        for key in input_keys:
            if key in content_val:
                val = content_val[key]
                if isinstance(val, str):
                    extracted.append({"content": val, "role": "user"})
                    break

        for key in output_keys:
            if key in content_val:
                val = content_val[key]
                if isinstance(val, str):
                    extracted.append({"content": val, "role": "assistant"})
                    break

        return extracted

    except Exception:
        return []


def normalize_langsmith_content(
    raw: Any, direction: str
) -> List[Dict[str, Any]]:
    """Normalize Langsmith entity input/output blob into GenAI message schema.

    Langsmith traces LangChain-based applications and records inputs/outputs
    in various formats depending on the chain type, model, and configuration.

    direction: 'input' | 'output'
    Returns list of messages: {role, parts, finish_reason?}
    """
    # CRITICAL: Handle nested arrays FIRST before any other processing
    # Langsmith often wraps messages in nested arrays: [[{"lc": 1, ...}]]
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        # Flatten the outer array and recurse
        return normalize_langsmith_content(raw[0], direction)

    # List[dict] messages already
    if isinstance(raw, list) and all(isinstance(m, dict) for m in raw):
        normalized: List[Dict[str, Any]] = []
        limit = INPUT_MAX if direction == "input" else OUTPUT_MAX
        for m in raw[:limit]:
            # FIRST: Check if this dict IS a LangChain serialized message
            # Format: {"lc": 1, "type": "constructor", "kwargs": {"content": "...", "type": "human"}}
            if m.get("lc") == 1 and "kwargs" in m:
                kwargs = m.get("kwargs", {})
                if isinstance(kwargs, dict):
                    msg_content = kwargs.get("content")
                    msg_type = kwargs.get("type", "unknown")

                    if msg_content:
                        # Map LangChain types to roles
                        if msg_type == "human":
                            role = "user"
                        elif msg_type == "ai":
                            role = "assistant"
                        elif msg_type == "system":
                            role = "system"
                        else:
                            role = (
                                "user" if direction == "input" else "assistant"
                            )

                        parts = [_coerce_text_part(msg_content)]
                        msg_dict: Dict[str, Any] = {
                            "role": role,
                            "parts": parts,
                        }
                        if direction == "output":
                            msg_dict["finish_reason"] = "stop"
                        normalized.append(msg_dict)
                        continue  # Skip to next message in list

            # SECOND: Check if message already has "parts" with nested LangChain data
            # Format: {"role": "user", "parts": [{"type": "text", "content": "{\"args\": [...]}"}]}
            if "parts" in m and isinstance(m.get("parts"), list):
                role = m.get(
                    "role", "user" if direction == "input" else "assistant"
                )
                extracted_msgs = []
                plain_parts = []

                for part in m["parts"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part_content = part.get("content", "")
                        # Try to extract nested LangChain messages from part content
                        lc_msgs = _extract_langchain_messages(part_content)
                        if lc_msgs:
                            extracted_msgs.extend(lc_msgs)
                        else:
                            plain_parts.append(_coerce_text_part(part_content))
                    else:
                        # Non-text part, keep as-is
                        plain_parts.append(
                            part
                            if isinstance(part, dict)
                            else _coerce_text_part(part)
                        )

                # If we found nested LangChain messages, use those
                if extracted_msgs:
                    for lc_msg in extracted_msgs:
                        msg: Dict[str, Any] = {
                            "role": lc_msg["role"],
                            "parts": [_coerce_text_part(lc_msg["content"])],
                        }
                        if direction == "output":
                            # Prefer finish_reason from extracted message, fallback to outer message
                            msg["finish_reason"] = (
                                lc_msg.get("finish_reason")
                                or m.get("finish_reason")
                                or m.get("finishReason")
                                or "stop"
                            )
                        normalized.append(msg)
                else:
                    # No nested messages, use plain parts
                    msg: Dict[str, Any] = {
                        "role": role,
                        "parts": plain_parts
                        if plain_parts
                        else [_coerce_text_part("")],
                    }
                    if direction == "output":
                        msg["finish_reason"] = (
                            m.get("finish_reason")
                            or m.get("finishReason")
                            or "stop"
                        )
                    normalized.append(msg)
                continue  # Skip to next message in list

            # Standard message format with role/content
            role = m.get(
                "role", "user" if direction == "input" else "assistant"
            )
            content_val = m.get("content")
            if content_val is None:
                temp = {
                    k: v
                    for k, v in m.items()
                    if k
                    not in ("role", "finish_reason", "finishReason", "parts")
                }
                content_val = temp or ""

            # CRITICAL FIX: Check if content contains nested LangChain messages
            # This handles the format where Langsmith serializes workflow inputs/outputs
            # with LangChain message objects embedded in JSON
            langchain_messages = _extract_langchain_messages(content_val)

            if langchain_messages:
                # We found nested LangChain messages - extract their content
                for lc_msg in langchain_messages:
                    parts = [_coerce_text_part(lc_msg["content"])]
                    msg: Dict[str, Any] = {
                        "role": lc_msg["role"],
                        "parts": parts,
                    }
                    if direction == "output":
                        fr = (
                            m.get("finish_reason")
                            or m.get("finishReason")
                            or "stop"
                        )
                        msg["finish_reason"] = fr
                    normalized.append(msg)
            else:
                # No nested LangChain messages - use content as-is
                parts = [_coerce_text_part(content_val)]
                msg: Dict[str, Any] = {"role": role, "parts": parts}
                if direction == "output":
                    fr = (
                        m.get("finish_reason")
                        or m.get("finishReason")
                        or "stop"
                    )
                    msg["finish_reason"] = fr
                normalized.append(msg)

        return normalized

    # Dict variants
    if isinstance(raw, dict):
        # OpenAI choices format
        if (
            direction == "output"
            and "choices" in raw
            and isinstance(raw["choices"], list)
        ):
            out_msgs: List[Dict[str, Any]] = []
            for choice in raw["choices"][:OUTPUT_MAX]:
                message = (
                    choice.get("message") if isinstance(choice, dict) else None
                )
                if message and isinstance(message, dict):
                    role = message.get("role", "assistant")
                    content_val = (
                        message.get("content") or message.get("text") or ""
                    )
                else:
                    role = "assistant"
                    content_val = (
                        choice.get("text")
                        or choice.get("content")
                        or json.dumps(choice)
                    )
                parts = [_coerce_text_part(content_val)]
                finish_reason = (
                    choice.get("finish_reason")
                    or choice.get("finishReason")
                    or "stop"
                )
                out_msgs.append(
                    {
                        "role": role,
                        "parts": parts,
                        "finish_reason": finish_reason,
                    }
                )
            return out_msgs

        # Gemini candidates format
        if (
            direction == "output"
            and "candidates" in raw
            and isinstance(raw["candidates"], list)
        ):
            out_msgs: List[Dict[str, Any]] = []
            for cand in raw["candidates"][:OUTPUT_MAX]:
                role = cand.get("role", "assistant")
                cand_content = cand.get("content")
                if isinstance(cand_content, list):
                    joined = "\n".join(
                        [
                            str(p.get("text", p.get("content", p)))
                            for p in cand_content
                        ]
                    )
                    content_val = joined
                else:
                    content_val = cand_content or json.dumps(cand)
                parts = [_coerce_text_part(content_val)]
                finish_reason = (
                    cand.get("finish_reason")
                    or cand.get("finishReason")
                    or "stop"
                )
                out_msgs.append(
                    {
                        "role": role,
                        "parts": parts,
                        "finish_reason": finish_reason,
                    }
                )
            return out_msgs

        # LangChain LLMResult "generations" format
        # Format: {"generations": [[{"text": "...", "message": {"lc": 1, ...}}]], ...}
        if (
            direction == "output"
            and "generations" in raw
            and isinstance(raw["generations"], list)
        ):
            out_msgs: List[Dict[str, Any]] = []
            for gen_batch in raw["generations"][:OUTPUT_MAX]:
                if not isinstance(gen_batch, list):
                    continue
                for gen in gen_batch[:OUTPUT_MAX]:
                    if not isinstance(gen, dict):
                        continue

                    content = ""
                    finish_reason = "stop"

                    # Try to extract from "message" if present (LangChain format)
                    message = gen.get("message")
                    if isinstance(message, dict):
                        # Check for LangChain serialized message
                        if message.get("lc") == 1 and "kwargs" in message:
                            kwargs = message["kwargs"]
                            content = kwargs.get("content", "")
                            # Extract finish_reason from response_metadata
                            resp_meta = kwargs.get("response_metadata", {})
                            if isinstance(resp_meta, dict):
                                finish_reason = resp_meta.get(
                                    "finish_reason", "stop"
                                )
                        else:
                            content = message.get("content", "")
                    else:
                        # Fallback to "text" field
                        content = gen.get("text", "")

                    # Get finish_reason from generation_info
                    gen_info = gen.get("generation_info", {})
                    if isinstance(gen_info, dict):
                        finish_reason = gen_info.get(
                            "finish_reason", finish_reason
                        )

                    if content:
                        out_msgs.append(
                            {
                                "role": "assistant",
                                "parts": [_coerce_text_part(content)],
                                "finish_reason": finish_reason,
                            }
                        )
            if out_msgs:
                return out_msgs

        # messages array
        if "messages" in raw and isinstance(raw["messages"], list):
            return normalize_langsmith_content(raw["messages"], direction)

        # wrapper args (LangGraph/Langsmith format with function call args)
        if (
            "args" in raw
            and isinstance(raw["args"], list)
            and len(raw["args"]) > 0
        ):
            # Extract first arg (usually contains messages and other params)
            first_arg = raw["args"][0]
            if isinstance(first_arg, dict):
                # Recursively process - will find "messages" array
                return normalize_langsmith_content(first_arg, direction)

        # wrapper inputs
        if "inputs" in raw:
            inner = raw["inputs"]
            if isinstance(inner, list):
                return normalize_langsmith_content(inner, direction)
            if isinstance(inner, dict):
                # Recursively process - might contain "messages" array
                return normalize_langsmith_content(inner, direction)

        # wrapper outputs (for output data)
        if "outputs" in raw:
            inner = raw["outputs"]
            if isinstance(inner, list):
                return normalize_langsmith_content(inner, direction)
            if isinstance(inner, dict):
                # Recursively process - might contain "messages" array
                return normalize_langsmith_content(inner, direction)

        # Langsmith run I/O format
        langsmith_msgs = _extract_langsmith_run_io(raw)
        if langsmith_msgs:
            normalized = []
            for lc_msg in langsmith_msgs:
                if (direction == "input" and lc_msg["role"] == "user") or (
                    direction == "output" and lc_msg["role"] == "assistant"
                ):
                    msg: Dict[str, Any] = {
                        "role": lc_msg["role"],
                        "parts": [_coerce_text_part(lc_msg["content"])],
                    }
                    if direction == "output":
                        msg["finish_reason"] = "stop"
                    normalized.append(msg)
            if normalized:
                return normalized

        # tool calls
        if (
            direction == "output"
            and "tool_calls" in raw
            and isinstance(raw["tool_calls"], list)
        ):
            out_msgs: List[Dict[str, Any]] = []
            for tc in raw["tool_calls"][:OUTPUT_MAX]:
                part = {
                    "type": "tool_call",
                    "name": tc.get("name", "tool"),
                    "arguments": tc.get("arguments"),
                    "id": tc.get("id"),
                }
                finish_reason = (
                    tc.get("finish_reason")
                    or tc.get("finishReason")
                    or "tool_call"
                )
                out_msgs.append(
                    {
                        "role": "assistant",
                        "parts": [part],
                        "finish_reason": finish_reason,
                    }
                )
            return out_msgs

        body = {k: v for k, v in raw.items() if k != "role"}
        if direction == "output":
            return [
                {
                    "role": "assistant",
                    "parts": [_coerce_text_part(body)],
                    "finish_reason": "stop",
                }
            ]
        return [{"role": "user", "parts": [_coerce_text_part(body)]}]

    # JSON string or Python repr string
    if isinstance(raw, str):
        # First try JSON parsing
        parsed = _try_parse_json_safe(raw)
        if parsed is not None:
            return normalize_langsmith_content(parsed, direction)

        # CRITICAL: Check for Python repr() format
        # This handles strings like: "{'messages': [HumanMessage(content=\"What's my balance?\")]}"
        if "Message(" in raw:
            repr_msgs = _extract_from_python_repr(raw)
            if repr_msgs:
                normalized = []
                for rm in repr_msgs:
                    msg: Dict[str, Any] = {
                        "role": rm["role"],
                        "parts": [_coerce_text_part(rm["content"])],
                    }
                    if direction == "output":
                        msg["finish_reason"] = "stop"
                    normalized.append(msg)
                return normalized

        # Fallback: wrap as plain text
        if direction == "output":
            return [
                {
                    "role": "assistant",
                    "parts": [_coerce_text_part(raw)],
                    "finish_reason": "stop",
                }
            ]
        return [{"role": "user", "parts": [_coerce_text_part(raw)]}]

    # List of raw strings
    if isinstance(raw, list) and all(isinstance(s, str) for s in raw):
        msgs: List[Dict[str, Any]] = []
        limit = INPUT_MAX if direction == "input" else OUTPUT_MAX
        for s in raw[:limit]:
            msgs.append(
                {
                    "role": "user" if direction == "input" else "assistant",
                    "parts": [_coerce_text_part(s)],
                }
            )
        return msgs

    # Generic fallback
    if direction == "output":
        return [
            {
                "role": "assistant",
                "parts": [_coerce_text_part(raw)],
                "finish_reason": "stop",
            }
        ]
    return [{"role": "user", "parts": [_coerce_text_part(raw)]}]


__all__ = [
    "normalize_langsmith_content",
    "maybe_truncate_template",
    "INPUT_MAX",
    "OUTPUT_MAX",
    "MSG_CONTENT_MAX",
    "PROMPT_TEMPLATE_MAX",
]
