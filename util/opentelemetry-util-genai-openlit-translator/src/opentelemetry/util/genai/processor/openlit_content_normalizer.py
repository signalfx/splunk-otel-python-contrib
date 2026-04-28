from __future__ import annotations

import json
from typing import Any, Dict, List

# Internal sizing caps (kept private to module, not exposed via env)
INPUT_MAX = 100
OUTPUT_MAX = 100
MSG_CONTENT_MAX = 16000
PROMPT_TEMPLATE_MAX = 4096


def maybe_truncate_template(value: Any) -> Any:
    if not isinstance(value, str) or len(value) <= PROMPT_TEMPLATE_MAX:
        return value
    return value[:PROMPT_TEMPLATE_MAX] + "…(truncated)"


def _coerce_text_part(content: Any) -> Dict[str, Any]:
    if not isinstance(content, str):
        try:
            content = json.dumps(content)[:MSG_CONTENT_MAX]
        except Exception:
            content = str(content)[:MSG_CONTENT_MAX]
    else:
        content = content[:MSG_CONTENT_MAX]
    return {"type": "text", "content": content}


def _extract_langchain_messages(content_val: Any) -> List[Dict[str, Any]]:
    """
    Extract actual message content from nested LangChain message objects.

    Handles formats like:
    - {"messages": [{"lc": 1, "kwargs": {"content": "text", "type": "human"}}]}
    - {"outputs": {"messages": [{"lc": 1, "kwargs": {"content": "text"}}]}}

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
        if "outputs" in content_val and isinstance(
            content_val["outputs"], dict
        ):
            content_val = content_val["outputs"]

        messages = content_val.get("messages", [])
        if not isinstance(messages, list):
            return []

        # Extract content from each LangChain message
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Check if this is a LangChain message (has "lc": 1 and "kwargs")
            if msg.get("lc") == 1 and "kwargs" in msg:
                kwargs = msg["kwargs"]
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
                            # Infer from message position
                            role = "user" if not extracted else "assistant"

                        extracted.append(
                            {"content": msg_content, "role": role}
                        )

        return extracted

    except Exception:
        return []


def normalize_openlit_content(
    raw: Any, direction: str
) -> List[Dict[str, Any]]:
    """Normalize openlit entity input/output blob into GenAI message schema.

    direction: 'input' | 'output'
    Returns list of messages: {role, parts, finish_reason?}
    """
    # List[dict] messages already
    if isinstance(raw, list) and all(isinstance(m, dict) for m in raw):
        normalized: List[Dict[str, Any]] = []
        limit = INPUT_MAX if direction == "input" else OUTPUT_MAX
        for m in raw[:limit]:
            role = m.get(
                "role", "user" if direction == "input" else "assistant"
            )
            content_val = m.get("content")
            if content_val is None:
                temp = {
                    k: v
                    for k, v in m.items()
                    if k not in ("role", "finish_reason", "finishReason")
                }
                content_val = temp or ""

            # CRITICAL FIX: Check if content contains nested LangChain messages
            # This handles the format where openlit serializes workflow inputs/outputs
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
        # OpenAI choices
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
        # Gemini candidates
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
        # messages array
        if "messages" in raw and isinstance(raw["messages"], list):
            return normalize_openlit_content(raw["messages"], direction)
        # wrapper args (LangGraph/openlit format with function call args)
        if (
            "args" in raw
            and isinstance(raw["args"], list)
            and len(raw["args"]) > 0
        ):
            # Extract first arg (usually contains messages and other params)
            first_arg = raw["args"][0]
            if isinstance(first_arg, dict):
                # Recursively process - will find "messages" array
                return normalize_openlit_content(first_arg, direction)
        # wrapper inputs
        if "inputs" in raw:
            inner = raw["inputs"]
            if isinstance(inner, list):
                return normalize_openlit_content(inner, direction)
            if isinstance(inner, dict):
                # Recursively process - might contain "messages" array
                return normalize_openlit_content(inner, direction)
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

    # JSON string
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return normalize_openlit_content(parsed, direction)
        except Exception:
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
    "normalize_openlit_content",
    "maybe_truncate_template",
    "INPUT_MAX",
    "OUTPUT_MAX",
    "MSG_CONTENT_MAX",
    "PROMPT_TEMPLATE_MAX",
]
