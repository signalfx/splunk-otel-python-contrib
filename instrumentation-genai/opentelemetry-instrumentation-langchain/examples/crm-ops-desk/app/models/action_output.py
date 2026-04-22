from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolReceipt(BaseModel):
    tool: str
    status: int
    latency_ms: float
    response: dict[str, Any]


class ActionOutput(BaseModel):
    resolution: str
    tool_receipts: list[ToolReceipt]
    cost_token_usd: float
