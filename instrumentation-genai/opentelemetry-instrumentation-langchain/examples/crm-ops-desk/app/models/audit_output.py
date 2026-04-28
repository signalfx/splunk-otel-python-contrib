from __future__ import annotations

from pydantic import BaseModel

from app.models.action_output import ToolReceipt


class Citation(BaseModel):
    source: str
    doc_id: str
    version: str
    relevance_score: float


class AuditOutput(BaseModel):
    interaction_id: str
    span_ids: list[str]
    citations: list[Citation]
    tool_receipts: list[ToolReceipt]
    final_verdict: str
    rationale: str
    created_at: str
