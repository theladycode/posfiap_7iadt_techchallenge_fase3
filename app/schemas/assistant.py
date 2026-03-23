from pydantic import BaseModel
from typing import List, Optional


class AuditSummaryResponse(BaseModel):
    request_id: str
    status: str
    execution_mode: str
    fallback_used: bool
    duration_ms: int
    llm_provider: str
    model_name: str
    supporting_decision: str
    protocol_sources: List[str]


class AssistantRequest(BaseModel):
    patient_id: str
    question: str
    user_role: str
    context: Optional[str] = None


class AssistantResponse(BaseModel):
    answer: str
    sources: List[str]
    risk_level: str
    requires_human_validation: bool
    audit: AuditSummaryResponse