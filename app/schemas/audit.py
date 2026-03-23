from typing import Any, Dict, List
from pydantic import BaseModel


class AuditDetailResponse(BaseModel):
    request_id: str
    timestamp: str
    patient_id: str
    question: str
    user_role: str
    context_used: Dict[str, Any]
    answer: str
    sources: List[str]
    protocol_sources: List[str]
    risk_level: str
    requires_human_validation: bool
    llm_provider: str
    model_name: str
    supporting_model_output: str
    supporting_decision: str
    status: str
    execution_mode: str
    fallback_used: bool
    started_at: str
    finished_at: str
    duration_ms: int
    error_message: str | None = None