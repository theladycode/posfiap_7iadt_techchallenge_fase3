from pydantic import BaseModel
from typing import List, Optional


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