from typing import TypedDict, List, Dict, Any


class AssistantState(TypedDict):
    patient_id: str
    question: str
    user_role: str
    patient_context: str
    protocol_context: str
    answer: str
    sources: List[str]
    risk_level: str
    requires_human_validation: bool
    audit_log: Dict[str, Any]