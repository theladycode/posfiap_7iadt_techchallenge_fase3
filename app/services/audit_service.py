from datetime import datetime
from typing import Dict, Any


def create_audit_log(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "patient_id": state["patient_id"],
        "question": state["question"],
        "answer": state["answer"],
        "sources": state["sources"],
        "risk_level": state["risk_level"],
        "requires_human_validation": state["requires_human_validation"],
    }