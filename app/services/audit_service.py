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
        "llm_provider": state.get("llm_provider", ""),
        "model_name": state.get("model_name", ""),
        "supporting_model_output": state.get("supporting_model_output", ""),
    }