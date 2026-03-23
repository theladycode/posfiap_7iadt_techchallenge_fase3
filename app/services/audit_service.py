from datetime import datetime
from typing import Dict, Any


def create_audit_log(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": state.get("request_id", ""),
        "patient_id": state["patient_id"],
        "question": state["question"],
        "user_role": state.get("user_role", ""),
        "context_used": {
            "user_context": state.get("context", ""),
            "patient_context": state.get("patient_context", ""),
            "protocol_context": state.get("protocol_context", ""),
        },
        "answer": state["answer"],
        "sources": state["sources"],
        "protocol_sources": state.get("protocol_sources", []),
        "risk_level": state["risk_level"],
        "requires_human_validation": state["requires_human_validation"],
        "llm_provider": state.get("llm_provider", ""),
        "model_name": state.get("model_name", ""),
        "supporting_model_output": state.get("supporting_model_output", ""),
        "supporting_decision": state.get("supporting_decision", ""),
        "status": state.get("status", "success"),
        "execution_mode": state.get("execution_mode", ""),
        "fallback_used": state.get("fallback_used", False),
        "started_at": state.get("started_at", ""),
        "finished_at": state.get("finished_at", ""),
        "duration_ms": state.get("duration_ms", 0),
        "error_message": state.get("error_message"),
    }