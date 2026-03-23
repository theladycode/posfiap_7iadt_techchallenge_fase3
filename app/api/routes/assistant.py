import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.schemas.assistant import AssistantRequest, AssistantResponse
from app.schemas.audit import AuditDetailResponse
from app.graph.workflow import assistant_workflow
from app.services.audit_repository import save_audit_log, get_audit_log

router = APIRouter(prefix="/assistant", tags=["Assistant"])


@router.post("/query", response_model=AssistantResponse)
def query_assistant(request: AssistantRequest):
    request_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    initial_state = {
        "request_id": request_id,
        "patient_id": request.patient_id,
        "question": request.question,
        "user_role": request.user_role,
        "context": request.context or "",
        "patient_context": "",
        "protocol_context": "",
        "protocol_sources": [],
        "answer": "",
        "sources": [],
        "risk_level": "low",
        "requires_human_validation": False,
        "audit_log": {},
        "llm_provider": "",
        "model_name": "",
        "supporting_model_output": "",
        "supporting_decision": "",
        "status": "success",
        "execution_mode": "",
        "fallback_used": False,
        "started_at": started_at.isoformat(),
        "finished_at": "",
        "duration_ms": 0,
    }

    try:
        result = assistant_workflow.invoke(initial_state)

        finished_at = datetime.now(timezone.utc)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result["finished_at"] = finished_at.isoformat()
        result["duration_ms"] = duration_ms

        result["audit_log"]["request_id"] = result["request_id"]
        result["audit_log"]["finished_at"] = result["finished_at"]
        result["audit_log"]["duration_ms"] = result["duration_ms"]
        result["audit_log"]["status"] = result.get("status", "success")
        result["audit_log"]["execution_mode"] = result.get("execution_mode", "")
        result["audit_log"]["fallback_used"] = result.get("fallback_used", False)

        save_audit_log(result["audit_log"])

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "risk_level": result["risk_level"],
            "requires_human_validation": result["requires_human_validation"],
            "audit": {
                "request_id": result["request_id"],
                "status": result.get("status", "success"),
                "execution_mode": result.get("execution_mode", result.get("llm_provider", "")),
                "fallback_used": result.get("fallback_used", False),
                "duration_ms": result["duration_ms"],
                "llm_provider": result.get("llm_provider", ""),
                "model_name": result.get("model_name", ""),
                "supporting_decision": result.get("supporting_decision", ""),
                "protocol_sources": result.get("protocol_sources", []),
            },
        }

    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        error_audit = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": request.patient_id,
            "question": request.question,
            "user_role": request.user_role,
            "context_used": {
                "user_context": request.context or "",
                "patient_context": "",
                "protocol_context": "",
            },
            "answer": "",
            "sources": [],
            "protocol_sources": [],
            "risk_level": "unknown",
            "requires_human_validation": False,
            "llm_provider": "",
            "model_name": "",
            "supporting_model_output": "",
            "supporting_decision": "",
            "status": "error",
            "execution_mode": "unknown",
            "fallback_used": False,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
            "error_message": str(exc),
        }

        save_audit_log(error_audit)
        raise

@router.get("/audits/{request_id}", response_model=AuditDetailResponse)
def get_audit_by_request_id(request_id: str):
    audit_log = get_audit_log(request_id)

    if not audit_log:
        raise HTTPException(status_code=404, detail="Audit log not found")

    return audit_log