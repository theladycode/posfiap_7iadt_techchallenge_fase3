from fastapi import APIRouter
from app.schemas.assistant import AssistantRequest, AssistantResponse
from app.graph.workflow import assistant_workflow

router = APIRouter(prefix="/assistant", tags=["Assistant"])


@router.post("/query", response_model=AssistantResponse)
def query_assistant(request: AssistantRequest):
    initial_state = {
        "patient_id": request.patient_id,
        "question": request.question,
        "user_role": request.user_role,
        "patient_context": "",
        "protocol_context": "",
        "answer": "",
        "sources": [],
        "risk_level": "low",
        "requires_human_validation": False,
        "audit_log": {},
    }

    result = assistant_workflow.invoke(initial_state)

    print("AUDIT LOG:", result["audit_log"])

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "risk_level": result["risk_level"],
        "requires_human_validation": result["requires_human_validation"],
    }