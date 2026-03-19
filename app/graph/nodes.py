from app.graph.state import AssistantState
from app.services.patient_service import get_patient_context
from app.services.protocol_service import get_protocol_context
from app.services.llm_service import generate_answer
from app.services.audit_service import create_audit_log


def load_patient_context(state: AssistantState) -> AssistantState:
    patient_context = get_patient_context(state["patient_id"])
    state["patient_context"] = patient_context
    return state


def load_protocol_context(state: AssistantState) -> AssistantState:
    protocol_context, protocol_sources = get_protocol_context(state["question"])
    state["protocol_context"] = protocol_context
    state["sources"] = ["mock_patient_record", *protocol_sources]
    return state


def generate_llm_response(state: AssistantState) -> AssistantState:
    answer = generate_answer(
        question=state["question"],
        patient_context=state["patient_context"],
        protocol_context=state["protocol_context"],
    )
    state["answer"] = answer
    return state


def validate_risk(state: AssistantState) -> AssistantState:
    question_lower = state["question"].lower()
    answer_lower = state["answer"].lower()

    high_risk_terms = [
        "prescrever",
        "prescrição",
        "dose",
        "dosagem",
        "medicamento",
        "remédio",
        "tratamento",
        "cirurgia",
        "internação",
        "antibiótico",
    ]

    is_risky_question = any(term in question_lower for term in high_risk_terms)
    is_risky_answer = any(term in answer_lower for term in high_risk_terms)
    is_risky = is_risky_question or is_risky_answer

    state["risk_level"] = "high" if is_risky else "low"
    state["requires_human_validation"] = is_risky

    return state


def create_audit(state: AssistantState) -> AssistantState:
    state["audit_log"] = create_audit_log(state)
    return state