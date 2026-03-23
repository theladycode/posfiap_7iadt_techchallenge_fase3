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
    state["protocol_sources"] = protocol_sources
    state["sources"] = ["mock_patient_record", *protocol_sources]
    return state


def generate_llm_response(state: AssistantState) -> AssistantState:
    llm_result = generate_answer(
        question=state["question"],
        context=state["context"],
        patient_context=state["patient_context"],
        protocol_context=state["protocol_context"],
    )

    state["answer"] = llm_result["answer"]
    state["llm_provider"] = llm_result.get("provider", "unknown")
    state["model_name"] = llm_result.get("model_name", "")
    state["supporting_model_output"] = llm_result.get("supporting_model_output", "")
    state["supporting_decision"] = llm_result.get("supporting_decision", "")
    state["status"] = llm_result.get("status", "success")
    state["execution_mode"] = llm_result.get("execution_mode", state["llm_provider"])
    state["fallback_used"] = llm_result.get("fallback_used", False)

    if state["llm_provider"] == "hybrid":
        state["sources"] = [
            *state["sources"],
            "fine_tuned_biomedical_model",
            "openai_gpt_4o_mini",
        ]
    elif state["llm_provider"] == "finetuned":
        state["sources"] = [*state["sources"], "fine_tuned_biomedical_model"]
    else:
        state["sources"] = [*state["sources"], "openai_gpt_4o_mini"]

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