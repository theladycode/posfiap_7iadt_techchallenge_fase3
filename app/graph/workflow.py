from langgraph.graph import StateGraph, END
from app.graph.state import AssistantState
from app.graph.nodes import (
    load_patient_context,
    load_protocol_context,
    generate_llm_response,
    validate_risk,
    create_audit,
)


def build_workflow():
    graph = StateGraph(AssistantState)

    graph.add_node("load_patient_context", load_patient_context)
    graph.add_node("load_protocol_context", load_protocol_context)
    graph.add_node("generate_llm_response", generate_llm_response)
    graph.add_node("validate_risk", validate_risk)
    graph.add_node("create_audit", create_audit)

    graph.set_entry_point("load_patient_context")

    graph.add_edge("load_patient_context", "load_protocol_context")
    graph.add_edge("load_protocol_context", "generate_llm_response")
    graph.add_edge("generate_llm_response", "validate_risk")
    graph.add_edge("validate_risk", "create_audit")
    graph.add_edge("create_audit", END)

    return graph.compile()


assistant_workflow = build_workflow()