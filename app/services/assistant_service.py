from app.services.llm_service import generate_answer


def handle_question(question: str) -> str:
    return generate_answer(question)