import logging

from app.core.settings import settings
from app.services.providers.openai_provider import generate_openai_answer

logger = logging.getLogger(__name__)


def generate_answer(
    question: str,
    context: str,
    patient_context: str,
    protocol_context: str,
) -> dict:
    provider = settings.LLM_PROVIDER.lower()

    logger.info("Gerando resposta com provider: %s", provider)

    if provider == "finetuned":
        from app.services.providers.finetuned_provider import generate_finetuned_answer

        fine_tuned_result = generate_finetuned_answer(
            question=question,
            context=context,
            patient_context=patient_context,
            protocol_context=protocol_context,
        )

        return {
            "answer": fine_tuned_result["answer"],
            "provider": "finetuned",
            "model_name": fine_tuned_result.get("model_name", ""),
            "supporting_model_output": fine_tuned_result.get("answer", ""),
            "supporting_decision": fine_tuned_result.get("decision", ""),
            "status": "success",
            "execution_mode": "finetuned",
            "fallback_used": False,
        }

    if provider == "hybrid":
        from app.services.providers.finetuned_provider import generate_finetuned_answer

        fine_tuned_result = generate_finetuned_answer(
            question=question,
            context=context,
            patient_context=patient_context,
            protocol_context=protocol_context,
        )

        openai_result = generate_openai_answer(
            question=question,
            context=context,
            patient_context=patient_context,
            protocol_context=protocol_context,
            supporting_context=fine_tuned_result["answer"],
        )

        return {
            "answer": openai_result["answer"],
            "provider": "hybrid",
            "model_name": f'{openai_result["model_name"]} + {fine_tuned_result["model_name"]}',
            "supporting_model_output": fine_tuned_result["answer"],
            "supporting_decision": fine_tuned_result.get("decision", ""),
            "status": "success",
            "execution_mode": "hybrid",
            "fallback_used": True,
        }

    openai_result = generate_openai_answer(
        question=question,
        context=context,
        patient_context=patient_context,
        protocol_context=protocol_context,
    )

    return {
        "answer": openai_result["answer"],
        "provider": "openai",
        "model_name": openai_result.get("model_name", ""),
        "supporting_model_output": "",
        "supporting_decision": "",
        "status": "success",
        "execution_mode": "openai",
        "fallback_used": False,
    }