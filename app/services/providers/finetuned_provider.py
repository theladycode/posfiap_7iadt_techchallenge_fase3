import logging
from app.models.model import get_model_manager

logger = logging.getLogger(__name__)

_model_initialized = False


def ensure_finetuned_model_loaded():
    global _model_initialized
    manager = get_model_manager()

    if not manager.is_loaded and not _model_initialized:
        logger.info("Carregando modelo fine-tuned sob demanda...")
        manager.load()
        _model_initialized = True

    return manager


def generate_finetuned_answer(
    question: str,
    context: str,
    patient_context: str,
    protocol_context: str,
) -> dict:
    manager = ensure_finetuned_model_loaded()

    effective_context = context.strip() if context else ""

    if not effective_context:
        effective_context = f"""
Patient context:
{patient_context}

Protocol context:
{protocol_context}
""".strip()

    result = manager.predict(
        question=question,
        context=effective_context,
    )

    return {
        "answer": result["answer"],
        "decision": result["decision"],
        "elapsed_ms": result["elapsed_ms"],
        "model_name": result["model_name"],
        "provider": "finetuned",
    }