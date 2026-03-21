from openai import OpenAI
from app.core.settings import settings
from app.core.config import AppConfig


def get_openai_client():
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_openai_answer(
    question: str,
    context: str,
    patient_context: str,
    protocol_context: str,
    supporting_context: str = "",
) -> dict:
    client = get_openai_client()

    prompt = f"""
Contexto adicional informado na requisição:
{context}

Contexto do paciente:
{patient_context}

Contexto do protocolo:
{protocol_context}

Análise complementar do modelo biomédico fine-tuned:
{supporting_context}

Pergunta:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AppConfig.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=AppConfig.DEFAULT_TEMPERATURE,
    )

    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "provider": "openai",
        "model_name": "gpt-4o-mini",
    }