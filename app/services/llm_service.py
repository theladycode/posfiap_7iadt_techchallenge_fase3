from openai import OpenAI
from app.core.settings import settings
from app.core.config import AppConfig

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_answer(question: str, patient_context: str, protocol_context: str) -> str:
    prompt = f"""
Contexto do paciente:
{patient_context}

Contexto do protocolo:
{protocol_context}

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

    return response.choices[0].message.content or ""