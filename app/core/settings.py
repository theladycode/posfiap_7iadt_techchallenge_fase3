from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Ambiente
    ENV: str = "dev"
    DEBUG: bool = True

    # Estratégia de provider
    LLM_PROVIDER: str = "openai"  # openai | finetuned | hybrid

    # Modelo fine-tuned / Hugging Face
    HF_MODEL_ID: str = ""
    MODEL_PATH: str = ""
    BASE_MODEL_NAME: str = "Qwen/Qwen3.5-0.8B"
    HF_TOKEN: str = ""

    # Inferência do modelo fine-tuned
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    MAX_SEQ_LENGTH: int = 2048

    # Execução
    FORCE_CPU: bool = True


settings = Settings()

print("OPENAI_API_KEY carregada?", bool(settings.OPENAI_API_KEY))
print("HF_MODEL_ID configurado?", bool(settings.HF_MODEL_ID))
print("MODEL_PATH configurado?", bool(settings.MODEL_PATH))
print("LLM_PROVIDER:", settings.LLM_PROVIDER)
print("HF_TOKEN configurado?", bool(settings.HF_TOKEN))