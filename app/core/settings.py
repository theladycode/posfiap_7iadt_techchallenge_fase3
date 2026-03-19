from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ENV: str = "dev"
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
print("API KEY carregada?", settings.OPENAI_API_KEY[:10])