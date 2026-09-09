from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FastAPI Education Assistant - RAG API"
    app_version: str = "1.0.0"
    debug: bool = True
    secret_key: str = "change-this-secret-key"
    access_token_expire_minutes: int = 30
    upload_dir: str = "uploads"
    log_file: str = "logs/questions.txt"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
