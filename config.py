from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b"
    openweather_api_key: str = ""
    tavily_api_key: str = ""

    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "tufin-agent"

    database_url: str = "./data/tasks.db"
    model: str = "ollama:llama3"


settings = Settings()
