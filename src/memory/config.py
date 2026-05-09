from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    database_url: str = "postgresql://memory:memory@db:5432/memory"

    alem_api_key: str = ""
    alem_llm_base_url: str = "https://llm.alem.ai/v1"
    alem_llm_model: str = "alemllm"

    embed_api_key: str = ""
    embed_base_url: str = "https://llm.alem.ai/v1"
    embed_model: str = "text-1024"
    embed_dim: int = 1024

    rerank_api_key: str = ""
    rerank_base_url: str = "https://llm.alem.ai/v1"
    rerank_model: str = "reranker"

    memory_auth_token: str = ""

    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.alem_api_key)

    @property
    def embed_enabled(self) -> bool:
        return bool(self.embed_api_key)

    @property
    def rerank_enabled(self) -> bool:
        return bool(self.rerank_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
