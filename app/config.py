"""Configurações centralizadas da aplicação carregadas via variáveis de ambiente."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros de modelos Ollama, ChromaDB, chunking e API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    llm_model: str = "qwen2.5:3b"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "editais_collection"
    chunk_size: int = 800
    chunk_overlap: int = 150
    default_top_k: int = 4
    ollama_timeout: float = 120.0
    max_upload_size_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    """Retorna instância cacheada de Settings."""
    return Settings()


settings = get_settings()
