from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_ROOT = Path(__file__).resolve().parent


class KnowledgeAgentSettings(BaseSettings):
    """Configuration for the CrewAI knowledge agent."""

    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="KNOWLEDGE_OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="KNOWLEDGE_EMBEDDING_MODEL")
    base_url: Optional[str] = Field(default=None, validation_alias="OPENAI_BASE_URL")
    temperature: float = 0.3
    knowledge_root: Path = Field(
        default=PACKAGE_ROOT / "data" / "knowledge",
        description="Location for persisted markdown documents",
    )
    chroma_path: Path = Field(
        default=PACKAGE_ROOT / "data" / "chroma",
        description="Location for Chroma vector store files",
    )
    max_tool_results: int = 6
    allow_remote_http: bool = True
    remote_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(
        env_file=(str(PACKAGE_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> KnowledgeAgentSettings:
    return KnowledgeAgentSettings()


__all__ = ["KnowledgeAgentSettings", "get_settings"]
