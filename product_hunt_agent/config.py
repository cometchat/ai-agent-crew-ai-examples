from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_ROOT = Path(__file__).resolve().parent


class ProductHuntSettings(BaseSettings):
    """Configuration for the CrewAI Product Hunt agent service."""

    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="PRODUCT_OPENAI_MODEL")
    base_url: Optional[str] = Field(default=None, validation_alias="OPENAI_BASE_URL")
    producthunt_api_token: Optional[str] = Field(default=None, validation_alias="PRODUCTHUNT_API_TOKEN")
    default_timezone: str = Field(default="America/New_York", validation_alias="PRODUCTHUNT_DEFAULT_TIMEZONE")
    http_timeout_seconds: float = Field(default=20.0, validation_alias="PRODUCTHUNT_HTTP_TIMEOUT", ge=1.0, le=60.0)
    algolia_app_id: str = Field(default="0H4SMABBSG", validation_alias="PRODUCTHUNT_ALGOLIA_APP_ID")
    algolia_api_key: str = Field(
        default="9670d2d619b9d07859448d7628eea5f3",
        validation_alias="PRODUCTHUNT_ALGOLIA_API_KEY",
    )

    model_config = SettingsConfigDict(
        env_file=(str(PACKAGE_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> ProductHuntSettings:
    return ProductHuntSettings()


__all__ = ["ProductHuntSettings", "get_settings"]
