"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """Root settings — all env vars flow through here."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON

    # ── PostgreSQL ────────────────────────────────────────────
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "forex_agent"
    postgres_user: str = "forex"
    postgres_password: SecretStr = SecretStr("changeme_in_production")

    @property
    def database_url(self) -> str:
        pw = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Sync URL for Alembic migrations."""
        pw = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # ── Anthropic ─────────────────────────────────────────────
    anthropic_api_key: SecretStr = SecretStr("")
    claude_reasoning_model: str = "claude-sonnet-4-20250514"
    claude_fast_model: str = "claude-haiku-4-5-20251001"

    # ── Data Providers ────────────────────────────────────────
    polygon_api_key: SecretStr = SecretStr("")
    twelve_data_api_key: SecretStr = SecretStr("")
    finnhub_api_key: SecretStr = SecretStr("")
    newsapi_key: SecretStr = SecretStr("")
    fred_api_key: SecretStr = SecretStr("")

    # ── Telegram ──────────────────────────────────────────────
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""

    # ── Capital.com Broker ────────────────────────────────────
    capital_api_key: SecretStr = SecretStr("")
    capital_email: str = ""
    capital_password: SecretStr = SecretStr("")
    capital_environment: str = "demo"  # "demo" or "live"

    # ── Trading ───────────────────────────────────────────────
    initial_capital: float = 10_000.0
    paper_trading: bool = True
    auto_execute: bool = False

    # ── Monitoring ────────────────────────────────────────────
    prometheus_port: int = 9100

    # ── Derived ───────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == AppEnv.DEVELOPMENT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance. Call this everywhere."""
    return Settings()
