"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.db.url_utils import classify_database_url, normalize_sqlalchemy_url


class Settings(BaseSettings):
    """Runtime settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    storage_backend: Literal["local", "s3"] = "local"
    local_storage_root: Path = Path("runtime/outputs")

    s3_bucket_name: str = ""
    aws_region: str | None = None

    docling_ocr_mode: Literal["off", "on", "auto"] = "auto"
    docling_images_scale: float = 1.5
    docling_threads: int = 1
    docling_accelerator_device: Literal["auto", "cpu", "cuda"] = "auto"
    docling_cuda_use_flash_attention2: bool = False
    docling_do_formula_enrichment: bool = False
    docling_do_code_enrichment: bool = False

    luna_enabled: bool = False
    luna_provider: str = "openai"
    luna_model: str = Field(
        default="",
        validation_alias=AliasChoices("LUNA_MODEL", "BIDPILOT_OPENAI_ANSWER_MODEL"),
    )
    luna_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LUNA_API_KEY", "BIDPILOT_OPENAI_API_KEY"),
    )
    luna_base_url: str | None = None
    luna_max_retries: int = 2
    luna_request_delay_seconds: float = 1.0
    luna_timeout_seconds: float = 60.0
    luna_prompt_version: str = "v1"
    luna_schema_version: str = "v1"
    text_enrichment_enabled: bool = False
    text_enrichment_model: str = ""
    text_enrichment_max_sections: int = 8
    text_enrichment_max_chars_per_section: int = 6000
    text_enrichment_max_total_chars: int = 30000
    ingest_auto_text_enrich: bool = False
    allow_external_api: bool = False

    max_pdf_size_mb: int = 50
    asset_crop_padding_px: int = 8
    log_level: str = "INFO"

    database_url: str = "sqlite:///./runtime/data/paperlens.db"
    # Accepted fallback when users store the Postgres DSN under SUPABASE_URL.
    supabase_url: str | None = None
    migration_database_url: str | None = None
    database_echo: bool = False
    database_connect_timeout_seconds: int = 10
    database_statement_timeout_ms: int = 30000
    ingest_async: bool = False

    # Supabase Auth. Disabled by default so local development remains self-contained.
    auth_enabled: bool = False
    supabase_auth_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_AUTH_URL", "SUPABASE_URL"),
    )
    supabase_jwks_url: str | None = None
    supabase_jwt_secret: str | None = None
    supabase_jwt_audience: str = "authenticated"
    supabase_jwt_issuer: str | None = None

    # Anonymous trial accounts for try-before-signup demos.
    guest_trial_enabled: bool = True
    guest_max_queries: int = 10
    guest_max_papers: int = 2
    guest_max_images: int = 2
    guest_session_ttl_hours: int = 72

    embedding_provider: str = Field(
        default="hashing",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER", "BIDPILOT_EMBEDDING_PROVIDER"),
    )
    embedding_model: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "BIDPILOT_OPENAI_EMBEDDING_MODEL"),
    )
    embedding_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "BIDPILOT_OPENAI_API_KEY"),
    )
    embedding_base_url: str | None = None
    embedding_dimensions: int = 384
    # Self-hosted servers (e.g. TEI) serve fixed-size models and may reject `dimensions`.
    embedding_send_dimensions: bool = True
    # Texts per embeddings request; TEI caps client batches (32 by default).
    embedding_batch_size: int | None = None
    retrieval_top_k: int = 8
    retrieval_use_mmr: bool = False

    llm_enabled: bool = False
    llm_provider: str = "openai"
    llm_model: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_MODEL", "BIDPILOT_OPENAI_ANSWER_MODEL"),
    )
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "BIDPILOT_OPENAI_API_KEY"),
    )
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    paperlens_api_base: str = "http://127.0.0.1:8000"
    cors_allowed_origins: str = (
        "http://127.0.0.1:3000,http://localhost:3000,"
        "http://127.0.0.1:8080,http://localhost:8080,"
        "https://paperlens.ailearn-vaic.cloud"
    )
    cors_allow_origin_regex: str = ""
    langsmith_enabled: bool = False
    langsmith_tracing: bool = False
    allow_paid_evaluation: bool = False

    # Unified agent request/response safety limits (deterministic guardrails).
    agent_max_message_chars: int = 8000
    agent_max_selected_papers: int = 8
    agent_max_image_bytes: int = 2 * 1024 * 1024
    agent_history_limit: int = 24
    agent_reasoning_enabled: bool = True
    agent_reasoning_effort: Literal["low", "medium", "high"] = "medium"

    @field_validator("local_storage_root", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        return Path(str(value))

    @model_validator(mode="after")
    def _resolve_database_url(self) -> Settings:
        default_sqlite = "sqlite:///./runtime/data/paperlens.db"
        url = self.database_url
        if (not url or url == default_sqlite) and self.supabase_url:
            candidate = self.supabase_url.strip()
            if candidate.lower().startswith(("postgres://", "postgresql://", "postgresql+")):
                url = candidate
        self.database_url = normalize_sqlalchemy_url(url)
        if self.migration_database_url:
            self.migration_database_url = normalize_sqlalchemy_url(self.migration_database_url)
        return self

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def database_info(self):
        return classify_database_url(self.database_url)

    @property
    def effective_migration_url(self) -> str:
        return self.migration_database_url or self.database_url

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
