"""Database engine and session helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.db.models import Base
from app.db.url_utils import classify_database_url, ensure_sslmode, normalize_sqlalchemy_url

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_sqlite_parent(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    raw = url.removeprefix("sqlite:///")
    if raw in {":memory:", ""} or raw.startswith("file:"):
        return
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_connect_args(url: str, settings: Settings) -> dict:
    info = classify_database_url(url)
    if info.dialect.startswith("sqlite"):
        return {"check_same_thread": False}
    if not info.dialect.startswith("postgres"):
        return {}

    # psycopg3 connect args
    args: dict = {
        "connect_timeout": settings.database_connect_timeout_seconds,
        "options": f"-c statement_timeout={settings.database_statement_timeout_ms}",
    }
    # Prefer libpq sslmode via URL query; also set prepare_threshold=None for transaction poolers
    # so SQLAlchemy/psycopg avoid prepared-statement session state.
    if info.connection_mode == "transaction_pooler":
        args["prepare_threshold"] = None
    return args


def get_engine(settings: Settings | None = None, *, url: str | None = None) -> Engine:
    global _engine, _SessionLocal
    cfg = settings or get_settings()
    database_url = normalize_sqlalchemy_url(url or cfg.database_url)
    if _engine is not None and url is None:
        return _engine

    info = classify_database_url(database_url)
    _ensure_sqlite_parent(database_url)

    if info.dialect.startswith("postgres"):
        database_url = ensure_sslmode(database_url, default="require")

    connect_args = _build_connect_args(database_url, cfg)
    engine_kwargs: dict = {
        "echo": cfg.database_echo,
        "future": True,
        "connect_args": connect_args,
    }

    if info.connection_mode == "transaction_pooler" or (
        info.dialect.startswith("postgres") and info.port == 6543
    ):
        engine_kwargs["poolclass"] = NullPool
        logger.info(
            "Using NullPool for PostgreSQL connection_mode=%s port=%s",
            info.connection_mode,
            info.port,
        )
    elif info.dialect.startswith("postgres"):
        engine_kwargs["pool_pre_ping"] = True

    public = info.as_public_dict()
    logger.info("Creating DB engine %s", public)

    engine = create_engine(database_url, **engine_kwargs)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_fk(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    if url is None:
        _engine = engine
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        get_engine(settings)
    assert _SessionLocal is not None
    return _SessionLocal


def init_db(settings: Settings | None = None, *, url: str | None = None) -> Engine:
    """Create all tables (SQLite-friendly migration bootstrap)."""
    engine = get_engine(settings, url=url)
    Base.metadata.create_all(bind=engine)
    _apply_v02_columns(engine)
    logger.info("Database schema initialized")
    return engine


def _apply_v02_columns(engine: Engine) -> None:
    """Add nullable-compatible account/session columns to pre-v0.2 databases."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        if "papers" in tables:
            columns = {column["name"] for column in inspector.get_columns("papers")}
            if "user_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE papers ADD COLUMN user_id VARCHAR(64) "
                        "NOT NULL DEFAULT 'local-user'"
                    )
                )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_papers_user_id ON papers (user_id)")
            )
        if "agent_conversations" in tables:
            columns = {
                column["name"] for column in inspector.get_columns("agent_conversations")
            }
            if "user_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE agent_conversations ADD COLUMN user_id VARCHAR(64) "
                        "NOT NULL DEFAULT 'local-user'"
                    )
                )
            if "title" not in columns:
                connection.execute(
                    text("ALTER TABLE agent_conversations ADD COLUMN title VARCHAR(512)")
                )
            # v0.3: context memory (see migrations/004_conversation_memory.sql).
            for name, ddl in (
                ("kind", "VARCHAR(32) NOT NULL DEFAULT 'agent'"),
                ("context_summary", "TEXT"),
                ("summarized_turns", "INTEGER NOT NULL DEFAULT 0"),
                ("context_state", "JSON"),
            ):
                if name not in columns:
                    connection.execute(
                        text(f"ALTER TABLE agent_conversations ADD COLUMN {name} {ddl}")
                    )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_agent_conversations_user_id "
                    "ON agent_conversations (user_id)"
                )
            )


def check_database(settings: Settings | None = None) -> dict[str, object]:
    """Connect and run SELECT 1. Returns redacted diagnostics."""
    cfg = settings or get_settings()
    if not cfg.database_url:
        raise ValueError("DATABASE_URL is missing")
    info = classify_database_url(cfg.database_url)
    if info.connection_mode == "unknown" and not info.dialect.startswith(("sqlite", "postgres")):
        raise ValueError("Malformed or unsupported DATABASE_URL")

    engine = get_engine(cfg)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        extras: dict[str, object] = {}
        if info.dialect.startswith("postgres"):
            row = conn.execute(
                text("SELECT current_database(), current_user, version()")
            ).one()
            extras["database"] = row[0]
            extras["user_redacted"] = True
            extras["version_preview"] = str(row[2]).split(",")[0]
        elif info.dialect.startswith("sqlite"):
            extras["database"] = info.database
            extras["version_preview"] = "sqlite"
    return {
        "ok": True,
        "backend": "PostgreSQL" if info.dialect.startswith("postgres") else info.dialect,
        **info.as_public_dict(),
        **extras,
    }


def reset_engine() -> None:
    """Clear cached engine (tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    factory = get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()