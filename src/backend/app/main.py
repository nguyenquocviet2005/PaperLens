"""PaperLens FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.db.session import init_db
from app.ingestion.parser import apply_thread_limits
from app.routes import router

settings = get_settings()
apply_thread_limits(settings.docling_threads)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="PaperLens",
    version=__version__,
    description="Research-paper ingestion, Docling parsing, and optional Luna enrichment.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_allow_origin_regex or None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    logging.getLogger(__name__).info(
        "PaperLens starting env=%s storage=%s db=%s luna_enabled=%s allow_external_api=%s",
        settings.app_env,
        settings.storage_backend,
        settings.database_url.split("://", 1)[0],
        settings.luna_enabled,
        settings.allow_external_api,
    )
    try:
        init_db(settings)
    except Exception:
        logging.getLogger(__name__).exception("Database init failed")
        raise
