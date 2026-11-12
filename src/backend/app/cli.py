"""PaperLens command-line utilities."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _cmd_check_database(_: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.db.session import check_database, reset_engine

    clear = getattr(get_settings, "cache_clear", None)
    if callable(clear):
        clear()
    reset_engine()
    settings = get_settings()
    try:
        result = check_database(settings)
    except Exception as exc:
        print("Database connection: FAILED")
        print(f"Error type: {type(exc).__name__}")
        # Never include connection URL in error text if present
        message = str(exc)
        for secretish in (settings.database_url, settings.supabase_url or ""):
            if secretish and secretish in message:
                message = message.replace(secretish, "[REDACTED]")
        print(f"Error: {message}")
        return 1

    print("Database connection: OK")
    print(f"Backend: {result.get('backend')}")
    print(f"Connection mode: {result.get('connection_mode')}")
    print(f"Database: {result.get('database')}")
    print(f"Port: {result.get('port')}")
    print(f"SSL: {'enabled' if result.get('ssl_enabled') else 'unknown/disabled'}")
    if result.get("version_preview"):
        print(f"Version: {result['version_preview']}")
    return 0


def _cmd_init_db(_: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.db.session import init_db, reset_engine
    from app.db.url_utils import classify_database_url

    clear = getattr(get_settings, "cache_clear", None)
    if callable(clear):
        clear()
    reset_engine()
    settings = get_settings()
    info = classify_database_url(settings.effective_migration_url)
    print(f"Initializing schema (mode={info.connection_mode}, db={info.database})")
    init_db(settings, url=settings.effective_migration_url)
    print("Schema initialization: OK")
    return 0


def _cmd_db_info(_: argparse.Namespace) -> int:
    from app.config import get_settings

    clear = getattr(get_settings, "cache_clear", None)
    if callable(clear):
        clear()
    settings = get_settings()
    print(json.dumps(settings.database_info.as_public_dict(), indent=2))
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    """Re-chunk and re-index stored papers with the current chunking/context rules."""
    from sqlalchemy import select

    from app.config import get_settings
    from app.db.models import Paper
    from app.db.session import reset_engine, session_scope
    from app.ingestion.pipeline import rebuild_paper_index

    clear = getattr(get_settings, "cache_clear", None)
    if callable(clear):
        clear()
    reset_engine()
    settings = get_settings()
    if args.paper_id:
        paper_ids = list(args.paper_id)
    else:
        with session_scope(settings) as session:
            paper_ids = list(session.scalars(select(Paper.id).where(Paper.status == "completed")))
    failures = 0
    for paper_id in paper_ids:
        try:
            summary = rebuild_paper_index(
                paper_id, settings=settings, refresh_context=not args.keep_context, enrich=args.enrich
            )
            print(json.dumps({k: v for k, v in summary.items() if k != "index"}))
        except Exception as exc:
            failures += 1
            print(f"{paper_id}: FAILED ({type(exc).__name__})")
    print(f"Re-indexed {len(paper_ids) - failures}/{len(paper_ids)} papers")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="PaperLens CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check-database", help="Verify database connectivity (redacted)")
    p_check.set_defaults(func=_cmd_check_database)

    p_init = sub.add_parser("init-db", help="Create tables if missing")
    p_init.set_defaults(func=_cmd_init_db)

    p_info = sub.add_parser("db-info", help="Print redacted database URL classification")
    p_info.set_defaults(func=_cmd_db_info)

    p_reindex = sub.add_parser(
        "reindex", help="Re-chunk and re-index stored papers (small-to-big, table data, context)"
    )
    p_reindex.add_argument("--paper-id", action="append", help="Only this paper (repeatable)")
    p_reindex.add_argument(
        "--enrich", action="store_true", help="Also run Luna on visuals (one VLM request each)"
    )
    p_reindex.add_argument(
        "--keep-context", action="store_true", help="Do not recompute surrounding text"
    )
    p_reindex.set_defaults(func=_cmd_reindex)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
