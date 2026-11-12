"""Redacted env inspection — never prints secret values."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def _classify_host(host: str | None) -> str:
    if not host:
        return "none"
    hl = host.lower()
    if hl.startswith("db.") and "supabase.co" in hl:
        return "supabase_direct_db"
    if "pooler.supabase" in hl:
        return "supabase_pooler"
    if "supabase.co" in hl:
        return "supabase_api_host"
    if "amazonaws.com" in hl or "neon.tech" in hl:
        return "other_cloud_db"
    return "other_host"


def inspect_key(name: str, value: str) -> dict:
    p = urlparse(value)
    scheme = p.scheme or ("none" if "://" not in value else "unknown")
    return {
        "key": name,
        "scheme": scheme,
        "host_category": _classify_host(p.hostname),
        "port": p.port,
        "database": (p.path or "/").lstrip("/") or None,
        "looks_like_postgres": bool(
            scheme.startswith("postgres")
            or value.lower().startswith("postgres")
            or value.lower().startswith("postgresql")
        ),
        "looks_like_https_api": scheme in {"http", "https"},
        "length": len(value),
        "has_userinfo": bool(p.username or p.password),
    }


def main() -> None:
    vals: dict[str, str] = {}
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        vals[k.strip().upper()] = v.strip().strip("\"'")

    print("HAS_DATABASE_URL", "DATABASE_URL" in vals)
    print("ENV_KEYS", ",".join(sorted(vals)))
    print("AUTH_ENABLED", vals.get("AUTH_ENABLED", "MISSING").lower())
    auth_url = vals.get("SUPABASE_AUTH_URL", "")
    print("SUPABASE_AUTH_URL", inspect_key("SUPABASE_AUTH_URL", auth_url) if auth_url else "MISSING")
    publishable = vals.get("SUPABASE_PUBLISHABLE_KEY") or vals.get(
        "SUPABASE_ANON_KEY", ""
    )
    print(
        "SUPABASE_PUBLISHABLE_KEY",
        {
            "present": bool(publishable),
            "key_type": (
                "publishable"
                if publishable.startswith("sb_publishable_")
                else "legacy_anon_or_unknown"
            ),
            "length": len(publishable),
        },
    )
    print("SUPABASE_JWT_SECRET_PRESENT", bool(vals.get("SUPABASE_JWT_SECRET")))
    for key in ("SUPABASE_URL", "SUPABASE_BACKEND", "STORAGE_BACKEND", "S3_BUCKET_NAME"):
        if key not in vals:
            print(key, "MISSING")
            continue
        info = inspect_key(key, vals[key])
        for k, v in info.items():
            print(f"{k}={v}")
        print("---")


if __name__ == "__main__":
    main()
