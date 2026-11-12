"""Write PaperLens runtime secrets to AWS Secrets Manager without printing their values.

Reads the repository `.env` and stores:
  OPENROUTER_API_KEY  -> paperlens/prod/llm-api-key   (key for the API's LLM_BASE_URL)
  OPENAI_API_KEY      -> paperlens/prod/openai-api-key (kept for switching back to OpenAI)
  PROD_DATABASE_URL   -> paperlens/prod/database-url

The secret containers are created by Terraform (infra/prod/storage.tf); only the
values are written here so they never land in Terraform state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REGION = "ap-northeast-1"

SECRET_SOURCES = {
    "paperlens/prod/llm-api-key": ("OPENROUTER_API_KEY",),
    "paperlens/prod/openai-api-key": ("OPENAI_API_KEY", "LLM_API_KEY"),
    "paperlens/prod/database-url": ("PROD_DATABASE_URL",),
}


def _load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip().upper()] = value.strip().strip("\"'")
    return values


def _put_secret(secret_id: str, value: str) -> int:
    fd, tmp_name = tempfile.mkstemp(prefix="paperlens-secret-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        completed = subprocess.run(
            [
                "aws",
                "secretsmanager",
                "put-secret-value",
                f"--region={REGION}",
                f"--secret-id={secret_id}",
                f"--secret-string=file://{tmp_name}",
                "--query=VersionId",
                "--output=text",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    if completed.returncode != 0:
        print(f"{secret_id}: FAILED ({completed.stderr.strip().splitlines()[-1:]})")
    else:
        print(f"{secret_id}: updated (version {completed.stdout.strip()[:8]}..., length {len(value)})")
    return completed.returncode


def main() -> int:
    env = _load_env()
    failures = 0
    for secret_id, keys in SECRET_SOURCES.items():
        value = next((env[key] for key in keys if env.get(key)), "")
        if not value:
            print(f"{secret_id}: SKIPPED (none of {', '.join(keys)} set in .env)")
            failures += 1
            continue
        failures += 1 if _put_secret(secret_id, value) else 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
