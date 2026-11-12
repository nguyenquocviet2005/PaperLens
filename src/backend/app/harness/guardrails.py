"""Input and output safety controls for the unified research agent.

These are deterministic request/response guardrails (length, MIME, secret
redaction, HTML sanitization). They are not a moderation or trust-and-safety
classifier.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.infrastructure.observability import count

# Dangerous active HTML / script patterns to neutralize in model output.
_DANGEROUS_HTML_RE = re.compile(
    r"(?is)"
    r"<script\b[^>]*>.*?</script>"
    r"|</?\s*(?:iframe|object|embed|link|meta|base|form)\b[^>]*>"
    r"|javascript\s*:"
    r"|vbscript\s*:"
    r"|on(?:error|load|click|mouse\w+|focus|blur|submit|input)\s*="
)

# Prompt / secret exfiltration style patterns (heuristic, not moderation).
_EXFIL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?is)\b(?:ignore|disregard|forget)\b.{0,40}\b(?:previous|above|system)\b"
            r".{0,40}\b(?:instructions?|prompt|rules?)\b"
        ),
        "Message attempts to override system instructions.",
    ),
    (
        re.compile(
            r"(?is)\b(?:reveal|show|print|dump|exfiltrate|leak)\b.{0,40}\b"
            r"(?:system\s*prompt|hidden\s*prompt|developer\s*message|"
            r"api[_\s-]?key|secret|token|password)\b"
        ),
        "Message requests system or secret material.",
    ),
    (
        re.compile(
            r"(?is)\b(?:repeat|output|display)\b.{0,30}\b(?:your|the)\b.{0,20}"
            r"\b(?:system\s*prompt|initial\s*instructions?)\b"
        ),
        "Message requests system prompt disclosure.",
    ),
    # Vietnamese equivalents of the patterns above.
    (
        re.compile(
            r"(?is)(?:bỏ\s*qua|phớt\s*lờ|lờ\s*đi|không\s*cần\s*tuân\s*theo)"
            r".{0,40}(?:hướng\s*dẫn|chỉ\s*dẫn|chỉ\s*thị|quy\s*tắc|prompt)"
            r".{0,40}(?:trước|trên|hệ\s*thống|ban\s*đầu|system)"
        ),
        "Message attempts to override system instructions.",
    ),
    (
        re.compile(
            r"(?is)(?:tiết\s*lộ|hiển\s*thị|in\s*ra|cho\s*(?:tôi|mình|tao)\s*xem|"
            r"lặp\s*lại|xuất\s*ra|nhắc\s*lại)"
            r".{0,40}(?:system\s*prompt|prompt\s*hệ\s*thống|lời\s*nhắc\s*hệ\s*thống|"
            r"chỉ\s*dẫn\s*hệ\s*thống|hướng\s*dẫn\s*hệ\s*thống|api[_\s-]?key|khóa\s*api|"
            r"khoá\s*api|mật\s*khẩu)"
        ),
        "Message requests system or secret material.",
    ),
    (
        re.compile(
            r"(?is)\b(?:LLM_API_KEY|LUNA_API_KEY|EMBEDDING_API_KEY|DATABASE_URL|"
            r"BIDPILOT_OPENAI_API_KEY)\b\s*[:=]"
        ),
        "Message contains credential assignment patterns.",
    ),
]

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)

ALLOWED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"})


class GuardrailError(ValueError):
    """Raised when an input fails deterministic safety checks."""

    def __init__(self, message: str, *, code: str = "guardrail_rejected") -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        count("GuardrailBlocks", dimensions={"Code": code})


@dataclass(frozen=True)
class ValidatedImage:
    mime: str
    data_url: str
    decoded_size: int


@dataclass(frozen=True)
class ValidatedAgentInput:
    message: str
    selected_papers: list[str]
    image: ValidatedImage | None = None


def _secret_values(settings: Settings) -> list[str]:
    candidates = [
        settings.llm_api_key,
        settings.luna_api_key,
        settings.embedding_api_key,
        settings.database_url,
        settings.migration_database_url,
        settings.supabase_url,
    ]
    values: list[str] = []
    for item in candidates:
        if not item:
            continue
        text = str(item).strip()
        if len(text) >= 8:
            values.append(text)
    return values


def validate_agent_input(
    message: str,
    *,
    selected_papers: list[str] | None = None,
    image_data_url: str | None = None,
    settings: Settings,
) -> ValidatedAgentInput:
    text = (message or "").strip()
    if not text and not image_data_url:
        raise GuardrailError("Message is required.", code="empty_message")

    max_len = settings.agent_max_message_chars
    if len(text) > max_len:
        raise GuardrailError(
            f"Message exceeds maximum length of {max_len} characters.",
            code="message_too_long",
        )

    if _CONTROL_CHAR_RE.search(text):
        raise GuardrailError(
            "Message contains disallowed control characters.",
            code="control_characters",
        )

    papers = list(dict.fromkeys(selected_papers or []))
    max_papers = settings.agent_max_selected_papers
    if len(papers) > max_papers:
        raise GuardrailError(
            f"At most {max_papers} papers may be selected.",
            code="too_many_papers",
        )

    for pattern, reason in _EXFIL_PATTERNS:
        if pattern.search(text):
            raise GuardrailError(reason, code="exfiltration_pattern")

    # Also reject if the user pasted a live configured secret into the prompt.
    for secret in _secret_values(settings):
        if secret in text:
            raise GuardrailError(
                "Message appears to include a configured secret value.",
                code="secret_in_input",
            )

    image: ValidatedImage | None = None
    if image_data_url:
        image = validate_image_data_url(image_data_url, settings=settings)

    return ValidatedAgentInput(message=text, selected_papers=papers, image=image)


def validate_image_data_url(data_url: str, *, settings: Settings) -> ValidatedImage:
    raw = (data_url or "").strip()
    if not raw.lower().startswith("data:"):
        raise GuardrailError(
            "Image must be a data URL (data:image/...;base64,...).",
            code="invalid_image",
        )
    match = _DATA_URL_RE.match(raw)
    if not match:
        raise GuardrailError(
            "Unsupported image data URL. Allowed MIME types: png, jpeg, webp, gif.",
            code="invalid_image_mime",
        )
    mime = match.group(1).strip().lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in ALLOWED_IMAGE_MIMES:
        raise GuardrailError(
            "Unsupported image data URL. Allowed MIME types: png, jpeg, webp, gif.",
            code="invalid_image_mime",
        )
    b64 = re.sub(r"\s+", "", match.group(2))
    try:
        decoded = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GuardrailError(
            "Image data URL is not valid base64.",
            code="invalid_image_base64",
        ) from exc

    max_bytes = settings.agent_max_image_bytes
    if len(decoded) > max_bytes:
        raise GuardrailError(
            f"Decoded image exceeds maximum size of {max_bytes} bytes.",
            code="image_too_large",
        )
    if len(decoded) == 0:
        raise GuardrailError("Image payload is empty.", code="empty_image")

    normalized = f"data:{mime};base64,{b64}"
    return ValidatedImage(mime=mime, data_url=normalized, decoded_size=len(decoded))


def redact_secrets(text: str, settings: Settings) -> str:
    out = text or ""
    for secret in _secret_values(settings):
        if secret in out:
            out = out.replace(secret, "[REDACTED]")
    return out


def sanitize_active_html(text: str) -> str:
    """Neutralize dangerous active HTML/JS constructs; not a full HTML sanitizer."""
    if not text:
        return text
    cleaned = _DANGEROUS_HTML_RE.sub("[removed]", text)
    # Also neutralize leftover event-handler attributes and javascript: URLs.
    cleaned = re.sub(r"(?i)\son[a-z]+\s*=\s*(['\"]).*?\1", "", cleaned)
    cleaned = re.sub(r"(?i)\son[a-z]+\s*=\s*[^\s>]+", "", cleaned)
    return cleaned


def apply_output_guardrails(
    result: dict[str, Any],
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Redact secrets, sanitize HTML, and check grounded integrity where feasible."""
    out = dict(result)
    answer = redact_secrets(str(out.get("answer") or ""), settings)
    answer = sanitize_active_html(answer)
    out["answer"] = answer

    citations = list(out.get("citations") or [])
    artifacts = list(out.get("artifacts") or [])
    grounded = bool(out.get("grounded"))

    # Soft integrity check: if marked grounded but no citations/artifacts, demote.
    grounded_kinds = {"paper_answer", "comparison", "research_report"}
    has_grounded_artifact = any(
        isinstance(a, dict) and a.get("kind") in grounded_kinds for a in artifacts
    )
    if grounded and not citations and not has_grounded_artifact:
        out["grounded"] = False
        out.setdefault("guardrail_notes", []).append(
            "grounded_flag_cleared_missing_evidence"
        )

    # Scrub secrets from nested JSON-ish string fields in artifacts/tool_calls.
    def _scrub(value: Any) -> Any:
        if isinstance(value, str):
            return sanitize_active_html(redact_secrets(value, settings))
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items()}
        return value

    out["artifacts"] = _scrub(artifacts)
    out["citations"] = _scrub(citations)
    out["tool_calls"] = _scrub(list(out.get("tool_calls") or []))
    return out


def guardrail_token(text: str, *, settings: Settings) -> str:
    """Apply lightweight output controls to one piece of text (no cross-token state)."""
    return sanitize_active_html(redact_secrets(text, settings))


class StreamGuard:
    """Redacts secrets across streamed tokens.

    Filtering each token on its own misses a secret split over two tokens: the user has
    already seen both halves by the time the final answer is redacted. The guard holds back
    the last len(longest secret) - 1 characters until more text (or the end) arrives, so a
    secret is always whole inside the buffer when it is checked.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.hold = max((len(s) for s in _secret_values(settings)), default=1) - 1
        self.buffer = ""

    def feed(self, text: str) -> str:
        self.buffer = redact_secrets(self.buffer + (text or ""), self.settings)
        if self.hold <= 0:
            out, self.buffer = self.buffer, ""
        else:
            cut = max(0, len(self.buffer) - self.hold)
            out, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return sanitize_active_html(out) if out else ""

    def flush(self) -> str:
        out, self.buffer = redact_secrets(self.buffer, self.settings), ""
        return sanitize_active_html(out) if out else ""
