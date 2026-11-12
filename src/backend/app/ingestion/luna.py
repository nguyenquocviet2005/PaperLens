"""Luna vision-language enrichment adapter (OpenAI-compatible)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.schemas import (
    FigureEnrichmentResult,
    FormulaEnrichmentResult,
    TableEnrichmentResult,
    VisualElement,
    VisualEnrichment,
    utc_now,
)
from app.infrastructure.observability import record_vlm_call
from app.infrastructure.storage import StorageBackend, paper_enrichment_key

logger = logging.getLogger(__name__)

ElementKind = Literal["figure", "table", "formula"]


class LunaDisabledError(Exception):
    """Raised when Luna calls are not permitted."""


class LunaError(Exception):
    """Provider or validation failure."""


SCHEMA_BY_KIND: dict[ElementKind, type[BaseModel]] = {
    "formula": FormulaEnrichmentResult,
    "figure": FigureEnrichmentResult,
    "table": TableEnrichmentResult,
}


LIST_FIELDS_BY_KIND: dict[ElementKind, set[str]] = {
    "formula": {"assumptions_or_conditions", "uncertainties"},
    "figure": {"components", "relationships", "evidence_from_caption", "uncertainties"},
    "table": {
        "columns",
        "main_results",
        "comparisons",
        "best_results",
        "limitations",
        "uncertainties",
    },
}


def _normalize_schema_payload(kind: ElementKind, payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce richer provider JSON into the stable public enrichment schema."""
    normalized = dict(payload)
    for field in LIST_FIELDS_BY_KIND[kind]:
        value = normalized.get(field)
        if value is None:
            normalized[field] = []
            continue
        values = value if isinstance(value, list) else [value]
        normalized[field] = [
            "; ".join(f"{key}: {item}" for key, item in entry.items())
            if isinstance(entry, dict)
            else str(entry)
            for entry in values
        ]
    return normalized


def _cache_key(
    *,
    asset_bytes: bytes,
    model: str,
    prompt_version: str,
    schema_version: str,
    kind: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(asset_bytes)
    digest.update(model.encode())
    digest.update(prompt_version.encode())
    digest.update(schema_version.encode())
    digest.update(kind.encode())
    return digest.hexdigest()


def _build_prompt(kind: ElementKind, element: VisualElement) -> str:
    shared = (
        "You are assisting with research-paper understanding. "
        "Use only what is visible in the image and the provided context. "
        "Do not invent unreadable content. "
        "If text or symbols are unclear, list them under uncertainties and lower confidence.\n\n"
        f"Element type: {kind}\n"
        f"Page: {element.page}\n"
        f"Section path: {' > '.join(element.section_path) if element.section_path else '(unknown)'}\n"
        f"Caption: {element.caption or '(none)'}\n"
        f"Nearby paragraphs: {element.surrounding_text}\n"
        f"Existing Docling transcription: {element.docling_text or '(empty)'}\n"
        f"Internal figure text: {element.internal_text}\n"
    )
    if kind == "formula":
        return (
            shared
            + "Return JSON with keys: latex, plain_reading, role_in_paper, explanation, "
            "symbols[{symbol,meaning,evidence}], assumptions_or_conditions, uncertainties, "
            "transcription_confidence."
        )
    if kind == "figure":
        return (
            shared
            + "Return JSON with keys: visual_type, description, main_message, components, "
            "relationships, evidence_from_caption, uncertainties, confidence."
        )
    return (
        shared
        + "Return JSON with keys: table_purpose, columns, main_results, comparisons, "
        "best_results, limitations, uncertainties, confidence."
    )


class LunaClient:
    """OpenAI-compatible Luna client with caching and retries."""

    def __init__(self, settings: Settings | None = None, storage: StorageBackend | None = None) -> None:
        self.settings = settings or get_settings()
        self.storage = storage

    @property
    def is_enabled(self) -> bool:
        return bool(self.settings.luna_enabled and self.settings.allow_external_api)

    def enrich_element(
        self,
        *,
        paper_id: str,
        element: VisualElement,
        image_bytes: bytes | None,
        force: bool = False,
    ) -> VisualEnrichment:
        kind: ElementKind = element.type  # type: ignore[assignment]
        if not self.is_enabled:
            raise LunaDisabledError(
                "Luna enrichment disabled. Set LUNA_ENABLED=true and ALLOW_EXTERNAL_API=true."
            )
        if not self.settings.luna_api_key:
            raise LunaError("LUNA_API_KEY is not configured")
        if not self.settings.luna_model:
            raise LunaError("LUNA_MODEL is not configured")
        if not image_bytes:
            raise LunaError(f"No image bytes available for {element.element_id}")

        cache_hash = _cache_key(
            asset_bytes=image_bytes,
            model=self.settings.luna_model,
            prompt_version=self.settings.luna_prompt_version,
            schema_version=self.settings.luna_schema_version,
            kind=kind,
        )
        cache_uri = paper_enrichment_key(paper_id, f"{kind}s", f"{element.element_id}_{cache_hash}.json")

        if self.storage and not force and self.storage.exists(cache_uri):
            payload = self.storage.read_json(cache_uri)
            return VisualEnrichment(
                provider=self.settings.luna_provider,
                model=self.settings.luna_model,
                prompt_version=self.settings.luna_prompt_version,
                schema_version=self.settings.luna_schema_version,
                cached=True,
                status="completed",
                result=payload.get("result"),
                usage=payload.get("usage"),
                enriched_at=utc_now(),
            )

        schema = SCHEMA_BY_KIND[kind]
        prompt = _build_prompt(kind, element)
        last_error: Exception | None = None
        usage: dict[str, Any] | None = None

        for attempt in range(self.settings.luna_max_retries + 1):
            started = time.perf_counter()
            usage = None
            try:
                raw, usage = self._call_provider(prompt=prompt, image_bytes=image_bytes)
                record_vlm_call(
                    model=self.settings.luna_model, seconds=time.perf_counter() - started,
                    usage=usage, ok=True,
                )
                parsed = schema.model_validate(_normalize_schema_payload(kind, raw))
                enrichment = VisualEnrichment(
                    provider=self.settings.luna_provider,
                    model=self.settings.luna_model,
                    prompt_version=self.settings.luna_prompt_version,
                    schema_version=self.settings.luna_schema_version,
                    cached=False,
                    status="completed",
                    result=parsed.model_dump(),
                    usage=usage,
                    enriched_at=utc_now(),
                )
                if self.storage:
                    self.storage.save_json(
                        cache_uri,
                        {
                            "element_id": element.element_id,
                            "kind": kind,
                            "cache_hash": cache_hash,
                            "result": enrichment.result,
                            "usage": usage,
                        },
                    )
                if self.settings.luna_request_delay_seconds > 0:
                    time.sleep(self.settings.luna_request_delay_seconds)
                return enrichment
            except (ValidationError, LunaError, json.JSONDecodeError, Exception) as exc:
                last_error = exc
                if not usage:
                    record_vlm_call(
                        model=self.settings.luna_model, seconds=time.perf_counter() - started,
                        usage=None, ok=False,
                    )
                logger.warning(
                    "Luna attempt %s failed for %s: %s",
                    attempt + 1,
                    element.element_id,
                    type(exc).__name__,
                )
                if attempt < self.settings.luna_max_retries:
                    time.sleep(min(2 ** attempt, 5))

        return VisualEnrichment(
            provider=self.settings.luna_provider,
            model=self.settings.luna_model,
            prompt_version=self.settings.luna_prompt_version,
            schema_version=self.settings.luna_schema_version,
            cached=False,
            status="failed",
            error=str(last_error) if last_error else "unknown error",
            usage=usage,
            enriched_at=utc_now(),
        )

    def _call_provider(self, *, prompt: str, image_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from openai import OpenAI

        # Retries are handled by enrich_element; the SDK's own 2 silent retries would
        # triple the requests spent on a failing element.
        client_kwargs: dict[str, Any] = {"api_key": self.settings.luna_api_key, "max_retries": 0}
        if self.settings.luna_base_url:
            client_kwargs["base_url"] = self.settings.luna_base_url
        client = OpenAI(**client_kwargs)

        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = client.chat.completions.create(
            model=self.settings.luna_model,
            timeout=self.settings.luna_timeout_seconds,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return data, usage


def mock_enrichment(kind: ElementKind) -> dict[str, Any]:
    """Deterministic mock payloads for tests."""
    if kind == "formula":
        return FormulaEnrichmentResult(
            latex="E = mc^2",
            plain_reading="E equals m c squared",
            role_in_paper="example equation",
            explanation="Mock enrichment",
            symbols=[{"symbol": "E", "meaning": "energy", "evidence": "mock"}],
            transcription_confidence=0.5,
            uncertainties=["mock"],
        ).model_dump()
    if kind == "figure":
        return FigureEnrichmentResult(
            visual_type="chart",
            description="Mock figure",
            main_message="mock",
            confidence=0.5,
        ).model_dump()
    return TableEnrichmentResult(
        table_purpose="mock",
        columns=["a", "b"],
        main_results=["mock"],
        confidence=0.5,
    ).model_dump()
