"""Bounded model enrichment over deterministic, cleaned Docling text."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import Settings
from app.schemas import PaperDocument
from app.infrastructure.storage import StorageBackend


class SectionTextEnrichment(BaseModel):
    summary: str = ""
    key_claims: list[str] = Field(default_factory=list)
    terminology: list[str] = Field(default_factory=list)
    methods_or_equations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def _model(settings: Settings) -> ChatOpenAI:
    model = settings.text_enrichment_model or settings.luna_model or settings.llm_model
    api_key = settings.luna_api_key or settings.llm_api_key
    base_url = settings.luna_base_url or settings.llm_base_url
    if not model or not api_key:
        raise ValueError("Text enrichment requires a configured Luna or LLM model and API key")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "timeout": settings.luna_timeout_seconds,
        "max_retries": settings.luna_max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if "luna" in model.lower():
        kwargs["reasoning_effort"] = "none"
    from app.infrastructure.observability import llm_callbacks

    return ChatOpenAI(**kwargs, callbacks=llm_callbacks())


def enrich_cleaned_text(
    paper: PaperDocument,
    *,
    settings: Settings,
    storage: StorageBackend,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Enrich cleaned sections without replacing canonical Docling text."""
    if not (
        settings.text_enrichment_enabled
        and settings.allow_external_api
    ):
        return []

    grouped: dict[str, list[Any]] = defaultdict(list)
    for element in paper.text_elements:
        section = " › ".join(element.section_path) or "Document"
        grouped[section].append(element)

    structured = _model(settings).with_structured_output(SectionTextEnrichment)
    results: list[dict[str, Any]] = []
    total_chars = 0
    model_name = settings.text_enrichment_model or settings.luna_model or settings.llm_model
    for section_index, (section, elements) in enumerate(grouped.items()):
        if section_index >= settings.text_enrichment_max_sections:
            break
        source = "\n\n".join(element.text for element in elements)
        remaining = settings.text_enrichment_max_total_chars - total_chars
        if remaining <= 0:
            break
        source = source[: min(settings.text_enrichment_max_chars_per_section, remaining)]
        total_chars += len(source)
        digest = hashlib.sha256(
            (
                f"{model_name}|{settings.luna_prompt_version}|{section}|{source}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        cache_key = (
            f"enrichment/papers/{paper.paper_id}/text/"
            f"section_{section_index:03d}_{digest}.json"
        )
        if not force and storage.exists(cache_key):
            cached = storage.read_json(cache_key)
            cached["cached"] = True
            results.append(cached)
            continue

        enrichment = structured.invoke(
            (
                "Enrich this cleaned scientific-paper section. Summarize only supported "
                "content, preserve technical meaning, list key claims and terminology, "
                "and explicitly note uncertainty. Do not invent citations or facts.\n\n"
                f"Section: {section}\n\nCleaned source:\n{source}"
            )
        )
        payload = {
            "section": section,
            "source_element_ids": [element.element_id for element in elements],
            "result": (
                enrichment.model_dump(mode="json")
                if isinstance(enrichment, SectionTextEnrichment)
                else dict(enrichment)
            ),
            "provider": settings.luna_provider,
            "model": model_name,
            "prompt_version": settings.luna_prompt_version,
            "source": "cleaned_docling_text",
            "cached": False,
            "enriched_at": datetime.now(timezone.utc).isoformat(),
            "cache_uri": cache_key,
        }
        storage.save_json(cache_key, payload)
        results.append(payload)
    return results
