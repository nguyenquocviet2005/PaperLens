"""External paper discovery (OpenAlex, arXiv) with normalization and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.infrastructure.storage import StorageBackend, get_storage

logger = logging.getLogger(__name__)


class DiscoveryPaper(BaseModel):
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: Literal["openalex", "arxiv", "semantic_scholar"] 
    source_url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    open_access: bool | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class DiscoveryResponse(BaseModel):
    query: str
    count: int
    results: list[DiscoveryPaper]
    cached: bool = False
    source: str
    warnings: list[str] = Field(default_factory=list)


class DiscoveryError(Exception):
    pass


def _cache_key(source: str, query: str, filters: dict[str, Any]) -> str:
    payload = json.dumps({"source": source, "query": query, "filters": filters}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _year_from_openalex(item: dict[str, Any]) -> int | None:
    year = item.get("publication_year")
    return int(year) if year else None


def search_openalex(
    query: str,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    per_page: int = 10,
    timeout: float = 20.0,
) -> list[DiscoveryPaper]:
    filters = []
    if year_min:
        filters.append(f"from_publication_date:{year_min}-01-01")
    if year_max:
        filters.append(f"to_publication_date:{year_max}-12-31")
    params: dict[str, Any] = {
        "search": query,
        "per_page": per_page,
        "mailto": "paperlens@local.dev",
    }
    if filters:
        params["filter"] = ",".join(filters)
    url = "https://api.openalex.org/works?" + urlencode(params)
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers={"User-Agent": "PaperLens/0.1"})
        resp.raise_for_status()
        data = resp.json()
    results: list[DiscoveryPaper] = []
    for item in data.get("results") or []:
        authorships = item.get("authorships") or []
        authors = []
        for a in authorships:
            author = (a.get("author") or {}).get("display_name")
            if author:
                authors.append(author)
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        oa = item.get("open_access") or {}
        doi = None
        ids = item.get("ids") or {}
        if ids.get("doi"):
            doi = str(ids["doi"]).replace("https://doi.org/", "")
        results.append(
            DiscoveryPaper(
                title=item.get("display_name") or "Untitled",
                abstract=None,  # OpenAlex abstract_inverted_index omitted for brevity
                authors=authors,
                year=_year_from_openalex(item),
                venue=source.get("display_name"),
                doi=doi,
                source="openalex",
                source_url=item.get("id"),
                pdf_url=oa.get("oa_url"),
                citation_count=item.get("cited_by_count"),
                open_access=bool(oa.get("is_oa")),
                raw={"id": item.get("id")},
            )
        )
    return results


def search_arxiv(
    query: str,
    *,
    max_results: int = 10,
    timeout: float = 20.0,
) -> list[DiscoveryPaper]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urlencode(params)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers={"User-Agent": "PaperLens/0.1"})
        resp.raise_for_status()
        text = resp.text
    root = ET.fromstring(text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[DiscoveryPaper] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        id_url = entry.findtext("atom:id", default="", namespaces=ns) or ""
        arxiv_id = id_url.rsplit("/", 1)[-1]
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
        results.append(
            DiscoveryPaper(
                title=title.replace("\n", " "),
                abstract=summary.replace("\n", " "),
                authors=[a for a in authors if a],
                year=year,
                venue="arXiv",
                arxiv_id=arxiv_id,
                source="arxiv",
                source_url=id_url,
                pdf_url=pdf_url,
                open_access=True,
            )
        )
    return results


def discover_papers(
    query: str,
    *,
    source: Literal["openalex", "arxiv", "auto"] = "auto",
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 10,
    settings: Settings | None = None,
    storage: StorageBackend | None = None,
    force_refresh: bool = False,
) -> DiscoveryResponse:
    cfg = settings or get_settings()
    store = storage or get_storage(cfg)
    selected = source
    if source == "auto":
        selected = "arxiv"  # default offline-friendly OA source
    filters = {"year_min": year_min, "year_max": year_max, "limit": limit}
    key = f"discovery/cache/{selected}/{_cache_key(selected, query, filters)}.json"
    if not force_refresh and store.exists(key):
        cached = store.read_json(key)
        return DiscoveryResponse.model_validate({**cached, "cached": True})

    warnings: list[str] = []
    try:
        if selected == "openalex":
            papers = search_openalex(query, year_min=year_min, year_max=year_max, per_page=limit)
        else:
            papers = search_arxiv(query, max_results=limit)
    except Exception as exc:
        logger.warning("Discovery source %s failed: %s", selected, exc)
        # Fallback
        if selected != "arxiv":
            try:
                papers = search_arxiv(query, max_results=limit)
                selected = "arxiv"
                warnings.append(f"primary_source_failed:{exc}")
            except Exception as exc2:
                raise DiscoveryError(str(exc2)) from exc2
        else:
            raise DiscoveryError(str(exc)) from exc

    # Deduplicate by DOI / arxiv / title
    seen: set[str] = set()
    deduped: list[DiscoveryPaper] = []
    for p in papers:
        identity = (p.doi or p.arxiv_id or p.title).lower().strip()
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(p)

    response = DiscoveryResponse(
        query=query,
        count=len(deduped),
        results=deduped,
        cached=False,
        source=selected,
        warnings=warnings,
    )
    try:
        store.save_json(key, response.model_dump(mode="json"))
    except Exception as exc:
        logger.warning("Failed to cache discovery results: %s", exc)
    # polite delay metadata only; caller controls pacing
    _ = time.time()
    return response


def find_library_duplicates(
    candidate: DiscoveryPaper,
    *,
    settings: Settings | None = None,
    user_id: str | None = None,
) -> list[str]:
    """Return paper IDs that match DOI/arXiv/title."""
    from sqlalchemy import select
    from app.db.models import Paper
    from app.db.session import session_scope

    cfg = settings or get_settings()
    matches: list[str] = []
    with session_scope(cfg) as session:
        stmt = select(Paper)
        if user_id is not None:
            stmt = stmt.where(Paper.user_id == user_id)
        rows = list(session.scalars(stmt))
        for row in rows:
            if candidate.doi and row.doi and candidate.doi.lower() == row.doi.lower():
                matches.append(row.id)
                continue
            if candidate.arxiv_id and row.arxiv_id and candidate.arxiv_id.lower() == row.arxiv_id.lower():
                matches.append(row.id)
                continue
            if candidate.title and row.title and candidate.title.lower().strip() == row.title.lower().strip():
                matches.append(row.id)
    return matches
