"""Repository layer for paper library operations (no SQL in routes)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Job, Paper, PaperElement, PaperSection, VisualElementRow
from app.schemas import ArtifactPaths, PaperDocument

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_pending_paper(
        self,
        *,
        paper_id: str,
        filename: str,
        storage_uri: str | None,
        status: str = "processing",
        parse_status: str | None = "queued",
        user_id: str = "local-user",
    ) -> Paper:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            paper = Paper(
                id=paper_id,
                user_id=user_id,
                filename=filename,
                storage_uri=storage_uri,
                status=status,
                parse_status=parse_status,
            )
            self.session.add(paper)
        else:
            if paper.user_id != user_id:
                raise ValueError("Paper belongs to another account")
            paper.filename = filename
            paper.storage_uri = storage_uri
            paper.status = status
            paper.parse_status = parse_status
            paper.updated_at = _utcnow()
        self.session.flush()
        return paper

    def create_job(self, *, paper_id: str, job_type: str, status: str = "running") -> Job:
        job = Job(paper_id=paper_id, job_type=job_type, status=status, progress=0.0)
        self.session.add(job)
        self.session.flush()
        return job

    def update_job(
        self,
        job: Job,
        *,
        status: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> Job:
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if result is not None:
            job.result = result
        job.updated_at = _utcnow()
        self.session.flush()
        return job

    def replace_document_graph(
        self,
        paper_doc: PaperDocument,
        artifacts: ArtifactPaths,
        *,
        user_id: str = "local-user",
    ) -> Paper:
        """Replace sections/elements/visuals for a paper from normalized document."""
        paper = self.session.get(Paper, paper_doc.paper_id)
        if paper is None:
            paper = Paper(id=paper_doc.paper_id, filename=paper_doc.filename, user_id=user_id)
            self.session.add(paper)
        elif paper.user_id != user_id:
            raise ValueError("Paper belongs to another account")

        paper.filename = paper_doc.filename
        paper.title = paper_doc.title
        paper.title_confidence = paper_doc.title_confidence
        paper.status = paper_doc.status
        paper.parse_status = str((paper_doc.parser or {}).get("status") or paper.parse_status)
        paper.page_count = paper_doc.page_count
        paper.storage_uri = paper_doc.source_pdf_uri
        paper.artifacts = artifacts.model_dump()
        paper.warnings = list(paper_doc.warnings)
        paper.error = None
        paper.updated_at = _utcnow()

        # Clear children for idempotent re-ingest
        paper.sections.clear()
        paper.elements.clear()
        paper.visual_elements.clear()
        self.session.flush()

        section_id_by_heading_order: dict[int, str] = {}
        for idx, section in enumerate(paper_doc.sections):
            row = PaperSection(
                paper_id=paper.id,
                heading=section.heading,
                section_path=[section.heading],
                level=section.level,
                page_start=section.page_start,
                page_end=section.page_end,
                order_index=idx,
            )
            self.session.add(row)
            self.session.flush()
            section_id_by_heading_order[idx] = row.id

        def _section_id_for_path(path: list[str]) -> str | None:
            if not path:
                return None
            heading = path[-1]
            for idx, section in enumerate(paper_doc.sections):
                if section.heading == heading:
                    return section_id_by_heading_order.get(idx)
            return None

        for el in paper_doc.text_elements:
            self.session.add(
                PaperElement(
                    paper_id=paper.id,
                    section_id=_section_id_for_path(el.section_path),
                    element_key=el.element_id,
                    type=el.type,
                    label=el.label,
                    page=el.page,
                    text=el.text,
                    bbox=el.bbox.model_dump() if el.bbox else None,
                    source_ref=el.source_ref,
                    order_index=el.order,
                    element_metadata={"section_path": el.section_path},
                )
            )

        for visual in [*paper_doc.tables, *paper_doc.figures, *paper_doc.formulas]:
            enrichment_status = "pending"
            enrichment_json = None
            if visual.enrichment is not None:
                enrichment_status = visual.enrichment.status
                enrichment_json = visual.enrichment.model_dump(mode="json")
            elif not visual.needs_enrichment:
                enrichment_status = "not_required"

            self.session.add(
                VisualElementRow(
                    paper_id=paper.id,
                    element_key=visual.element_id,
                    type=visual.type,
                    page=visual.page,
                    caption=visual.caption,
                    image_uri=visual.image_uri,
                    structured_data_uri=visual.structured_data_uri,
                    enrichment_status=enrichment_status,
                    enrichment_json=enrichment_json,
                    needs_enrichment=visual.needs_enrichment,
                    docling_text=visual.docling_text,
                    bbox=visual.bbox.model_dump() if visual.bbox else None,
                    source_ref=visual.source_ref,
                    visual_metadata={
                        "section_path": visual.section_path,
                        "surrounding_text": visual.surrounding_text,
                        "internal_text": visual.internal_text,
                    },
                )
            )

        self.session.flush()
        return paper

    def mark_failed(
        self,
        paper_id: str,
        *,
        filename: str,
        storage_uri: str | None,
        status: str,
        parse_status: str,
        error: str | None,
        artifacts: ArtifactPaths,
        warnings: list[str],
        page_count: int = 0,
        user_id: str = "local-user",
    ) -> Paper:
        paper = self.upsert_pending_paper(
            paper_id=paper_id,
            filename=filename,
            storage_uri=storage_uri,
            status=status,
            parse_status=parse_status,
            user_id=user_id,
        )
        paper.error = error
        paper.artifacts = artifacts.model_dump()
        paper.warnings = warnings
        paper.page_count = page_count
        paper.updated_at = _utcnow()
        self.session.flush()
        return paper

    def get_paper(self, paper_id: str, *, user_id: str | None = None) -> Paper | None:
        stmt: Select[tuple[Paper]] = (
            select(Paper)
            .where(Paper.id == paper_id)
            .options(
                selectinload(Paper.sections),
                selectinload(Paper.visual_elements),
                selectinload(Paper.jobs),
            )
        )
        if user_id is not None:
            stmt = stmt.where(Paper.user_id == user_id)
        return self.session.scalar(stmt)

    def list_papers(
        self,
        *,
        q: str | None = None,
        status: str | None = None,
        author: str | None = None,
        year: int | None = None,
        limit: int = 50,
        offset: int = 0,
        user_id: str | None = None,
    ) -> list[Paper]:
        stmt = select(Paper).order_by(Paper.created_at.desc())
        if user_id is not None:
            stmt = stmt.where(Paper.user_id == user_id)
        if status:
            stmt = stmt.where(Paper.status == status)
        if year is not None:
            stmt = stmt.where(Paper.publication_year == year)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(or_(Paper.title.ilike(like), Paper.filename.ilike(like)))

        # Fetch a bounded window then apply portable author filter in Python
        # (authors are JSON; SQL JSON search differs across SQLite/Postgres).
        fetch_limit = max(limit + offset, limit) if not author else max(200, limit + offset)
        papers = list(self.session.scalars(stmt.limit(fetch_limit)))
        if author:
            author_l = author.lower()
            papers = [
                p
                for p in papers
                if (p.authors and any(author_l in str(a).lower() for a in p.authors))
                or (p.title and author_l in p.title.lower())
            ]
        return papers[offset : offset + limit]
    def delete_paper(self, paper_id: str, *, user_id: str | None = None) -> bool:
        paper = self.get_paper(paper_id, user_id=user_id)
        if paper is None:
            return False
        self.session.delete(paper)
        self.session.flush()
        return True

    def count_related(self, paper_id: str) -> dict[str, int]:
        paper = self.get_paper(paper_id)
        if paper is None:
            return {}
        return {
            "sections": len(paper.sections),
            "visual_elements": len(paper.visual_elements),
            "jobs": len(paper.jobs),
        }
