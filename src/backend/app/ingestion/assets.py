"""Extract table, figure, and formula assets from a Docling document."""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.schemas import BoundingBox, VisualElement
from app.infrastructure.storage import StorageBackend, paper_asset_key
from app.ingestion.surrounding import build_document_index, select_surrounding_text

logger = logging.getLogger(__name__)


class AssetExtractionError(Exception):
    """Non-fatal asset extraction failure for a single element."""


def _label_value(item: Any) -> str | None:
    label = getattr(item, "label", None)
    if label is None:
        return None
    return str(getattr(label, "value", label))


def _text_of(item: Any) -> str:
    return str(getattr(item, "text", "") or "").strip()


def _bbox_and_page(item: Any) -> tuple[BoundingBox | None, int | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    first = prov[0]
    page = getattr(first, "page_no", None)
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return None, int(page) if page is not None else None
    origin = getattr(bbox, "coord_origin", None)
    origin_value = getattr(origin, "value", None) or (str(origin) if origin else "BOTTOMLEFT")
    return (
        BoundingBox(
            l=float(bbox.l),
            t=float(bbox.t),
            r=float(bbox.r),
            b=float(bbox.b),
            coord_origin=str(origin_value),
        ),
        int(page) if page is not None else None,
    )


def bbox_to_pil_crop(
    bbox: BoundingBox,
    image_size: tuple[int, int],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    padding: int = 8,
) -> tuple[int, int, int, int]:
    """Convert Docling bbox to Pillow crop box (left, upper, right, lower).

    Docling often uses BOTTOMLEFT origin; Pillow uses top-left.
    """
    width, height = image_size
    page_h = float(page_height) if page_height else float(height)
    page_w = float(page_width) if page_width else float(width)

    left = float(bbox.l)
    right = float(bbox.r)
    origin = (bbox.coord_origin or "BOTTOMLEFT").upper()
    if "BOTTOMLEFT" in origin:
        top_from_bottom = max(float(bbox.t), float(bbox.b))
        bottom_from_bottom = min(float(bbox.t), float(bbox.b))
        upper = page_h - top_from_bottom
        lower = page_h - bottom_from_bottom
    else:
        upper = min(float(bbox.t), float(bbox.b))
        lower = max(float(bbox.t), float(bbox.b))

    # Scale PDF-space coordinates onto the rendered page image when sizes differ.
    if page_w > 0 and abs(page_w - width) > 1:
        sx = width / page_w
        left *= sx
        right *= sx
    if page_h > 0 and abs(page_h - height) > 1:
        sy = height / page_h
        upper *= sy
        lower *= sy

    left_i = max(0, int(left) - padding)
    upper_i = max(0, int(upper) - padding)
    right_i = min(width, int(right) + padding)
    lower_i = min(height, int(lower) + padding)
    if right_i <= left_i:
        right_i = min(width, left_i + 1)
    if lower_i <= upper_i:
        lower_i = min(height, upper_i + 1)
    return left_i, upper_i, right_i, lower_i

def _get_page_image(document: Any, page_no: int | None) -> Image.Image | None:
    if page_no is None:
        return None
    pages = getattr(document, "pages", {}) or {}
    page = pages.get(page_no) or pages.get(str(page_no))
    if page is None:
        return None
    image = getattr(page, "image", None)
    if image is None:
        return None
    pil = getattr(image, "pil_image", None)
    if pil is not None:
        return pil
    # Some versions expose bytes / URI
    if hasattr(image, "pil_image"):
        return image.pil_image
    return None


def _page_size(document: Any, page_no: int | None) -> tuple[float | None, float | None]:
    if page_no is None:
        return None, None
    pages = getattr(document, "pages", {}) or {}
    page = pages.get(page_no) or pages.get(str(page_no))
    if page is None:
        return None, None
    size = getattr(page, "size", None)
    if size is None:
        return None, None
    return getattr(size, "width", None), getattr(size, "height", None)


def _embedder(cfg: Settings):
    """Embedding function for ranking context paragraphs; disabled after the first failure."""
    if not cfg.surrounding_text_use_embeddings:
        return None
    try:
        from app.rag.embeddings import get_embeddings

        model, _name = get_embeddings(cfg)
    except Exception as exc:
        logger.warning("Surrounding-text embeddings unavailable: %s", type(exc).__name__)
        return None
    state = {"ok": True}

    def embed(texts):
        if not state["ok"]:
            raise RuntimeError("embeddings disabled after an earlier failure")
        try:
            return model.embed_documents(list(texts))
        except Exception:
            state["ok"] = False
            raise

    return embed


def _table_markdown(item: Any, document: Any) -> str | None:
    try:
        markdown = item.export_to_markdown(doc=document)
    except Exception:
        return None
    return markdown.strip() or None


def _caption_for(item: Any, document: Any | None = None) -> str | None:
    parts: list[str] = []
    cap_attr = getattr(item, "caption_text", None)
    if callable(cap_attr):
        try:
            # Newer Docling APIs may require the document argument.
            try:
                text = cap_attr(document) if document is not None else cap_attr()
            except TypeError:
                text = cap_attr()
            if text:
                parts.append(str(text).strip())
        except Exception:
            pass
    captions = getattr(item, "captions", None) or []
    for cap in captions:
        text = _text_of(cap) if not isinstance(cap, str) else cap
        if text:
            parts.append(text.strip())
    merged = " ".join(p for p in parts if p).strip()
    return merged or None


def _formula_needs_enrichment(text: str | None) -> bool:
    if text is None:
        return True
    cleaned = text.strip()
    if not cleaned:
        return True
    # Obviously invalid placeholders
    if cleaned.lower() in {"formula", "equation", "n/a", "none", "?"}:
        return True
    return False


def _save_pil(storage: StorageBackend, key: str, image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return storage.save_bytes(key, buf.getvalue(), content_type="image/png")


def _item_image(item: Any) -> Image.Image | None:
    image = getattr(item, "image", None)
    if image is None:
        return None
    pil = getattr(image, "pil_image", None)
    if pil is not None:
        return pil
    return None


def extract_assets(
    document: Any,
    *,
    paper_id: str,
    storage: StorageBackend,
    settings: Settings | None = None,
) -> dict[str, list[VisualElement]]:
    """Export tables, figures, and formulas; return visual element manifests."""
    from docling_core.types.doc import FormulaItem, PictureItem, TableItem

    cfg = settings or get_settings()
    tables: list[VisualElement] = []
    figures: list[VisualElement] = []
    formulas: list[VisualElement] = []
    warnings: list[str] = []

    order = 0
    table_idx = figure_idx = formula_idx = 0
    context_index = build_document_index(document)
    embed = _embedder(cfg)

    def context(kind: str, item: Any, caption: str | None, page: int | None) -> list[str]:
        return select_surrounding_text(
            context_index,
            kind=kind,
            caption=caption,
            source_ref=getattr(item, "self_ref", None),
            page=page,
            budget_tokens=cfg.surrounding_text_max_tokens,
            embed=embed,
        )

    # Collect picture-internal text refs
    picture_children: dict[str, list[str]] = {}
    ref_to_type: dict[str, str] = {}
    for item, _level in document.iterate_items():
        ref = getattr(item, "self_ref", None)
        if ref:
            ref_to_type[ref] = type(item).__name__

    for item, _level in document.iterate_items():
        order += 1
        parent = getattr(item, "parent", None)
        parent_ref = getattr(parent, "cref", None) if parent else None
        if parent_ref and ref_to_type.get(parent_ref) == "PictureItem":
            text = _text_of(item)
            if text:
                picture_children.setdefault(parent_ref, []).append(text)

        if isinstance(item, TableItem):
            table_idx += 1
            element_id = f"table_{table_idx:03d}"
            bbox, page = _bbox_and_page(item)
            caption = _caption_for(item, document)
            image_uri = None
            structured_uri = None

            # Structured exports
            try:
                df = item.export_to_dataframe(doc=document)
                csv_key = paper_asset_key(paper_id, "tables", f"{element_id}.csv")
                storage.save_text(csv_key, df.to_csv(index=False))
                structured_uri = csv_key
            except Exception as exc:
                warnings.append(f"{element_id} CSV export failed: {exc}")

            try:
                html = item.export_to_html(doc=document)
                html_key = paper_asset_key(paper_id, "tables", f"{element_id}.html")
                storage.save_text(html_key, html)
            except Exception as exc:
                warnings.append(f"{element_id} HTML export failed: {exc}")

            pil = _item_image(item)
            if pil is None and bbox is not None:
                page_img = _get_page_image(document, page)
                if page_img is not None:
                    pw, ph = _page_size(document, page)
                    try:
                        crop_box = bbox_to_pil_crop(
                            bbox,
                            page_img.size,
                            page_width=float(pw) if pw else None,
                            page_height=float(ph) if ph else None,
                            padding=cfg.asset_crop_padding_px,
                        )
                        pil = page_img.crop(crop_box)
                    except Exception as exc:
                        warnings.append(f"{element_id} crop failed: {exc}")
            if pil is not None:
                try:
                    image_uri = _save_pil(
                        storage,
                        paper_asset_key(paper_id, "tables", f"{element_id}.png"),
                        pil,
                    )
                except Exception as exc:
                    warnings.append(f"{element_id} PNG save failed: {exc}")

            tables.append(
                VisualElement(
                    element_id=element_id,
                    type="table",
                    page=page,
                    caption=caption,
                    surrounding_text=context("table", item, caption, page),
                    bbox=bbox,
                    image_uri=image_uri,
                    structured_data_uri=structured_uri,
                    docling_text=_table_markdown(item, document),
                    needs_enrichment=False,
                    source_ref=getattr(item, "self_ref", None),
                    order=order,
                )
            )

        elif isinstance(item, PictureItem):
            figure_idx += 1
            element_id = f"figure_{figure_idx:03d}"
            bbox, page = _bbox_and_page(item)
            caption = _caption_for(item, document)
            image_uri = None
            pil = _item_image(item)
            if pil is None and bbox is not None:
                page_img = _get_page_image(document, page)
                if page_img is not None:
                    pw, ph = _page_size(document, page)
                    try:
                        crop_box = bbox_to_pil_crop(
                            bbox,
                            page_img.size,
                            page_width=float(pw) if pw else None,
                            page_height=float(ph) if ph else None,
                            padding=cfg.asset_crop_padding_px,
                        )
                        pil = page_img.crop(crop_box)
                    except Exception as exc:
                        warnings.append(f"{element_id} crop failed: {exc}")
            if pil is not None:
                try:
                    image_uri = _save_pil(
                        storage,
                        paper_asset_key(paper_id, "figures", f"{element_id}.png"),
                        pil,
                    )
                except Exception as exc:
                    warnings.append(f"{element_id} PNG save failed: {exc}")

            self_ref = getattr(item, "self_ref", None)
            figures.append(
                VisualElement(
                    element_id=element_id,
                    type="figure",
                    page=page,
                    caption=caption,
                    surrounding_text=context("figure", item, caption, page),
                    bbox=bbox,
                    image_uri=image_uri,
                    docling_text=None,
                    needs_enrichment=False,
                    source_ref=self_ref,
                    internal_text=list(picture_children.get(self_ref or "", [])),
                    order=order,
                )
            )

        elif isinstance(item, FormulaItem):
            formula_idx += 1
            element_id = f"formula_{formula_idx:03d}"
            bbox, page = _bbox_and_page(item)
            text = _text_of(item)
            image_uri = None
            page_img = _get_page_image(document, page)
            if page_img is not None and bbox is not None:
                pw, ph = _page_size(document, page)
                try:
                    crop_box = bbox_to_pil_crop(
                        bbox,
                        page_img.size,
                        page_width=float(pw) if pw else None,
                        page_height=float(ph) if ph else None,
                        padding=cfg.asset_crop_padding_px,
                    )
                    crop = page_img.crop(crop_box)
                    image_uri = _save_pil(
                        storage,
                        paper_asset_key(paper_id, "formulas", f"{element_id}.png"),
                        crop,
                    )
                except Exception as exc:
                    warnings.append(f"{element_id} crop failed: {exc}")

            formulas.append(
                VisualElement(
                    element_id=element_id,
                    type="formula",
                    page=page,
                    caption=None,
                    surrounding_text=context("formula", item, None, page),
                    bbox=bbox,
                    image_uri=image_uri,
                    docling_text=text or None,
                    needs_enrichment=_formula_needs_enrichment(text),
                    source_ref=getattr(item, "self_ref", None),
                    order=order,
                )
            )

    # Optionally persist a few page previews (not embedded in JSON)
    try:
        pages = getattr(document, "pages", {}) or {}
        for page_no, page in list(pages.items())[:3]:
            pil = _get_page_image(document, int(page_no) if not isinstance(page_no, int) else page_no)
            if pil is None:
                continue
            key = paper_asset_key(paper_id, "pages", f"page_{int(page_no):03d}.png")
            _save_pil(storage, key, pil)
    except Exception as exc:
        warnings.append(f"page image export failed: {exc}")

    if warnings:
        logger.warning("Asset extraction warnings (%s): %s", len(warnings), warnings[:5])

    return {
        "tables": tables,
        "figures": figures,
        "formulas": formulas,
        "warnings": warnings,  # type: ignore[dict-item]
    }
