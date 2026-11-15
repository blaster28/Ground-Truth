"""Lecture des PDF : une entrée par page physique (index à partir de 1, le format cité).

Deux chemins :
  - page avec couche texte  -> PyMuPDF (rapide, exact) ;
  - page sans couche texte  -> OCR (pages « numérisées » du corpus), fait à l'indexation, quand le
    générateur n'est pas chargé.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MIN_CHARS = 50      # en dessous, on considère que la page n'a pas de couche texte
OCR_DPI = 132      # A4 -> ~1540 px sur le grand côté, la taille conseillée pour LightOnOCR-2


def clean_text(text: str) -> str:
    text = text.replace("­", "").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # césures de fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def doc_title(doc, first_text: str) -> str:
    """Titre du document : métadonnée PDF si elle est parlante, sinon les premières lignes."""
    meta = (doc.metadata or {}).get("title") or ""
    if len(meta) > 15 and not meta.lower().endswith((".pdf", ".indd", ".docx", ".doc")):
        return meta.strip()
    lines = [l.strip() for l in first_text.splitlines() if len(l.strip()) > 3]
    return " ".join(lines[:3])[:200]


def read_pdf(pdf: Path, ocr=None) -> list[dict]:
    import pymupdf

    pages = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            text = clean_text(page.get_text("text"))
            source = "texte"
            if len(text) < MIN_CHARS and ocr is not None:
                pix = page.get_pixmap(dpi=OCR_DPI)
                text = clean_text(ocr(pix))
                source = "ocr"
            pages.append({"doc_id": pdf.stem, "page": i, "text": text, "source": source})
        title = doc_title(doc, pages[0]["text"] if pages else "")
    for p in pages:
        p["title"] = title
    return pages


def read_dir(pdf_dir: str, ocr=None, skip: set[str] | None = None) -> list[dict]:
    pages = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        if skip and pdf.stem in skip:
            continue
        pages.extend(read_pdf(pdf, ocr))
    return pages


def load_dates(manifest: str = "corpus/manifest.json") -> dict[str, str]:
    """Date de publication par doc_id (utile aux questions temporelles). Vide si pas de manifeste."""
    path = Path(manifest)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data["documents"] if isinstance(data, dict) else data
    return {d["doc_id"]: d.get("date", "") for d in docs}
