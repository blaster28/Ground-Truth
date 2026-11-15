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
    if doc is not None:
        meta = (doc.metadata or {}).get("title") or ""
        if len(meta) > 15 and not meta.lower().endswith((".pdf", ".indd", ".docx", ".doc")):
            return meta.strip()
        return ""
    lines = [l.strip(" #*") for l in first_text.splitlines() if len(l.strip(" #*")) > 3]
    return " ".join(lines[:3])[:200]


def read_pdf(pdf: Path) -> tuple[list[dict], list, str]:
    """Pages d'un PDF, et la liste (index dans pages, image) des pages sans couche texte."""
    import pymupdf

    from .ocr import to_pil

    pages, scans = [], []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            text = clean_text(page.get_text("text"))
            if len(text) < MIN_CHARS:
                scans.append((len(pages), to_pil(page.get_pixmap(dpi=OCR_DPI))))
            pages.append({"doc_id": pdf.stem, "page": i, "text": text, "source": "texte"})
        meta_title = doc_title(doc, "")
    return pages, scans, meta_title


def read_dir(pdf_dir: str, ocr=None, skip: set[str] | None = None, cache_path: str | None = None) -> list[dict]:
    """Lit tous les PDF du dossier (hors `skip`). Les pages sans texte passent à l'OCR par lots ;
    `cache_path` mémorise le résultat de l'OCR pour ne jamais refaire une page déjà lue."""
    cache = {}
    if cache_path and Path(cache_path).exists():
        from .ocr import looks_broken

        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        cache = {k: v for k, v in cache.items() if not looks_broken(v)}  # jamais de page corrompue
    pages, todo, bounds = [], [], []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        if skip and pdf.stem in skip:
            continue
        doc_pages, scans, meta_title = read_pdf(pdf)
        bounds.append((len(pages), len(pages) + len(doc_pages), meta_title))
        for local, image in scans:
            p = doc_pages[local]
            key = f"{p['doc_id']}/{p['page']}"
            if key in cache:
                p["text"], p["source"] = cache[key], "ocr"
            elif ocr is not None:
                todo.append((len(pages) + local, key, image))
        pages.extend(doc_pages)
    if todo:
        def save(start: int, texts: list[str]) -> None:  # cache écrit après chaque lot : rien n'est perdu
            for (idx, key, _), text in zip(todo[start:], texts):
                pages[idx]["text"], pages[idx]["source"] = clean_text(text), "ocr"
                cache[key] = pages[idx]["text"]
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        ocr.many([im for _, _, im in todo], [key for _, key, _ in todo], on_batch=save)
    for start, end, meta_title in bounds:  # le titre se lit après l'OCR (première page scannée)
        title = meta_title or doc_title(None, pages[start]["text"] if end > start else "")
        for p in pages[start:end]:
            p["title"] = title
    return pages


def load_dates(manifest: str = "corpus/manifest.json") -> dict[str, str]:
    """Date de publication par doc_id (utile aux questions temporelles). Vide si pas de manifeste."""
    path = Path(manifest)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data["documents"] if isinstance(data, dict) else data
    return {d["doc_id"]: d.get("date", "") for d in docs}
