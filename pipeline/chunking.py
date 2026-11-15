"""Stratégies de chunking. Un chunk ne dépasse jamais une page : la page est l'unité citée.

  - "page"       : une page = un chunk (la baseline) ;
  - "fenetre"    : fenêtres de mots de taille fixe avec recouvrement (le naïf) ;
  - "structure"  : découpe sur les paragraphes, regroupés jusqu'à une taille cible ;
  - contextuel   : option qui préfixe chaque chunk par le titre et la date de son document, pour
                   qu'un passage isolé (« le taux atteint 18,8 % ») reste retrouvable.
"""
from __future__ import annotations

import re


def _windows(words: list[str], size: int, overlap: int) -> list[str]:
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_fenetre(text: str, size: int = 220, overlap: int = 50) -> list[str]:
    return _windows(text.split(), size, overlap)


def chunk_structure(text: str, target: int = 220, overlap: int = 50) -> list[str]:
    """Regroupe les paragraphes jusqu'à ~`target` mots ; un paragraphe trop long est fenêtré."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:  # PyMuPDF ne sépare pas toujours les paragraphes : repli sur les lignes
        paras = [l for l in text.splitlines() if l.strip()]
    chunks, cur, n = [], [], 0
    for para in paras:
        w = len(para.split())
        if w > target * 1.5:
            if cur:
                chunks.append("\n".join(cur))
                cur, n = [], 0
            chunks.extend(_windows(para.split(), target, overlap))
            continue
        if n + w > target and cur:
            chunks.append("\n".join(cur))
            tail = cur[-1]  # recouvrement : on reprend le dernier paragraphe s'il est court
            cur, n = ([tail], len(tail.split())) if len(tail.split()) <= overlap else ([], 0)
        cur.append(para)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def context_prefix(page: dict, dates: dict[str, str]) -> str:
    date = dates.get(page["doc_id"], "")
    head = f"Document : {page.get('title') or page['doc_id']}"
    if date:
        head += f" (publié le {date})"
    return head


def make_chunks(pages: list[dict], strategy: str = "structure", contextual: bool = True,
                dates: dict[str, str] | None = None) -> list[dict]:
    """Renvoie des chunks {doc_id, page, text, embed_text}. `text` va au générateur, `embed_text`
    (éventuellement préfixé du contexte) va aux index."""
    dates = dates or {}
    out = []
    for p in pages:
        if strategy == "page":
            parts = [p["text"]] if p["text"].strip() else []
        elif strategy == "fenetre":
            parts = chunk_fenetre(p["text"])
        else:
            parts = chunk_structure(p["text"])
        prefix = context_prefix(p, dates) if contextual else ""
        for part in parts:
            if len(part) < 20:
                continue
            out.append({"doc_id": p["doc_id"], "page": p["page"], "text": part,
                        "embed_text": f"{prefix}\n{part}" if prefix else part})
    return out
