"""Éditions successives d'une même publication (palier 3 : « la plus récente fait foi »).

Beaucoup d'Insee Première reviennent chaque année sous le même intitulé (« Les salaires dans la
fonction publique territoriale en 2023 », puis « ... en 2024 »), avec des chiffres révisés. On
regroupe ces documents en séries pour pouvoir :
  - ramener l'édition la plus récente dans les candidats quand seule l'ancienne a été retrouvée ;
  - écarter les anciennes éditions quand la question demande explicitement la dernière donnée.
"""
from __future__ import annotations

import re
import unicodedata

RECENT = re.compile(r"derni[eè]re?s?\b|plus r[ée]cente?s?|r[ée]cemment|latest|most recent|"
                    r"newest|last (edition|release|survey)", re.I)
COMPARE = re.compile(r"r[ée]vis|pr[ée]c[ée]dent|successi|[ée]ditions?\b.*\b[ée]ditions?|entre les (deux )?(bilans|comptes|"
                     r"[ée]ditions|publications)|each (new )?release|previous|both|ann[ée]e apr[èe]s ann[ée]e|"
                     r"trois derni[èe]res|three most recent|year by year|each of|chacune? des|"
                     r"two years|deux ann[ée]es", re.I)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def series_head(title: str) -> str:
    """Intitulé sans le millésime : « les salaires dans la fonction publique territoriale »."""
    # bandeau commun à tous les Insee Première (lu par l'OCR sur les scans) : ce n'est pas un intitulé
    title = _norm(title).replace("insee mesurer pour comprendre", "").replace("insee premiere", "").strip()
    m = re.match(r"^(.{20,}?)\ben (?:19|20)\d\d\b", title)
    return m.group(1).strip() if m else ""


def build_series(pages: list[dict], dates: dict[str, str], chunks: list[dict], emb,
                 min_cos: float = 0.90, min_days: int = 180) -> dict[str, str]:
    """doc_id -> doc_id de l'édition qui le remplace (absent s'il n'y en a pas).

    Deux documents sont deux éditions d'une même série s'ils se ressemblent beaucoup (cosinus des
    vecteurs moyens de leurs chunks >= `min_cos`) et sont publiés à au moins `min_days` jours
    d'écart. Le titre seul ne suffit pas : il manque souvent (métadonnée absente, page scannée), et
    les trois « Salaires dans la fonction publique ... » d'une même année se ressemblent trop.
    S'il y a plusieurs candidats, on préfère celui qui porte le même intitulé, puis le plus proche."""
    import datetime

    import numpy as np

    rows: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        rows.setdefault(c["doc_id"], []).append(i)
    center = {}
    for d, idx in rows.items():
        v = np.asarray(emb[idx], dtype=np.float32).mean(axis=0)
        center[d] = v / (np.linalg.norm(v) or 1.0)
    day = {d: datetime.date.fromisoformat(dates[d][:10]) for d in center if len(dates.get(d, "")) >= 10}
    head = {p["doc_id"]: series_head(p.get("title", "")) for p in pages if p["page"] == 1}

    newer = {}
    for a in day:
        candidates = []
        for b in day:
            if b == a or a.split("-")[0] != b.split("-")[0] or (day[b] - day[a]).days < min_days:
                continue
            cos = float(center[a] @ center[b])
            if cos >= min_cos:
                same_head = bool(head.get(a)) and head.get(a) == head.get(b)
                candidates.append((same_head, cos, b))
        if candidates:
            newer[a] = max(candidates)[2]
    for a in list(newer):  # chaîne a -> b -> c : on va directement à la plus récente
        seen = {a}
        while newer[a] in newer and newer[newer[a]] not in seen:
            seen.add(newer[a])
            newer[a] = newer[newer[a]]
    return newer


def wants_latest(question: str) -> bool:
    """La question demande la donnée la plus récente, sans comparer deux éditions."""
    return bool(RECENT.search(question)) and not COMPARE.search(question)
