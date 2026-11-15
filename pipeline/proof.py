"""Réponse « preuve d'abord » : le générateur doit recopier la phrase qui justifie sa réponse.

Le générateur (3B) se trompe surtout en prenant un chiffre voisin dans la page. On lui demande
donc trois lignes :

    PREUVE: "phrase(s) recopiée(s) mot pour mot d'un extrait"
    REPONSE: une ou deux phrases
    SOURCES: [doc_id p.N]

puis le pipeline vérifie, sans LLM :
  1. la preuve existe vraiment dans les extraits fournis (sinon elle est inventée) ;
  2. chaque nombre de la réponse figure dans la preuve (sinon il vient d'ailleurs) ;
  3. la page citée est celle où la preuve a été retrouvée (citation fiable par construction).
Un échec à l'une de ces vérifications donne INTROUVABLE.
"""
from __future__ import annotations

import re
import unicodedata

SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français (Insee, Cour des comptes), "
    "uniquement à partir des extraits fournis. Chaque extrait commence par sa référence [doc_id p.N], "
    "le titre du document et sa date de publication.\n"
    "Si les extraits ne contiennent pas exactement l'information demandée (même année, même lieu, "
    "même catégorie), ou si la question affirme quelque chose que les extraits ne confirment pas, "
    "réponds uniquement : INTROUVABLE\n"
    "Sinon réponds en trois lignes, dans ce format exact :\n"
    "PREUVE: \"la ou les phrases des extraits qui contiennent la réponse, recopiées mot pour mot\"\n"
    "REPONSE: une ou deux phrases simples en français, sans mise en forme, qui reprennent les chiffres "
    "de la preuve tels quels avec leur unité et ne donnent que ce qui est demandé\n"
    "SOURCES: [doc_id p.N]\n"
    "Règles : ne fais aucun calcul qui n'est pas demandé. Si deux documents donnent des valeurs "
    "différentes pour la même donnée, utilise la publication la plus récente. Si la question demande "
    "plusieurs éléments, la preuve et la réponse doivent tous les contenir."
)

VERIFY_SYSTEM = (
    "Tu es un vérificateur strict. On te donne une question, un passage d'un rapport et une réponse "
    "proposée. Réponds OUI seulement si le passage répond exactement à la question posée (même "
    "année, même lieu, même catégorie, même indicateur) et si la réponse proposée dit la même chose "
    "que le passage. Dans tous les autres cas, ou en cas de doute, réponds NON. "
    "Réponds par un seul mot : OUI ou NON."
)


def norm(text: str) -> str:
    """Minuscules, sans accents, sans ponctuation ni mise en forme : pour comparer deux textes."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def split_fields(raw: str) -> tuple[str, str]:
    """Extrait (preuve, réponse) de la sortie du générateur. Chaînes vides si le format manque."""
    proof = re.search(r"PREUVE\s*:\s*(.*?)(?=\n\s*\**REPONSE\s*:|\n\s*\**RÉPONSE\s*:|\Z)", raw, flags=re.S | re.I)
    answer = re.search(r"R[EÉ]PONSE\s*:\s*(.*?)(?=\n\s*\**SOURCES\s*:|\Z)", raw, flags=re.S | re.I)
    clean = lambda m: m.group(1).strip().strip('"«»“” *').strip() if m else ""
    return clean(proof), clean(answer)


def proof_coverage(proof: str, block: str) -> float:
    """Part des mots de la preuve retrouvés dans `block` sous forme de suites de 4 mots consécutifs.

    Tolère une coupure ou un mot changé, mais pas une phrase inventée."""
    p, b = norm(proof).split(), norm(block)
    if len(p) < 4:
        return 1.0 if " ".join(p) in b else 0.0
    covered = [False] * len(p)
    for i in range(len(p) - 3):
        if " ".join(p[i:i + 4]) in b:
            covered[i:i + 4] = [True] * 4
    return sum(covered) / len(p)


def locate(proof: str, used: list[dict]) -> tuple[dict | None, float]:
    """Bloc du contexte qui contient le mieux la preuve, et sa couverture."""
    best, best_cov = None, 0.0
    for h in used:
        cov = proof_coverage(proof, h["text"])
        if cov > best_cov:
            best, best_cov = h, cov
    return best, best_cov


def numbers(text: str) -> set[str]:
    found = re.findall(r"\d+(?:[\s  .,]\d+)*", text)
    return {re.sub(r"[\s  .,]", "", n) for n in found}


def numbers_in_proof(answer: str, proof: str, question: str) -> bool:
    """Chaque nombre de la réponse (hors chiffres isolés) doit venir de la preuve ou de la question."""
    allowed = numbers(proof) | numbers(question)
    flat = re.sub(r"[\s  .,]", "", proof)
    return all(len(n) <= 1 or n in allowed or n in flat for n in numbers(answer))


def _significant(text: str) -> set[str]:
    return {n for n in numbers(text) if len(n) > 1}


def agree(a: str, b: str) -> bool:
    """Deux réponses concordent si les nombres de l'une sont inclus dans ceux de l'autre ;
    sans aucun nombre, si elles partagent plus de 60 % des mots de la plus courte."""
    na, nb = _significant(a), _significant(b)
    if na or nb:
        return bool(na) and bool(nb) and (na <= nb or nb <= na)
    wa, wb = set(norm(a).split()), set(norm(b).split())
    return len(wa & wb) / max(1, min(len(wa), len(wb))) > 0.6
