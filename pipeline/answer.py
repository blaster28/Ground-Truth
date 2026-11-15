"""Réponse aux questions, en deux étapes dans deux processus séparés.

  1. `retrieve` : recherche hybride + reranking pour toutes les questions -> fichier de passages ;
  2. `generate` : générateur officiel seul en mémoire, un appel par question.

Pourquoi deux processus : le générateur monte à ~13 Go de VRAM sur un T4 de 15 Go ; l'embedder et
le reranker ne tiennent pas à côté, et `del` + `empty_cache()` ne libèrent pas tout (voir README).

Abstention, trois garde-fous successifs :
  a. score du reranker du meilleur passage sous `tau`  -> INTROUVABLE sans appeler le générateur ;
  b. le générateur répond lui-même INTROUVABLE ;
  c. ancrage numérique : un nombre de la réponse absent des extraits fournis -> INTROUVABLE.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import parse

TAU = 0.05            # seuil du reranker, calibré sur la validation (voir pipeline/calibrate.py)
CONTEXT_TOKENS = 3000  # contexte volontairement sous les 4 096 : plus rapide, moins de « lost in the middle »
MAX_CHUNKS = 8

SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français (Insee, Cour des comptes), "
    "uniquement à partir des extraits fournis. Chaque extrait commence par sa référence [doc_id p.N], "
    "le titre du document et sa date de publication.\n"
    "Règles :\n"
    "1. Si les extraits ne contiennent pas l'information demandée, ou si la question repose sur une "
    "affirmation que les extraits ne confirment pas, réponds exactement : INTROUVABLE\n"
    "2. Sinon réponds en français, en une à trois phrases. Recopie les chiffres exactement comme dans "
    "l'extrait, avec leur unité (%, €, M€, Md€, points...). Ne calcule rien qui ne soit pas demandé.\n"
    "3. Si deux documents donnent des valeurs différentes pour la même donnée, la publication la plus "
    "récente fait foi.\n"
    "4. Si la question demande plusieurs éléments, donne-les tous.\n"
    "5. Termine par une dernière ligne listant les extraits utilisés, au format\n"
    "SOURCES: [doc_id p.N], [doc_id p.N]"
)


# ------------------------------------------------------------------ étape 1 : recherche
def retrieve(index_dir: str, questions_path: str, out_path: str, mode: str = "hybrid",
             rerank: bool = True) -> None:
    from groundtruth.io import load_questions

    from .index import Searcher, ranked_pages

    searcher = Searcher(index_dir, dense=mode != "bm25", rerank=rerank)
    titles = {p["doc_id"]: p.get("title", "") for p in searcher.pages}
    out, t0 = [], time.time()
    for q in load_questions(questions_path):
        hits = searcher.search(q["question"], mode=mode)
        for h in hits:
            h["title"] = titles.get(h["doc_id"], "")
        out.append({"qid": q["qid"], "question": q["question"], "hits": hits[:12],
                    "retrieved": ranked_pages(hits)[:10]})
    n = max(len(out), 1)
    print(f"[retrieve] {len(out)} questions, {(time.time() - t0) / n:.2f} s par question")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ étape 2 : génération
def _norm_num(s: str) -> str:
    return re.sub(r"[\s  .,]", "", s)


def numbers(text: str) -> set[str]:
    """Nombres d'un texte, sans séparateurs (« 308 000 » = « 308000 », « 18,8 » = « 188 »)."""
    found = re.findall(r"\d+(?:[\s  .,]\d+)*", text)
    return {_norm_num(n) for n in found}


def grounded(answer: str, context: str) -> bool:
    """Vrai si chaque nombre « significatif » de la réponse figure dans le contexte."""
    ctx = numbers(context)
    ctx_flat = _norm_num(context)
    for n in numbers(answer):
        if len(n) <= 1:  # chiffres isolés (« 2 fois », « 3 éléments ») : trop ambigus pour trancher
            continue
        if n not in ctx and n not in ctx_flat:
            return False
    return True


def build_context(gen, question: str, hits: list[dict], dates: dict[str, str]) -> tuple[list[dict], str, list[dict]]:
    """Ajoute les passages par ordre de pertinence tant que le budget de tokens tient."""
    blocks, used = [], []
    for h in hits[:MAX_CHUNKS]:
        date = dates.get(h["doc_id"], "")
        head = f"[{h['doc_id']} p.{h['page']}] {h.get('title', '')}" + (f" (publié le {date})" if date else "")
        candidate = blocks + [f"{head}\n{h['text']}"]
        msgs = _messages(question, candidate)
        if gen.count_tokens(msgs) > CONTEXT_TOKENS and blocks:
            break
        blocks, used = candidate, used + [h]
    return _messages(question, blocks), "\n\n".join(blocks), used


def _messages(question: str, blocks: list[str]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(blocks) + f"\n\nQuestion : {question}"}]


def parse_answer(raw: str, used: list[dict]) -> tuple[str, list[dict]]:
    """Sépare la réponse de la ligne SOURCES et ne garde que les citations de pages fournies."""
    allowed = {(h["doc_id"], h["page"]) for h in used}
    answer = re.split(r"\n?\**SOURCES\**\s*:", raw, flags=re.I)[0].strip()
    cits = []
    for d, p in re.findall(r"\[([\w\-]+),?\s*p\.?\s*(\d+)\]", raw):
        key = (d, int(p))
        if key in allowed and key not in cits:
            cits.append(key)
    answer = re.sub(r"\s*\[[\w\-]+,?\s*p\.?\s*\d+\]", "", answer).strip()
    if not cits and used:  # le modèle n'a rien cité de valide : la page la mieux classée
        cits = [(used[0]["doc_id"], used[0]["page"])]
    return answer, [{"doc_id": d, "page": p} for d, p in cits[:5]]


def answer_one(gen, item: dict, dates: dict[str, str], tau: float = TAU, check_numbers: bool = True) -> dict:
    hits = item["hits"]
    pred = {"qid": item["qid"], "answer": "INTROUVABLE", "citations": [], "retrieved": item["retrieved"]}
    gen.start_question(item["qid"])
    if not hits or hits[0]["score"] < tau:
        pred["abstention"] = "seuil"
        return pred
    msgs, context, used = build_context(gen, item["question"], hits, dates)
    raw = gen.chat(msgs, max_new_tokens=220)
    pred["raw"] = raw
    if "INTROUVABLE" in raw.upper():
        pred["abstention"] = "generateur"
        return pred
    answer, cits = parse_answer(raw, used)
    if not answer:
        pred["abstention"] = "vide"
        return pred
    if check_numbers and not grounded(answer, context + " " + item["question"]):
        pred["abstention"] = "ancrage"
        return pred
    pred["answer"], pred["citations"] = answer, cits
    return pred


def generate(retrieved_path: str, out_path: str, backend: str = "hf", tau: float = TAU,
             check_numbers: bool = True, run_name: str = "pipeline") -> None:
    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import write_predictions

    items = json.loads(Path(retrieved_path).read_text(encoding="utf-8"))
    dates = parse.load_dates()
    gen = OfficialGenerator(backend=backend)
    preds, t0 = [], time.time()
    for item in items:
        pred = answer_one(gen, item, dates, tau, check_numbers)
        preds.append(pred)
        print(f"{pred['qid']}: {pred['answer'][:90]}", flush=True)
    print(f"[generate] {(time.time() - t0) / max(len(preds), 1):.1f} s par question")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    debug = Path(out_path).with_suffix(".debug.json")
    debug.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    clean = [{k: p[k] for k in ("qid", "answer", "citations", "retrieved")} for p in preds]
    write_predictions(out_path, clean, run_name=run_name)
