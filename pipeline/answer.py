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

from . import parse, proof

TAU = 0.05            # seuil du reranker, calibré sur la validation (voir pipeline/calibrate.py)
CONTEXT_TOKENS = 3400  # sous les 4 096 : marge pour la question et la consigne
MAX_CHUNKS = 8
FULL_PAGES = 3         # pages dont le texte complet est transmis à l'étape de génération
PAGE_CHARS = 4500      # une page entière est tronquée au-delà (≈ 1 300 tokens)

SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français (Insee, Cour des comptes), "
    "uniquement à partir des extraits fournis. Chaque extrait commence par sa référence [doc_id p.N], "
    "le titre du document et sa date de publication.\n"
    "Règles :\n"
    "1. Si les extraits ne contiennent pas l'information demandée, ou si la question repose sur une "
    "affirmation que les extraits ne confirment pas, réponds exactement : INTROUVABLE\n"
    "2. Sinon réponds en français, en une ou deux phrases simples, sans mise en forme (pas de gras, "
    "pas de liste). Donne uniquement ce qui est demandé, sans détail ni commentaire supplémentaire.\n"
    "3. Recopie les chiffres exactement comme dans l'extrait, avec leur unité (%, €, M€, Md€, "
    "points...). Vérifie que le chiffre correspond bien à l'année, au lieu et à la catégorie de la "
    "question. Ne fais aucun calcul qui n'est pas demandé.\n"
    "4. Si deux documents donnent des valeurs différentes pour la même donnée, la publication la plus "
    "récente fait foi.\n"
    "5. Si la question demande plusieurs éléments, donne-les tous, dans une phrase.\n"
    "6. Termine par une dernière ligne listant les extraits utilisés, au format\n"
    "SOURCES: [doc_id p.N], [doc_id p.N]"
)


# ------------------------------------------------------------------ étape 1 : recherche
def retrieve(index_dir: str, questions_path: str, out_path: str, mode: str = "hybrid",
             rerank: bool = True, use_editions: bool = True) -> None:
    from groundtruth.io import load_questions

    from .index import Searcher, ranked_pages

    searcher = Searcher(index_dir, dense=mode != "bm25", rerank=rerank, use_editions=use_editions)
    titles = {p["doc_id"]: p.get("title", "") for p in searcher.pages}
    out, t0 = [], time.time()
    for q in load_questions(questions_path):
        hits = searcher.search(q["question"], mode=mode)
        for h in hits:
            h["title"] = titles.get(h["doc_id"], "")
        pages = ranked_pages(hits)
        full = [{**p, "title": titles.get(p["doc_id"], ""), "superseded_by": searcher.newer.get(p["doc_id"], ""),
                 "text": searcher.page_text[(p["doc_id"], p["page"])]["text"]} for p in pages[:FULL_PAGES]]
        out.append({"qid": q["qid"], "question": q["question"], "hits": hits[:12],
                    "full_pages": full, "retrieved": pages[:10]})
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


def _head(h: dict, dates: dict[str, str]) -> str:
    date = dates.get(h["doc_id"], "")
    head = f"[{h['doc_id']} p.{h['page']}] {h.get('title', '')}" + (f" (publié le {date})" if date else "")
    if h.get("superseded_by"):  # le générateur doit savoir qu'une édition plus récente existe
        head += f" — ANCIENNE ÉDITION, remplacée par {h['superseded_by']}"
    return head


def build_context(gen, item: dict, dates: dict[str, str], mode: str = "mixte",
                  n_full: int = 2, system: str | None = None) -> tuple[list[dict], str, list[dict]]:
    """Construit le prompt en ajoutant des blocs par ordre de pertinence tant que le budget tient.

    mode "chunks" : les meilleurs passages seulement.
    mode "mixte"  : le texte complet des `n_full` meilleures pages (un tableau coupé par le chunking
                    perd ses en-têtes : la page entière les garde), puis les meilleurs passages des
                    autres pages.
    """
    question, hits = item["question"], item["hits"]
    candidates = []
    if mode == "mixte":
        full = item.get("full_pages", [])[:n_full]
        done = {(p["doc_id"], p["page"]) for p in full}
        candidates += [{**p, "text": p["text"][:PAGE_CHARS]} for p in full]
        candidates += [h for h in hits if (h["doc_id"], h["page"]) not in done]
    else:
        candidates = list(hits)
    blocks, used = [], []
    for h in candidates[:MAX_CHUNKS]:
        candidate = blocks + [f"{_head(h, dates)}\n{h['text']}"]
        if gen.count_tokens(_messages(question, candidate, system)) > CONTEXT_TOKENS:
            if blocks:
                continue  # ce bloc ne tient pas : on tente le suivant, plus court
            candidate = [candidate[0][:6000]]
        blocks, used = candidate, used + [h]
    return _messages(question, blocks, system), "\n\n".join(blocks), used


def _messages(question: str, blocks: list[str], system: str | None = None) -> list[dict]:
    return [{"role": "system", "content": system or SYSTEM},
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
    answer = answer.replace("**", "")
    if not cits and used:  # le modèle n'a rien cité de valide : la page la mieux classée
        cits = [(used[0]["doc_id"], used[0]["page"])]
    return answer, [{"doc_id": d, "page": p} for d, p in cits[:5]]


def answer_one(gen, item: dict, dates: dict[str, str], tau: float = TAU, check_numbers: bool = True,
               mode: str = "mixte", strategy: str = "consensus", verify: bool = False,
               min_coverage: float = 0.7, second_context: str = "mixte") -> dict:
    """strategy "direct" : réponse libre + ligne SOURCES.
    strategy "preuve" : le générateur recopie la phrase qui justifie sa réponse et le pipeline la
    vérifie (voir pipeline/proof.py).
    strategy "consensus" : « preuve », puis un second appel « direct » ; on ne répond que si les deux
    réponses concordent (2 appels par question au plus)."""
    hits = item["hits"]
    pred = {"qid": item["qid"], "answer": "INTROUVABLE", "citations": [], "retrieved": item["retrieved"]}
    gen.start_question(item["qid"])
    if not hits or hits[0]["score"] < tau:
        pred["abstention"] = "seuil"
        return pred
    system = SYSTEM if strategy == "direct" else proof.SYSTEM
    msgs, context, used = build_context(gen, item, dates, mode, system=system)
    raw = gen.chat(msgs, max_new_tokens=220 if strategy == "direct" else 300)
    pred["raw"] = raw
    if "INTROUVABLE" in raw.upper():
        pred["abstention"] = "generateur"
        return pred
    if strategy == "direct":
        answer, cits = parse_answer(raw, used)
        if not answer:
            pred["abstention"] = "vide"
        elif check_numbers and not grounded(answer, context + " " + item["question"]):
            pred["abstention"] = "ancrage"
        else:
            pred["answer"], pred["citations"] = answer, cits
        return pred

    quote, answer = proof.split_fields(raw)
    answer = answer.replace("**", "")
    if not quote or not answer:
        pred["abstention"] = "format"
        return pred
    block, coverage = proof.locate(quote, used)
    pred["coverage"] = round(coverage, 2)
    if block is None or coverage < min_coverage:
        pred["abstention"] = "preuve absente"
        return pred
    if check_numbers and not proof.numbers_in_proof(answer, quote, item["question"]):
        pred["abstention"] = "ancrage"
        return pred
    if verify:
        check = [{"role": "system", "content": proof.VERIFY_SYSTEM},
                 {"role": "user", "content": f"Question : {item['question']}\n\nPassage : {quote}\n\n"
                                             f"Réponse proposée : {answer}"}]
        verdict = gen.chat(check, max_new_tokens=5)
        pred["verdict"] = verdict
        if "OUI" not in verdict.upper():
            pred["abstention"] = "verification"
            return pred
    if strategy == "consensus":
        # second avis : même question, consigne « réponse directe ». Le générateur se trompe souvent,
        # mais rarement deux fois de la même façon : on ne répond que si les deux avis concordent.
        msgs2, _, _ = build_context(gen, item, dates, second_context, system=SYSTEM)
        raw2 = gen.chat(msgs2, max_new_tokens=220)
        pred["raw2"] = raw2
        second, _ = parse_answer(raw2, used)
        if "INTROUVABLE" in raw2.upper() or not proof.agree(answer, second):
            pred["abstention"] = "desaccord"
            return pred
    # citations : la page où la preuve a été retrouvée, puis celles que le générateur a citées
    _, cits = parse_answer(raw, used)
    first = {"doc_id": block["doc_id"], "page": block["page"]}
    pred["answer"] = answer
    pred["citations"] = ([first] + [c for c in cits if c != first])[:5]
    return pred


def generate(retrieved_path: str, out_path: str, backend: str = "hf", tau: float = TAU,
             check_numbers: bool = True, run_name: str = "pipeline", mode: str = "mixte",
             strategy: str = "consensus", verify: bool = False, second_context: str = "mixte") -> None:
    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import write_predictions

    items = json.loads(Path(retrieved_path).read_text(encoding="utf-8"))
    dates = parse.load_dates()
    gen = OfficialGenerator(backend=backend)
    preds, t0 = [], time.time()
    for item in items:
        pred = answer_one(gen, item, dates, tau, check_numbers, mode, strategy, verify,
                          second_context=second_context)
        preds.append(pred)
        print(f"{pred['qid']}: {pred['answer'][:90]}", flush=True)
    print(f"[generate] {(time.time() - t0) / max(len(preds), 1):.1f} s par question")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    debug = Path("runs") / (Path(out_path).stem + ".debug.json")  # détail des abstentions, pour l'analyse
    debug.parent.mkdir(parents=True, exist_ok=True)
    debug.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    clean = [{k: p[k] for k in ("qid", "answer", "citations", "retrieved")} for p in preds]
    write_predictions(out_path, clean, run_name=run_name)
