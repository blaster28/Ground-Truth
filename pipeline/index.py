"""Construction, mise à jour incrémentale et chargement de l'index.

Contenu du dossier d'index :
    pages.json    toutes les pages lues (doc_id, page, text, source, title)
    chunks.json   les chunks (doc_id, page, text, embed_text)
    emb.npy       un vecteur normalisé par chunk (float16), même ordre que chunks.json
    meta.json     réglages utilisés (pour que `update` et `answer` restent cohérents)

L'index BM25 n'est pas stocké : il se reconstruit en quelques secondes au chargement, ce qui garde
des statistiques (idf, longueur moyenne) exactes après une mise à jour.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from . import chunking, parse

EMBEDDER = "BAAI/bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"

STOPWORDS = set("""a au aux avec ce ces cette dans de des du elle en et est il ils la le les leur leurs
mais ne ou par pas pour qu que qui sa se ses son sur un une y d l s n c j m t on ont sont ete etre
quel quelle quels quelles combien quoi comment the of in and to is what how many much was were for
on by with""".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    text = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", text)   # « 308 000 » -> « 308000 »
    toks = []
    for t in re.findall(r"[a-z0-9]+", text):
        if t in STOPWORDS:
            continue
        if len(t) > 4 and t[-1] in "sx" and not t[-2].isdigit():  # pluriels : racinisation minimale
            t = t[:-1]
        toks.append(t)
    return toks


# ------------------------------------------------------------------ modèles
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder():
    from sentence_transformers import SentenceTransformer

    dev = device()
    model = SentenceTransformer(EMBEDDER, device=dev)
    model.max_seq_length = 512
    if dev == "cuda":
        model.half()
    return model


def embed(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float16)
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 500, convert_to_numpy=True)
    return vecs.astype(np.float16)


# ------------------------------------------------------------------ écriture
def _save(index_dir: Path, pages, chunks, emb, meta) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(index_dir / "emb.npy", emb)
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _load(index_dir: Path):
    pages = json.loads((index_dir / "pages.json").read_text(encoding="utf-8"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    emb = np.load(index_dir / "emb.npy")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return pages, chunks, emb, meta


def _process(pdf_dir: str, index_dir: Path, meta: dict, skip: set[str] | None = None):
    """Lit, découpe et encode les PDF d'un dossier (hors documents de `skip`)."""
    from . import ocr as ocr_mod

    t0 = time.time()
    reader = ocr_mod.LazyOCR(meta["ocr"])
    pages = parse.read_dir(pdf_dir, ocr=reader, skip=skip, cache_path=str(index_dir / "ocr_cache.json"))
    reader.close()  # libère la VRAM de l'OCR avant de charger l'embedder
    n_ocr = sum(p["source"] == "ocr" for p in pages)
    print(f"[index] {len(pages)} pages lues dont {n_ocr} par OCR ({time.time() - t0:.0f} s)")
    chunks = chunking.make_chunks(pages, meta["chunking"], meta["contextual"], parse.load_dates())
    t0 = time.time()
    emb = embed(load_embedder(), [c["embed_text"] for c in chunks])
    print(f"[index] {len(chunks)} chunks encodés ({time.time() - t0:.0f} s)")
    return pages, chunks, emb


def build(pdf_dir: str, index_dir: str, strategy: str = "structure", contextual: bool = True,
          ocr: str = "auto") -> None:
    meta = {"embedder": EMBEDDER, "chunking": strategy, "contextual": contextual, "ocr": ocr}
    pages, chunks, emb = _process(pdf_dir, Path(index_dir), meta)
    _save(Path(index_dir), pages, chunks, emb, meta)


def update(index_dir: str, pdf_dir: str) -> None:
    """Ajoute les documents inconnus sans relire ni ré-encoder ceux déjà indexés."""
    idx = Path(index_dir)
    pages, chunks, emb, meta = _load(idx)
    known = {p["doc_id"] for p in pages}
    new_pages, new_chunks, new_emb = _process(pdf_dir, idx, meta, skip=known)
    print(f"[update] {len({p['doc_id'] for p in new_pages})} nouveaux documents")
    if not new_pages:
        return
    _save(idx, pages + new_pages, chunks + new_chunks, np.concatenate([emb, new_emb]), meta)


# ------------------------------------------------------------------ recherche
class Searcher:
    def __init__(self, index_dir: str, dense: bool = True, rerank: bool = True):
        import bm25s

        self.pages, self.chunks, self.emb, self.meta = _load(Path(index_dir))
        self.emb = self.emb.astype(np.float32)
        self.page_text = {(p["doc_id"], p["page"]): p for p in self.pages}
        self.bm25 = bm25s.BM25()
        self.bm25.index([tokenize(c["embed_text"]) for c in self.chunks], show_progress=False)
        self.embedder = load_embedder() if dense else None
        self.reranker = None
        if rerank:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANKER, device=device(), max_length=512)
            if device() == "cuda":
                self.reranker.model.half()

    def lexical(self, query: str, k: int) -> list[int]:
        toks = tokenize(query)
        if not toks:
            return []
        res, _ = self.bm25.retrieve([toks], k=min(k, len(self.chunks)), show_progress=False)
        return [int(i) for i in res[0]]

    def dense(self, query: str, k: int) -> list[int]:
        q = embed(self.embedder, [query]).astype(np.float32)[0]
        scores = self.emb @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [int(i) for i in top[np.argsort(-scores[top])]]

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion : score = somme des 1 / (k + rang) sur chaque classement."""
        score: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking, 1):
                score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
        return sorted(score, key=score.get, reverse=True)

    def search(self, query: str, mode: str = "hybrid", k_first: int = 50, k_rerank: int = 30) -> list[dict]:
        """Chunks classés {doc_id, page, text, score}. `score` est celui du reranker s'il est actif
        (probabilité de pertinence entre 0 et 1), sinon un score de rang."""
        if mode == "bm25":
            ids = self.lexical(query, k_first)
        elif mode == "dense":
            ids = self.dense(query, k_first)
        else:
            ids = self.rrf([self.lexical(query, k_first), self.dense(query, k_first)])
        if self.reranker is not None and ids:
            ids = ids[:k_rerank]
            scores = self.reranker.predict([(query, self.chunks[i]["embed_text"]) for i in ids],
                                           batch_size=16, show_progress_bar=False)
            order = np.argsort(-scores)
            ids, scores = [ids[j] for j in order], [float(scores[j]) for j in order]
        else:
            scores = [1.0 / (1 + r) for r in range(len(ids))]
        return [{"doc_id": self.chunks[i]["doc_id"], "page": self.chunks[i]["page"],
                 "text": self.chunks[i]["text"], "score": s} for i, s in zip(ids, scores)]


def ranked_pages(hits: list[dict]) -> list[dict]:
    """Pages distinctes dans l'ordre des chunks (une page vaut son meilleur chunk)."""
    seen, out = set(), []
    for h in hits:
        key = (h["doc_id"], h["page"])
        if key not in seen:
            seen.add(key)
            out.append({"doc_id": h["doc_id"], "page": h["page"]})
    return out
