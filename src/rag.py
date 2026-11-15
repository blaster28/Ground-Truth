"""RAG de base du palier 1 : il assemble les briques from scratch de src/.

    python -m src.rag evaluer --methode bm25              # recherche seule, sur la validation
    python -m src.rag evaluer --methode dense
    python -m src.rag repondre --methode bm25 --backend hf --sortie runs/palier1_bm25.json

Chaîne complète : PDF -> pages (PyMuPDF) -> chunks (taille fixe) -> recherche (BM25 ou dense)
-> prompt avec les meilleurs chunks -> générateur officiel -> réponse + citations [doc_id p.N].
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # voir le README : JAX ne doit pas réserver le GPU

import numpy as np

from src import dense
from src.bm25 import BM25, tokeniser
from src.chunking import decouper
from src.metrics import mrr, ndcg_at_k, recall_at_k

CONSIGNE = (
    "Tu réponds à des questions sur des rapports publics français, uniquement à partir des extraits "
    "fournis. Chaque extrait est précédé de sa référence [doc_id p.N]. "
    "Si la réponse n'est pas dans les extraits, réponds exactement : INTROUVABLE. "
    "Sinon, réponds en une ou deux phrases en recopiant les chiffres tels quels, puis indique tes "
    "sources sur une dernière ligne au format SOURCES: [doc_id p.N], [doc_id p.N]"
)


# ------------------------------------------------------------------ préparation du corpus
def lire_pages(dossier_pdf):
    """Une entrée par page : le doc_id est le nom du fichier, la page est comptée à partir de 1."""
    import pymupdf

    pages = []
    for pdf in sorted(Path(dossier_pdf).glob("*.pdf")):
        with pymupdf.open(pdf) as document:
            for numero, page in enumerate(document, 1):
                pages.append({"doc_id": pdf.stem, "page": numero, "texte": page.get_text("text")})
    return pages


def construire_chunks(pages, taille=200, recouvrement=40):
    """Découpe chaque page ; chaque chunk garde son doc_id et sa page pour pouvoir être cité."""
    chunks = []
    for page in pages:
        for morceau in decouper(page["texte"], taille, recouvrement):
            chunks.append({"doc_id": page["doc_id"], "page": page["page"], "texte": morceau})
    return chunks


def charger_chunks(dossier_pdf, cache="index/palier1_chunks.json"):
    """Lire 110 PDF prend du temps : on garde le résultat dans un fichier."""
    if Path(cache).exists():
        return json.loads(Path(cache).read_text(encoding="utf-8"))
    chunks = construire_chunks(lire_pages(dossier_pdf))
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    Path(cache).write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return chunks


# ------------------------------------------------------------------ les deux moteurs de recherche
class MoteurBM25:
    def __init__(self, chunks):
        self.bm25 = BM25()
        self.bm25.indexer([tokeniser(c["texte"]) for c in chunks])

    def rechercher(self, question, k):
        return [numero for numero, score in self.bm25.rechercher(tokeniser(question), k)]


class MoteurDense:
    def __init__(self, chunks, cache="index/palier1_embeddings.npy"):
        self.modele = dense.charger_modele()
        if Path(cache).exists():  # encoder tout le corpus est long : on le fait une seule fois
            self.matrice = np.load(cache)
        else:
            self.matrice = dense.encoder(self.modele, [c["texte"] for c in chunks])
            np.save(cache, self.matrice)

    def rechercher(self, question, k):
        return [numero for numero, score in dense.rechercher(self.modele, question, self.matrice, k)]


def creer_moteur(methode, chunks):
    return MoteurBM25(chunks) if methode == "bm25" else MoteurDense(chunks)


def pages_classees(chunks, numeros):
    """Passe d'un classement de chunks à un classement de pages (plusieurs chunks par page)."""
    pages = []
    for numero in numeros:
        page = (chunks[numero]["doc_id"], chunks[numero]["page"])
        if page not in pages:
            pages.append(page)
    return pages


# ------------------------------------------------------------------ évaluation de la recherche
def evaluer(moteur, chunks, questions):
    """Moyenne de Recall@5, Recall@10, MRR et nDCG@10 sur les questions qui ont une réponse."""
    totaux = {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}
    nombre = 0
    for q in questions:
        if not q["answerable"]:
            continue  # une question piège n'a pas de page pertinente
        pertinentes = {(e["doc_id"], e["page"]) for groupe in q["evidence"] for e in groupe}
        trouvees = pages_classees(chunks, moteur.rechercher(q["question"], 50))
        totaux["recall@5"] += recall_at_k(trouvees, pertinentes, 5)
        totaux["recall@10"] += recall_at_k(trouvees, pertinentes, 10)
        totaux["mrr"] += mrr(trouvees, pertinentes, len(trouvees) or 1)
        totaux["ndcg@10"] += ndcg_at_k(trouvees, pertinentes, 10)
        nombre += 1
    return {nom: round(total / nombre, 4) for nom, total in totaux.items()}


# ------------------------------------------------------------------ génération
def construire_prompt(generateur, question, extraits, limite=4000):
    """Retire les extraits les moins bien classés tant que le prompt dépasse la limite de tokens."""
    while True:
        blocs = [f"[{c['doc_id']} p.{c['page']}]\n{c['texte']}" for c in extraits]
        messages = [{"role": "system", "content": CONSIGNE},
                    {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(blocs) + f"\n\nQuestion : {question}"}]
        if generateur.count_tokens(messages) <= limite or len(extraits) == 1:
            return messages
        extraits = extraits[:-1]


def extraire_citations(texte):
    """Sépare la réponse de la ligne SOURCES et lit les références [doc_id p.N]."""
    reponse = re.split(r"\n?SOURCES\s*:", texte)[0].strip()
    citations = []
    for doc_id, page in re.findall(r"\[([\w\-]+) p\.?\s*(\d+)\]", texte):
        citation = {"doc_id": doc_id, "page": int(page)}
        if citation not in citations:
            citations.append(citation)
    return reponse, citations[:5]  # 5 citations au maximum, sinon elles sont jugées invalides


def repondre(moteur, chunks, questions, backend, sortie, nb_extraits=5):
    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import write_predictions

    generateur = OfficialGenerator(backend=backend)
    predictions = []
    for q in questions:
        generateur.start_question(q["qid"])  # remet à zéro le compteur d'appels (4 au maximum)
        numeros = moteur.rechercher(q["question"], 50)
        extraits = [chunks[n] for n in numeros[:nb_extraits]]
        messages = construire_prompt(generateur, q["question"], extraits)
        brut = generateur.chat(messages, max_new_tokens=200)
        if "INTROUVABLE" in brut.upper():
            reponse, citations = "INTROUVABLE", []
        else:
            reponse, citations = extraire_citations(brut)
        predictions.append({
            "qid": q["qid"], "answer": reponse, "citations": citations,
            "retrieved": [{"doc_id": d, "page": p} for d, p in pages_classees(chunks, numeros)[:10]],
        })
        print(f"{q['qid']}: {reponse[:80]}")
    write_predictions(sortie, predictions, run_name="palier1")


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("commande", choices=["evaluer", "repondre"])
    parseur.add_argument("--methode", default="bm25", choices=["bm25", "dense"])
    parseur.add_argument("--pdf", default="corpus/pdf")
    parseur.add_argument("--questions", default="data/validation_v1.json")
    parseur.add_argument("--backend", default="hf", choices=["hf", "openai", "echo"])
    parseur.add_argument("--sortie", default="runs/palier1.json")
    args = parseur.parse_args()

    from groundtruth.io import load_questions

    chunks = charger_chunks(args.pdf)
    questions = load_questions(args.questions)
    moteur = creer_moteur(args.methode, chunks)
    if args.commande == "evaluer":
        print(args.methode, evaluer(moteur, chunks, questions))
    else:
        repondre(moteur, chunks, questions, args.backend, args.sortie)


if __name__ == "__main__":
    main()
