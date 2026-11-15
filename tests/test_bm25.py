import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # bm25s importe JAX : on l'empêche de réserver le GPU

import math

import pytest
from src.bm25 import BM25, tokeniser

CORPUS = [
    "Le chat mange la souris dans la cuisine",
    "Le chien dort dans le jardin",
    "La souris grise mange le fromage, le chat regarde la souris",
    "Les dépenses de santé atteignent 261 milliards d'euros en 2024",
    "Le budget de la santé publique augmente de 3 % en 2024",
]


def test_tokeniser_minuscules_accents_ponctuation():
    assert tokeniser("Les Dépenses de santé : 261 Md€ !") == ["les", "depenses", "de", "sante", "261", "md"]


def test_index_inverse():
    bm25 = BM25()
    bm25.indexer([["a", "b", "a"], ["b", "c"]])
    assert bm25.index["a"] == {0: 2}
    assert bm25.index["b"] == {0: 1, 1: 1}
    assert bm25.longueurs == [3, 2]
    assert bm25.longueur_moyenne == pytest.approx(2.5)


def test_score_calcule_a_la_main():
    # 2 documents, le mot "a" est dans un seul : df = 1, N = 2
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.indexer([["a", "b", "a"], ["b", "c"]])
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    normalisation = 1 - 0.75 + 0.75 * 3 / 2.5
    attendu = idf * 2 / (2 + 1.5 * normalisation)
    assert bm25.scores(["a"])[0] == pytest.approx(attendu)


def test_mot_rare_pese_plus_que_mot_frequent():
    bm25 = BM25()
    bm25.indexer([tokeniser(d) for d in CORPUS])
    assert bm25.idf("fromage") > bm25.idf("le")


def test_mot_inconnu():
    bm25 = BM25()
    bm25.indexer([tokeniser(d) for d in CORPUS])
    assert bm25.rechercher(["licorne"]) == []


def test_identique_a_bm25s():
    """Vérification contre la librairie de référence : mêmes scores, même classement."""
    bm25s = pytest.importorskip("bm25s")
    docs = [tokeniser(d) for d in CORPUS]

    reference = bm25s.BM25(k1=1.5, b=0.75)  # méthode par défaut : "lucene"
    reference.index(docs, show_progress=False)
    le_notre = BM25(k1=1.5, b=0.75)
    le_notre.indexer(docs)

    for question in ["chat et souris", "dépenses de santé en 2024", "le jardin du chien"]:
        requete = tokeniser(question)
        ids_ref, scores_ref = reference.retrieve([requete], k=len(docs), show_progress=False)
        nos_resultats = dict(le_notre.rechercher(requete, k=len(docs)))
        for num_doc, score_ref in zip(ids_ref[0], scores_ref[0]):
            if score_ref > 0:  # bm25s liste aussi les documents de score nul, pas nous
                assert nos_resultats[int(num_doc)] == pytest.approx(float(score_ref), rel=1e-4)
        assert len(nos_resultats) == sum(s > 0 for s in scores_ref[0])
