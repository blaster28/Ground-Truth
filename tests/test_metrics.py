import math

import pytest
from src.metrics import mrr, ndcg_at_k, recall_at_k


def test_mrr_premier_pertinent_au_rang_2():
    assert mrr(["d3", "d1", "d2"], {"d1", "d2"}, 10) == pytest.approx(0.5)


def test_mrr_aucun_pertinent():
    assert mrr(["d3", "d4"], {"d1"}, 10) == 0.0


def test_mrr_hors_du_top_k():
    assert mrr(["d3", "d4", "d1"], {"d1"}, 2) == 0.0


def test_ndcg_classement_parfait():
    assert ndcg_at_k(["d1", "d2", "d3"], {"d1", "d2"}, 10) == pytest.approx(1.0)


def test_ndcg_calcul_a_la_main():
    # un seul pertinent, trouvé au rang 3 : DCG = 1/log2(4) = 0,5 ; IDCG = 1/log2(2) = 1
    assert ndcg_at_k(["d5", "d6", "d1"], {"d1"}, 10) == pytest.approx(0.5)


def test_ndcg_deux_pertinents_dont_un_trouve():
    # d1 au rang 2 : DCG = 1/log2(3) ; IDCG = 1/log2(2) + 1/log2(3)
    attendu = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["d9", "d1"], {"d1", "d2"}, 10) == pytest.approx(attendu)


def test_ndcg_doublons():
    assert ndcg_at_k(["d1", "d1"], {"d1"}, 10) == pytest.approx(1.0)


def test_recall_exemple():
    docs = ["d3", "d1", "d7", "d2"]
    pertinents = {"d1", "d2"}
    assert recall_at_k(docs, pertinents, 2) == pytest.approx(0.5)
    assert recall_at_k(docs, pertinents, 4) == pytest.approx(1.0)

def test_recall_k_plus_grand_que_la_liste():
    assert recall_at_k(["d1"], {"d1", "d2"}, 10) == pytest.approx(0.5)

def test_recall_doublons():
    assert recall_at_k(["d1", "d1"], {"d1", "d2"}, 2) == pytest.approx(0.5)

def test_recall_k_invalide():
    with pytest.raises(ValueError):
        recall_at_k(["d1"], {"d1"}, 0)

def test_recall_pertinents_vides():
    with pytest.raises(ValueError):
        recall_at_k(["d1"], set(), 5)