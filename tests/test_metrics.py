import pytest
from src.metrics import recall_at_k

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