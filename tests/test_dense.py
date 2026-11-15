import numpy as np
import pytest
from src.dense import normaliser, similarites, top_k


def test_normaliser_donne_une_norme_de_1():
    v = normaliser(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.linalg.norm(v, axis=1) == pytest.approx([1.0, 1.0])
    assert v[0] == pytest.approx([0.6, 0.8])


def test_cosinus_cas_connus():
    chunks = normaliser(np.array([[1.0, 0.0],    # même direction que la requête
                                  [0.0, 1.0],    # perpendiculaire
                                  [-1.0, 0.0],   # direction opposée
                                  [5.0, 5.0]]))  # à 45 degrés (la longueur ne compte pas)
    requete = normaliser(np.array([[2.0, 0.0]]))[0]
    assert similarites(requete, chunks) == pytest.approx([1.0, 0.0, -1.0, 2 ** 0.5 / 2])


def test_top_k_ordre_decroissant():
    resultat = top_k(np.array([0.1, 0.9, 0.5, 0.7]), k=2)
    assert [i for i, _ in resultat] == [1, 3]
    assert resultat[0][1] == pytest.approx(0.9)


def test_top_k_plus_grand_que_le_nombre_de_scores():
    assert len(top_k(np.array([0.2, 0.1]), k=10)) == 2
