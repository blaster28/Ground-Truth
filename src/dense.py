"""Recherche dense from scratch : embeddings, similarité cosinus et top-k en NumPy (sans FAISS).

L'idée : un modèle d'embedding transforme un texte en vecteur (ici 1 024 nombres) de sorte que deux
textes de sens proche donnent des vecteurs qui pointent dans la même direction, même s'ils
n'ont aucun mot en commun. C'est ce qui manque à BM25, qui ne voit que les mots exacts.

Similarité cosinus entre deux vecteurs a et b :

    cos(a, b) = (a . b) / (||a|| * ||b||)        entre -1 et 1, 1 = même direction

Astuce : si on divise chaque vecteur par sa norme une fois pour toutes (norme = 1), le cosinus
devient un simple produit scalaire, et comparer la requête à TOUS les chunks tient en un seul
produit matrice x vecteur.

Seul le calcul des embeddings utilise un modèle (open weights) ; la recherche est codée ici.
"""
import numpy as np

MODELE = "BAAI/bge-m3"  # multilingue (questions en anglais sur corpus français), open weights


def charger_modele(nom=MODELE):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nom)


def normaliser(vecteurs):
    """Ramène chaque ligne de la matrice à une norme de 1."""
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)  # une norme par ligne, forme (n, 1)
    normes[normes == 0] = 1.0  # évite une division par zéro pour un vecteur nul
    return vecteurs / normes


def encoder(modele, textes, taille_lot=32):
    """Textes -> matrice (nombre de textes, dimension), lignes de norme 1."""
    vecteurs = modele.encode(textes, batch_size=taille_lot, convert_to_numpy=True,
                             show_progress_bar=len(textes) > 200)
    return normaliser(vecteurs.astype(np.float32))


def similarites(vecteur_requete, matrice_chunks):
    """Cosinus entre la requête et chaque chunk (les vecteurs sont déjà normalisés).

    matrice_chunks a la forme (n, d), vecteur_requete la forme (d,) : le produit donne n scores.
    """
    return matrice_chunks @ vecteur_requete


def top_k(scores, k=10):
    """Indices des k plus grands scores, du meilleur au moins bon.

    np.argsort trie par ordre croissant : on trie donc l'opposé des scores.
    """
    ordre = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in ordre[:k]]


def rechercher(modele, question, matrice_chunks, k=10):
    """Les k chunks les plus proches de la question : liste de (numéro de chunk, cosinus)."""
    vecteur_requete = encoder(modele, [question])[0]
    return top_k(similarites(vecteur_requete, matrice_chunks), k)
