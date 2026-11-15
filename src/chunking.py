"""Chunker naïf : des morceaux de taille fixe (en mots) qui se recouvrent.

Pourquoi un recouvrement : si une phrase utile est coupée en deux par la frontière entre deux
chunks, elle se retrouve entière dans au moins l'un des deux.
"""


def decouper(texte, taille=200, recouvrement=40):
    """Découpe `texte` en chunks de `taille` mots ; deux chunks voisins partagent `recouvrement` mots.

    Exemple avec taille=5 et recouvrement=2 : le pas d'avancement vaut 5 - 2 = 3, donc les chunks
    couvrent les mots 0-4, 3-7, 6-10, etc.
    """
    if taille <= 0:
        raise ValueError("taille doit être strictement positive")
    if recouvrement < 0 or recouvrement >= taille:
        raise ValueError("recouvrement doit être compris entre 0 et taille - 1")

    mots = texte.split()  # découpe sur les espaces et les retours à la ligne
    pas = taille - recouvrement
    chunks = []
    debut = 0
    while debut < len(mots):
        morceau = mots[debut:debut + taille]
        chunks.append(" ".join(morceau))
        if debut + taille >= len(mots):  # ce chunk va jusqu'au dernier mot : on a fini
            break
        debut += pas
    return chunks
