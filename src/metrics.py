import math


def recall_at_k(docs, docs_pertinent, k):
    if k <= 0:
        raise ValueError("k doit être strictement positif")
    if len(docs_pertinent) == 0:
        raise ValueError("ensemble de documents pertinents vide")

    pertinents = set(docs_pertinent)
    trouves = set()
    for doc in docs[:k]:
        if doc in pertinents:
            trouves.add(doc)

    return len(trouves) / len(pertinents)

def mrr(docs, docs_pertinent, k):
    if k <= 0:
        raise ValueError("k doit être strictement positif")
    if len(docs_pertinent) == 0:
        raise ValueError("ensemble de documents pertinents vide")

    pertinents = set(docs_pertinent)
    for i, doc in enumerate(docs[:k]):
        if doc in pertinents:
            return 1 / (i + 1)
    return 0.0


def ndcg_at_k(docs, docs_pertinent, k=10):
    """nDCG@k avec une pertinence binaire (un document est pertinent ou non).

    DCG  = somme des gains des documents pertinents, chacun divisé par log2(rang + 1) :
           un bon document au rang 1 rapporte 1, au rang 2 il rapporte 1/log2(3) = 0,63, etc.
    IDCG = le DCG du classement parfait (tous les pertinents en tête).
    nDCG = DCG / IDCG, donc entre 0 et 1.
    """
    if k <= 0:
        raise ValueError("k doit être strictement positif")
    if len(docs_pertinent) == 0:
        raise ValueError("ensemble de documents pertinents vide")

    pertinents = set(docs_pertinent)
    deja_vus = set()  # un document pertinent répété ne doit compter qu'une fois
    dcg = 0.0
    for i, doc in enumerate(docs[:k]):
        rang = i + 1
        if doc in pertinents and doc not in deja_vus:
            dcg += 1 / math.log2(rang + 1)
            deja_vus.add(doc)

    # classement idéal : les pertinents occupent les premiers rangs (au plus k d'entre eux)
    nb_ideal = min(len(pertinents), k)
    idcg = 0.0
    for rang in range(1, nb_ideal + 1):
        idcg += 1 / math.log2(rang + 1)

    return dcg / idcg