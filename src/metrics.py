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