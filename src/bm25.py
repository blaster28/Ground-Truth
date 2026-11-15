"""BM25 from scratch : tokenisation, index inversé, score Okapi.

L'idée de BM25 : un document est pertinent pour une requête s'il contient ses mots, avec trois
corrections de bon sens.
  1. Un mot rare compte plus qu'un mot fréquent (« vaccination » en dit plus que « france ») : l'IDF.
  2. Répéter un mot aide, mais de moins en moins (10 occurrences ne valent pas 10 fois une) : la
     saturation, réglée par k1.
  3. Un document long contient mécaniquement plus de mots : on normalise par sa longueur, réglé
     par b.

Score d'un document d pour une requête q :

    score(q, d) = somme, pour chaque mot t de q, de   idf(t) * tf / (tf + k1 * (1 - b + b * |d| / moy))

    tf   = nombre d'occurrences de t dans d
    |d|  = longueur de d en mots, moy = longueur moyenne des documents
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)),  N = nombre de documents, df = nombre de documents
             qui contiennent t

C'est la variante « Lucene » de BM25, celle que `bm25s` utilise par défaut, ce qui permet de
comparer nos scores aux siens (voir tests/test_bm25.py).
"""
import math
import re
import unicodedata


def tokeniser(texte):
    """Texte -> liste de mots en minuscules, sans accents ni ponctuation.

    « Les Dépenses de santé : 261 Md€ » -> ["les", "depenses", "de", "sante", "261", "md"]
    """
    texte = texte.lower()
    # NFKD sépare une lettre accentuée en lettre + accent (« é » -> « e » + « ´ ») ;
    # l'encodage ASCII avec "ignore" jette ensuite tout ce qui n'est pas ASCII, donc l'accent.
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", texte)  # suites de lettres ou de chiffres


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # saturation de la fréquence (plus k1 est grand, plus répéter un mot rapporte)
        self.b = b    # force de la normalisation par la longueur (0 = aucune, 1 = totale)
        self.index = {}       # index inversé : mot -> {numéro de document: nombre d'occurrences}
        self.longueurs = []   # nombre de mots de chaque document
        self.longueur_moyenne = 0.0
        self.nb_docs = 0

    def indexer(self, documents):
        """Construit l'index inversé à partir d'une liste de documents déjà tokenisés."""
        self.index = {}
        self.longueurs = []
        for num_doc, mots in enumerate(documents):
            self.longueurs.append(len(mots))
            for mot in mots:
                if mot not in self.index:
                    self.index[mot] = {}
                # .get(num_doc, 0) vaut 0 la première fois qu'on voit ce mot dans ce document
                self.index[mot][num_doc] = self.index[mot].get(num_doc, 0) + 1
        self.nb_docs = len(documents)
        self.longueur_moyenne = sum(self.longueurs) / self.nb_docs

    def idf(self, mot):
        """Rareté du mot : grand s'il apparaît dans peu de documents."""
        df = len(self.index.get(mot, {}))  # nombre de documents qui contiennent le mot
        return math.log(1 + (self.nb_docs - df + 0.5) / (df + 0.5))

    def scores(self, requete):
        """Score BM25 de chaque document qui contient au moins un mot de la requête.

        Grâce à l'index inversé, on ne parcourt que les documents concernés, pas tout le corpus.
        Renvoie un dictionnaire {numéro de document: score}.
        """
        resultat = {}
        for mot in requete:
            if mot not in self.index:
                continue  # mot absent du corpus : il n'apporte rien
            idf = self.idf(mot)
            for num_doc, tf in self.index[mot].items():
                normalisation = 1 - self.b + self.b * self.longueurs[num_doc] / self.longueur_moyenne
                score_mot = idf * tf / (tf + self.k1 * normalisation)
                resultat[num_doc] = resultat.get(num_doc, 0.0) + score_mot
        return resultat

    def rechercher(self, requete, k=10):
        """Les k meilleurs documents : liste de (numéro de document, score), du meilleur au moins bon."""
        resultat = self.scores(requete)
        # tri par score décroissant ; à score égal, le plus petit numéro d'abord (résultat stable)
        classes = sorted(resultat.items(), key=lambda paire: (-paire[1], paire[0]))
        return classes[:k]
