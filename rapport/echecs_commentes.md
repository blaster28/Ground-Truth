# Cinq réponses ratées, commentées

Source : stratégie « réponse directe, 1 appel » sur `validation_v1` (fichiers
`runs/dev_direct_val1.debug.json` et `runs/dev_direct_rapport1.json`, générateur et juge
quantifiés, en local). C'est la configuration la plus proche d'un RAG de base : la recherche du
pipeline, puis une seule réponse du générateur, sans garde-fou de consensus. Sur cette
configuration, la v1 compte 25 réponses fausses et 2 hallucinations sur des pièges.

Constat commun : dans quatre cas sur cinq, la page de preuve était dans les pages trouvées
(l'échec 3 est l'exception). L'erreur vient de la lecture du générateur, pas de la recherche. Dans les cinq cas, la
version finale (consensus + éditions) répond INTROUVABLE, ce qui transforme −1 en 0 (ou en +1 pour
le piège).

## 1. Mauvaise ligne d'un tableau (val1-083, type tableau)

- **Question** : en 2023, quel salaire net mensuel sépare en deux moitiés les femmes qui
  travaillent pour l'État, en équivalent temps plein ?
- **Référence** : 2 591 euros (femmes de la FPE).
- **Réponse** : « 2 682 euros », valeur de la ligne « ensemble » du même tableau.
- **Recherche** : page de preuve (insee-ip2065 p. 3) classée 1re.
- **Cause** : le générateur lit la bonne page mais prend la cellule de la ligne voisine.
- **Version finale** : INTROUVABLE, désaccord entre les deux avis.

## 2. Valeur d'un graphique absente du texte (val1-042, type graphique)

- **Question** : en 2024, combien d'argent public est dépensé par les collectivités et autres
  organismes locaux ?
- **Référence** : 322 Md€.
- **Réponse** : « 166 Md€ », un autre chiffre de la même page.
- **Recherche** : page de preuve (insee-ip2093 p. 2) classée 2e.
- **Cause** : la valeur est portée par un graphique, que l'extraction de texte ne récupère pas ;
  le générateur se rabat sur le seul nombre plausible qu'il voit.
- **Version finale** : INTROUVABLE, preuve recopiée absente des extraits.

## 3. Page voisine manquante (val1-066, type tableau)

- **Question** : en 2025, à combien s'élevait en moyenne un étalement de dettes fiscales et
  sociales accordé par les CCSF ?
- **Référence** : 492 637 euros en moyenne.
- **Réponse** : « 1,8 Md€ pour 3 786 nouveaux plans », un total et non une moyenne.
- **Recherche** : la page de preuve (cc-difficultes-entreprises p. 61) n'est pas dans les pages
  trouvées ; le générateur cite la p. 60.
- **Cause** : seul échec de recherche des cinq ; le générateur répond quand même à partir de la
  page voisine.
- **Version finale** : INTROUVABLE, ancrage numérique.

## 4. Hallucination sur une question piège (val1-051, type piège)

- **Question** : en 2024, combien de millions d'hectolitres de vin ont été produits en France ?
- **Référence** : l'information n'est pas dans le corpus (INTROUVABLE).
- **Réponse** : « 42,9 millions d'hectolitres », en citant deux Insee Première sur l'agriculture.
- **Cause** : le sujet est bien présent dans le corpus, donc la recherche trouve des pages très
  pertinentes, et le générateur fabrique un chiffre plausible.
- **Version finale** : INTROUVABLE, preuve recopiée absente des extraits (abstention juste, +1).

## 5. Bon nombre, mauvaise entité (val1-097, type tableau)

- **Question** : de combien ont augmenté les crédits des « Sages de la rue de Montpensier »
  entre 2020 et 2025 ?
- **Référence** : la dotation du Conseil constitutionnel, +43,4 %.
- **Réponse** : « La dotation de la Cour des comptes a augmenté de 43,4 % ».
- **Recherche** : page de preuve (cc-presidence-republique-2025 p. 19) classée 2e.
- **Cause** : le chiffre est juste, mais la périphrase n'est pas comprise et la réponse attribue
  la hausse à la mauvaise institution : le juge la compte fausse.
- **Version finale** : INTROUVABLE, désaccord entre les deux avis.
