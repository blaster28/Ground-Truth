# Rapport d'ablation

Ground Truth, piscine I Lab. Toutes les mesures portent sur les jeux de validation du kit :
`validation_v1` (100 questions, dont 23 pièges) et `validation_v2` (40 questions, sans piège).

## 1. Le pipeline complet

| Étape | Choix |
|---|---|
| Lecture | PyMuPDF ; pages sans couche texte lues par LightOnOCR-2-1B, par lots, avec cache |
| Chunking | par paragraphes regroupés (~220 mots), jamais à cheval sur deux pages, préfixe titre + date |
| Recherche | BM25 + dense (bge-m3), fusion RRF, puis reranker bge-reranker-v2-m3 sur 30 passages |
| Éditions | détection des éditions successives d'une publication, la plus récente est ramenée ou imposée |
| Contexte | texte complet des 2 meilleures pages, puis meilleurs passages, sous 3 400 tokens |
| Réponse | consensus : le générateur recopie sa preuve, puis un second avis doit concorder |
| Citation | la page où la preuve recopiée a été retrouvée |

## 2. La recherche : chaque brique sert

Mesure officielle du kit sur les 77 questions de `validation_v1` qui ont une réponse.

| Configuration | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| BM25 seul | 0,695 | 0,786 | 0,565 |
| Dense seul | 0,805 | 0,890 | 0,642 |
| Hybride (RRF), sans reranker | 0,805 | 0,922 | 0,627 |
| **Hybride + reranker** | **0,883** | **0,935** | **0,737** |

- **L'OCR est la brique la plus rentable sur son périmètre.** Sans OCR, le Recall@5 des questions
  « scan » est de 0,11 ; avec, il est de 0,89.
- **Le reranker apporte le plus au classement** : +0,11 de MRR, la bonne page remonte en tête.
- **La fusion seule n'améliore pas le top 5** par rapport au dense, mais le top 10 (+0,03) : elle
  élargit ce que le reranker peut ensuite reclasser.
- **Ramener les pluriels au singulier** dans BM25 : Recall@5 de 0,688 à 0,695, MRR de 0,532 à 0,573.

Éditions successives, sur `validation_v2` (index v1 + v2) :

| | Recall@5 | Recall@10 | Recall@5 des questions temporelles |
|---|---|---|---|
| Sans gestion des éditions | 0,812 | 0,975 | 0,75 |
| Avec | 0,863 | 1,000 | 0,92 |

## 3. La génération : savoir se taire rapporte plus que mieux répondre

Score de confiance sur les 140 questions (v1 + v2). Générateur officiel en version quantifiée (GGUF Q8)
servi en local, juge officiel en version quantifiée. Ce juge local redonne exactement le score
officiel obtenu sur T4 pour les mêmes réponses (0,18), ce qui valide son usage pour comparer.

| Stratégie | v1 | v2 | v1 + v2 |
|---|---|---|---|
| Toujours répondre INTROUVABLE (plancher) | 0,23 | 0,00 | 0,164 |
| Réponse directe, 1 appel | 0,245 | −0,10 | 0,146 |
| Preuve recopiée et vérifiée, 1 appel | 0,22 | −0,025 | 0,150 |
| Preuve + relecture par le générateur, 2 appels | 0,21 | 0,05 | 0,164 |
| Consensus de deux avis, 2 appels | 0,42 | 0,025 | 0,307 |
| **Consensus + gestion des éditions** | **0,42** | **0,125** | **0,336** |

Trois enseignements :

1. **Le générateur de 3B a raison environ une fois sur deux quand il répond.** Avec un barème à −1,
   répondre à tout fait moins bien que ne jamais répondre. La recherche n'est pas en cause : la
   bonne page est dans le contexte dans la grande majorité des erreurs.
2. **Il ne sait pas se relire.** Quand on lui demande de valider sa propre réponse, il refuse
   presque tout (96 % d'abstentions).
3. **Il se trompe rarement deux fois de la même façon.** Deux consignes différentes qui donnent
   les mêmes nombres : 70 % de réponses justes. Sur la v1, les réponses fausses passent de 25 à 5 et
   les 23 pièges sont tous évités, au prix de 48 abstentions à tort.

## 4. Ce qui ne sert à rien, ou pas encore prouvé

- **Seuil sur le score du reranker.** Le meilleur passage d'une question piège a un score médian
  de 0,96 (0,99 pour une vraie question) : les pièges portent sur des sujets présents dans le
  corpus. Un seuil à 0,05 n'écarte que 4 % des pièges.
- **Second avis sur un autre contexte** (passages seuls au lieu des pages entières) : 0,286 contre
  0,307, dans le bruit.
- **Préfixe contextuel des chunks** : il fait baisser BM25 seul (Recall@5 de 0,630 à 0,578, mesuré
  sans OCR). Son effet sur le pipeline complet n'a pas été isolé.
- **Pages entières dans le contexte** et **consigne plus stricte** : introduites ensemble, effet
  de chacune non isolé.
- **Stratégies de chunking** (BM25 seul, sans OCR) : page 0,591 / fenêtre 0,597 / structure 0,630
  au top 5, mais 0,688 / 0,701 / 0,669 au top 10. Pas de gagnant net.

## 5. Où le système hallucine encore

| Type | Score (consensus + éditions) | Cause |
|---|---|---|
| Tableau | 0,12 (v1), −0,17 (v2) | mauvaise ligne, mauvaise année ou mauvaise catégorie |
| Graphique | 0,08 (v1), 0,00 (v2) | les valeurs d'un graphique ne sont pas dans le texte extrait |
| Temporel | 0,08 | comparaison de deux éditions, que le filtre de récence ne traite pas |
| Synthèse | 0,00 (v1) | sans nombre, le test de concordance repose sur les mots et écarte presque tout |
| Multi-documents | 0,00 | il faut deux preuves ; l'abstention l'emporte |

## 6. Un bug qui a coûté un passage entier

Sur T4, l'OCR tourne en float16. La dernière couche de l'encodeur d'image dépasse alors 65 504,
le maximum du float16 : les activations deviennent infinies et le modèle n'écrit que des « ! ».
Premier passage officiel : score « scan » de 0,00, et 0,18 / −0,15 au total. Correctif : encodeur
d'image et projection en float32, le reste en float16, plus un garde-fou qui arrête l'indexation
si l'OCR rend un caractère répété. Texte vérifié identique à la référence sur 4 pages entières.

## 7. Limites de ce rapport

- Les scores de la section 3 viennent de versions quantifiées du générateur et du juge.
- 140 questions : un écart de 0,03 représente 4 questions.
- Non fait : la comparaison de deux parsings sur 30 pages annotées à la main, la courbe complète
  du seuil d'abstention, les questions red team.
