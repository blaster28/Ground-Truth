# Fiche système

**Ce que fait le système.** Il répond à des questions sur 130 rapports publics français (Insee
Première, Cour des comptes), cite la page de sa source, et répond `INTROUVABLE` quand il n'est pas
sûr. Il est réglé pour se taire plutôt que se tromper.

## Pipeline

1. **Lecture.** PyMuPDF ; les pages sans couche texte passent par l'OCR LightOnOCR-2-1B.
2. **Chunking.** Paragraphes regroupés en morceaux d'environ 220 mots, limités à une page, précédés
   du titre et de la date du document.
3. **Recherche.** BM25 et recherche dense (BAAI/bge-m3) fusionnés par Reciprocal Rank Fusion, puis
   reranker (BAAI/bge-reranker-v2-m3). Si une édition plus récente de la même publication existe,
   elle est ajoutée aux candidats.
4. **Réponse.** Le générateur officiel recopie la phrase qui justifie sa réponse ; le pipeline
   vérifie qu'elle existe dans les extraits et que les nombres de la réponse en viennent. Un second
   appel, avec une autre consigne, doit donner les mêmes nombres. Sinon : `INTROUVABLE`.
5. **Citation.** La page où la phrase recopiée a été retrouvée.

## Respect des limites

| Limite | Valeur du système |
|---|---|
| 4 096 tokens d'entrée par appel | 3 400 au maximum |
| 4 appels par question | 2 au maximum |
| 30 s par question en moyenne sur T4 | environ 4 s par appel, 0,5 s de recherche |
| Mise à jour en 20 min | moins de 10 min |
| Modèles open weights, aucun réseau à l'exécution | oui |

## Performances mesurées (validation)

- Recherche : bonne page dans le top 5 pour 88 % des questions (v1), 86 % (v2).
- Score de confiance : 0,42 sur la v1, 0,125 sur la v2 (générateur et juge quantifiés, en local).
- Sur la v1 : 24 bonnes réponses toutes bien citées, 5 réponses fausses, 23 pièges évités sur 23.

## Limites connues

- **Il s'abstient beaucoup** : environ 7 questions sur 10 sur la v1, dont 48 qui avaient une réponse.
- **L'OCR domine le temps d'indexation**, autour d'une heure sur T4.
- **Les graphiques ne sont lus que par leur texte** (titres, légendes, étiquettes) ; les valeurs
  portées par les barres ou les courbes sont perdues.
- **Les réponses sont toujours en français**, y compris pour une question posée en anglais.

## Quand ne pas lui faire confiance

| Type de question | Pourquoi |
|---|---|
| Valeur lue sur un graphique | la donnée n'est pas dans le texte extrait |
| Cellule précise d'un grand tableau | risque de confusion de ligne, d'année ou de catégorie |
| Comparaison de deux éditions d'une même publication | le système privilégie une édition |
| Synthèse à plusieurs éléments | il s'abstient presque toujours |
| Croisement de deux documents | il faut deux preuves, il s'abstient le plus souvent |

À l'inverse, une réponse donnée sur une question de texte ou de page scannée est le plus souvent
juste, et sa citation est fiable par construction.
