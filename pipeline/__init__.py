"""Pipeline des paliers 2 et 3 : parsing + OCR, chunking, recherche hybride, reranking, abstention.

Le code from scratch du palier 1 vit dans `src/` ; ici les librairies sont libres.
"""
import os

# bm25s importe JAX, qui réserve 75 % de la VRAM à l'import (voir README) : on le force sur CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
