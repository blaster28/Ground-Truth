#!/usr/bin/env bash
# Phase d'installation de la finale : seule phase où le réseau est disponible.
# Installe tes dépendances et télécharge TOUS les poids dont ton pipeline a besoin
# (générateur officiel, embedder, reranker, OCR...) dans le cache Hugging Face.
set -euo pipefail
pip install -q -r requirements.txt
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "mistralai/Ministral-3-3B-Instruct-2512",   # générateur officiel (obligatoire)
    "BAAI/bge-m3",                               # embedder (recherche dense)
    "BAAI/bge-reranker-v2-m3",                   # reranker
    "lightonai/LightOnOCR-2-1B",                 # OCR des pages numérisées
]:
    # on évite les copies ONNX / poids en double, inutiles ici
    snapshot_download(repo, ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data", "pytorch_model.bin", "imgs/*"])
PY
