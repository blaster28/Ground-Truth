"""OCR des pages sans couche texte. Le modèle n'est chargé que si une page en a besoin.

Les pages sont traitées par lots : la génération coûte ~55 ms par token quel que soit le nombre de
pages en parallèle (coût fixe de la boucle de génération), donc un lot de N pages va presque N fois
plus vite que N pages une par une.
"""
from __future__ import annotations

import time

LIGHTON = "lightonai/LightOnOCR-2-1B"


class LazyOCR:
    """`engine` : "auto", "none", "lighton" ou "rapidocr". `many(images)` -> un texte par image."""

    def __init__(self, engine: str = "auto", batch_size: int | None = None):
        self.engine = engine
        self.batch_size = batch_size
        self._fn = None

    def many(self, images: list, labels: list[str] | None = None, on_batch=None) -> list[str]:
        """`on_batch(début, textes du lot)` est appelé après chaque lot (sauvegarde du cache)."""
        if self.engine == "none" or not images:
            return ["" for _ in images]
        if self._fn is None:
            self._fn, auto_batch = _load(self.engine)
            self.batch_size = self.batch_size or auto_batch
        out = []
        for start in range(0, len(images), self.batch_size):
            t0 = time.time()
            batch = images[start:start + self.batch_size]
            texts = self._fn(batch)
            out.extend(texts)
            if on_batch is not None:
                on_batch(start, texts)
            names = ", ".join((labels or [])[start:start + self.batch_size][:2])
            print(f"[ocr] pages {start + 1}-{start + len(batch)} / {len(images)} ({names}...) : "
                  f"{time.time() - t0:.0f} s", flush=True)
        return out

    def close(self) -> None:
        self._fn = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def to_pil(pix):
    from PIL import Image

    return Image.frombytes("RGB" if pix.n >= 3 else "L", (pix.w, pix.h), pix.samples).convert("RGB")


def _load(engine: str):
    """Renvoie (fonction lot d'images -> textes, taille de lot conseillée)."""
    if engine == "auto":  # OCR VLM sur GPU ; sans GPU il serait trop lent, repli sur RapidOCR
        import torch

        engine = "lighton" if torch.cuda.is_available() else "rapidocr"
    if engine == "lighton":
        return _lighton()
    if engine == "rapidocr":
        return _rapidocr(), 1
    raise ValueError(f"moteur OCR inconnu : {engine}")


def _lighton(max_new_tokens: int = 3000):
    """OCR VLM (page image -> Markdown, tableaux compris). Usage repris de la fiche du modèle."""
    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    # T4 (sm75) : le bfloat16 n'y est qu'émulé, on reste en float16 comme le générateur officiel
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = LightOnOcrForConditionalGeneration.from_pretrained(LIGHTON, dtype=dtype).to("cuda").eval()
    processor = LightOnOcrProcessor.from_pretrained(LIGHTON)
    processor.tokenizer.padding_side = "left"  # génération par lot : le remplissage va à gauche
    vram_go = torch.cuda.get_device_properties(0).total_memory / 2**30
    # Mesuré : un lot de 8 pages monte à 13,3 Go de VRAM (≈ 1,4 Go par page + 1,9 Go de modèle).
    # Si la VRAM déborde, Windows bascule sur la mémoire partagée et tout devient 20 fois plus lent.
    batch_size = 4 if vram_go >= 14 else 2 if vram_go >= 7 else 1

    def run(images: list) -> list[str]:
        convs = [[{"role": "user", "content": [{"type": "image", "image": im}]}] for im in images]
        inputs = processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda", dtype=dtype) if v.is_floating_point() else v.to("cuda")
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return run, batch_size


def _rapidocr():
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    model = RapidOCR()

    def run(images: list) -> list[str]:
        texts = []
        for im in images:
            res, _ = model(np.asarray(im))
            texts.append("\n".join(r[1] for r in res or []))
        return texts

    return run
