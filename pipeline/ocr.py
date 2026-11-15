"""OCR des pages sans couche texte. Le modèle n'est chargé que si une page en a besoin."""
from __future__ import annotations


class LazyOCR:
    """Appelable : pixmap PyMuPDF -> texte. `engine` : "auto", "none", ou un nom de moteur."""

    def __init__(self, engine: str = "auto"):
        self.engine = engine
        self._fn = None

    def __call__(self, pix) -> str:
        if self.engine == "none":
            return ""
        if self._fn is None:
            self._fn = _load(self.engine)
        return self._fn(pix)

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


def _to_pil(pix):
    from PIL import Image

    return Image.frombytes("RGB" if pix.n >= 3 else "L", (pix.w, pix.h), pix.samples).convert("RGB")


def _load(engine: str):
    if engine == "auto":  # OCR VLM sur GPU ; sans GPU il serait trop lent, repli sur RapidOCR
        import torch

        engine = "lighton" if torch.cuda.is_available() else "rapidocr"
    if engine == "lighton":
        return _lighton()
    if engine == "rapidocr":
        return _rapidocr()
    raise ValueError(f"moteur OCR inconnu : {engine}")


LIGHTON = "lightonai/LightOnOCR-2-1B"


def _lighton(max_new_tokens: int = 3000):
    """OCR VLM (page image -> Markdown, tableaux compris). Usage repris de la fiche du modèle."""
    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    # T4 (sm75) : le bfloat16 n'y est qu'émulé, on reste en float16 comme le générateur officiel
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = LightOnOcrForConditionalGeneration.from_pretrained(LIGHTON, dtype=dtype).to("cuda").eval()
    processor = LightOnOcrProcessor.from_pretrained(LIGHTON)

    def run(pix) -> str:
        conversation = [{"role": "user", "content": [{"type": "image", "image": _to_pil(pix)}]}]
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: v.to("cuda", dtype=dtype) if v.is_floating_point() else v.to("cuda")
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return run


def _rapidocr():
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    model = RapidOCR()

    def run(pix) -> str:
        img = np.asarray(_to_pil(pix))
        res, _ = model(img)
        return "\n".join(r[1] for r in res or [])

    return run
