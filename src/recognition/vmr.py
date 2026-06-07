# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import os
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger


class VMRRecognizer:
    """Vehicle Make & Model Recognition using CLIP zero-shot classification."""

    BRANDS = [
        "Lada", "Kia", "Hyundai", "Toyota", "Volkswagen", "Skoda", "Renault",
        "Nissan", "Mitsubishi", "BMW", "Mercedes", "Audi", "Ford", "Chevrolet",
        "Honda", "Mazda", "Opel", "Peugeot", "Citroen", "Lexus", "Infiniti",
        "Porsche", "Land Rover", "Volvo", "Subaru", "Suzuki", "Daihatsu",
        "GAZ", "UAZ", "Geely", "Chery", "Haval", "Changan", "Exeed",
        "Gazelle", "KAMAZ", "MAN", "Scania", "DAF", "Volvo Truck",
    ]

    def __init__(self, min_confidence: float = 0.3, enabled: bool = True, model_dir: Optional[str] = None):
        self.min_confidence = min_confidence
        self.enabled = enabled
        self.model_dir = model_dir
        self._model = None
        self._processor = None
        self._texts_encoded = None
        self._load_failed = False

    def _ensure_model(self):
        if self._model is None and self.enabled and not self._load_failed:
            try:
                cache_dir = str(Path(self.model_dir) / "huggingface") if self.model_dir else None
                if cache_dir:
                    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
                from transformers import CLIPModel, CLIPProcessor
                import torch
                model_name = "openai/clip-vit-base-patch32"
                self._model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
                self._processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(device)
                self._model.eval()
                # Pre-encode brand texts once
                with torch.no_grad():
                    texts = [f"a photo of a {b} car" for b in self.BRANDS]
                    inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(device)
                    out = self._model.get_text_features(**inputs)
                    feats = out.pooler_output if hasattr(out, 'pooler_output') else out[0]
                    self._texts_encoded = feats / feats.norm(dim=-1, keepdim=True)
                logger.info(f"VMR: CLIP model loaded ({device}), {len(self.BRANDS)} brands")
            except Exception as e:
                logger.error(f"VMR: failed to load CLIP: {e}")
                self._load_failed = True

    def classify(self, crop: np.ndarray) -> Optional[dict]:
        """Classify vehicle crop → brand. Returns {brand, confidence} or None."""
        self._ensure_model()
        if not self.enabled or self._model is None or crop.size == 0:
            return None

        try:
            import torch
            device = next(self._model.parameters()).device
            from PIL import Image
            img = Image.fromarray(crop[..., ::-1])  # BGR → RGB
            inputs = self._processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self._model.get_image_features(**inputs)
                img_feat = out.pooler_output if hasattr(out, 'pooler_output') else out[0]
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                similarities = (img_feat @ self._texts_encoded.T).squeeze(0)
                top2 = similarities.topk(2)
                best_idx = int(top2.indices[0].item())
                best_score = float(top2.values[0].item())
                second_score = float(top2.values[1].item())

            # Accept if above absolute threshold, or if clearly separated from #2
            if best_score >= self.min_confidence or (best_score >= 0.12 and best_score - second_score >= 0.05):
                return {"brand": self.BRANDS[best_idx], "confidence": round(best_score, 3)}
            return None
        except Exception as e:
            logger.debug(f"VMR classification error: {e}")
            return None
