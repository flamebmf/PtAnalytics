# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

MODEL_URLS = {
    "n": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt",
    "s": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt",
    "m": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1m.pt",
}


class PlateDetector:
    """YOLO-based license plate detector. Runs on vehicle ROIs."""

    def __init__(self, model_path: Optional[str] = None, size: str = "n", conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self._model: Optional[YOLO] = None

        if model_path and Path(model_path).exists():
            try:
                self._model = YOLO(model_path)
                logger.info(f"Plate model loaded: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load plate model {model_path}: {e}")

        if self._model is None:
            cache = Path("/app/models/plates")
            cache.mkdir(parents=True, exist_ok=True)
            local = cache / f"license-plate-v1{size}.pt"
            if local.exists():
                try:
                    self._model = YOLO(str(local))
                    logger.info(f"Plate model loaded from cache: {local}")
                except Exception as e:
                    logger.error(f"Failed to load cached plate model: {e}")

        if self._model is None:
            url = MODEL_URLS.get(size, MODEL_URLS["n"])
            try:
                import urllib.request
                cache = Path("/app/models/plates")
                cache.mkdir(parents=True, exist_ok=True)
                dst = cache / f"license-plate-v1{size}.pt"
                logger.info(f"Downloading plate model from {url}...")
                urllib.request.urlretrieve(url, dst)
                self._model = YOLO(str(dst))
                logger.info(f"Plate model downloaded and loaded: {dst}")
            except Exception as e:
                logger.error(f"Failed to download plate model: {e}")

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def detect(self, roi: np.ndarray) -> list[dict]:
        """Detect plates in a vehicle ROI. Returns list of {bbox, confidence}."""
        if self._model is None or roi.size == 0:
            return []

        h, w = roi.shape[:2]
        if h < 30 or w < 30:
            return []

        try:
            results = self._model(roi, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            logger.debug(f"Plate inference error: {e}")
            return []

        plates = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    plates.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(conf),
                    })
        return plates
