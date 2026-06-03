# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from loguru import logger


class CropClassifier:
    """Classifies individual crop ROIs using a fine-tuned YOLO model.

    Loads fine-tuned.pt (trained on cropped objects) and runs inference on
    small image patches. Returns the predicted class name and confidence.
    Used as a secondary classifier alongside the main COCO detector.
    """

    def __init__(self, model_path: str, confidence: float = 0.25, imgsz: int = 640):
        self.confidence = confidence
        self.imgsz = imgsz
        if not os.path.isfile(model_path):
            logger.warning(f"CropClassifier: model not found at {model_path}, disabled")
            self.model = None
            return
        self.model = YOLO(model_path)
        self.names = self.model.names
        logger.info(f"CropClassifier: loaded {os.path.basename(model_path)} with {len(self.names)} classes: {list(self.names.values())}")

    def classify(self, crop: np.ndarray) -> tuple[str | None, float]:
        """Run inference on a crop. Returns (class_name, confidence) or (None, 0)."""
        if self.model is None or crop is None or crop.size == 0:
            return None, 0.0
        results = self.model(
            crop,
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
            max_det=1,
        )
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best.cls[0])
            conf = float(best.conf[0])
            return self.names[cls_id], conf
        return None, 0.0
