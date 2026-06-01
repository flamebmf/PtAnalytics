# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import warnings
from typing import Optional

import cv2
import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*SimilarityTransform", category=FutureWarning)


class FaceRecognizer:
    """Face detection + embedding extraction using InsightFace."""

    def __init__(self, min_confidence: float = 0.5, search_threshold: float = 0.6, enabled: bool = True):
        self.min_confidence = min_confidence
        self.search_threshold = search_threshold
        self.enabled = enabled
        self._face_app = None

    def _ensure_model(self):
        if self._face_app is None and self.enabled:
            try:
                import insightface
                logger.info("Loading InsightFace model...")
                self._face_app = insightface.app.FaceAnalysis(
                    name="buffalo_l",   # detection + recognition
                    providers=["CPUExecutionProvider"],
                )
                self._face_app.prepare(ctx_id=-1, det_size=(640, 640))
                logger.info("InsightFace model loaded")
            except Exception as e:
                logger.error(f"Failed to init InsightFace: {e}")
                self.enabled = False

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
          {bbox: (x1,y1,x2,y2), embedding: np.ndarray, confidence: float, face_hash: str}
        """
        if not self.enabled:
            return []

        self._ensure_model()
        if self._face_app is None:
            return []

        try:
            faces = self._face_app.get(frame)
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.normed_embedding
            conf = float(face.det_score) if hasattr(face, "det_score") else 1.0

            if conf < self.min_confidence:
                continue

            face_hash = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]

            results.append({
                "bbox": tuple(bbox),
                "embedding": embedding,
                "confidence": conf,
                "face_hash": face_hash,
            })

        return results

    def match_embedding(self, embedding: np.ndarray, stored_embeddings: list[np.ndarray]) -> Optional[int]:
        """Find closest match by cosine similarity. Returns index or None."""
        if not stored_embeddings:
            return None

        similarities = [np.dot(embedding, se) for se in stored_embeddings]
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= self.search_threshold:
            return best_idx
        return None
