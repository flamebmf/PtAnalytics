from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from .plate_detector import PlateDetector


class LPRRecognizer:
    """License Plate Recognition: plate detection + PaddleOCR."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        enabled: bool = True,
        plate_model: Optional[str] = None,
        plate_size: str = "n",
    ):
        self.min_confidence = min_confidence
        self.enabled = enabled
        self._ocr = None
        self._plate_detector = PlateDetector(
            model_path=plate_model,
            size=plate_size,
            conf_threshold=0.35,
        )

    def _ensure_ocr(self):
        if self._ocr is None and self.enabled:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang="en", use_angle_cls=False)
                logger.info("PaddleOCR initialized for LPR")
            except Exception as e:
                logger.error(f"Failed to init PaddleOCR: {e}")
                self.enabled = False

    def _ocr_plate(self, plate_crop: np.ndarray) -> Optional[str]:
        """Run OCR on a tight plate crop. Returns cleaned plate text or None."""
        try:
            h, w = plate_crop.shape[:2]
            if h < 15 or w < 30:
                return None

            # Upscale for better OCR
            scale = max(1.0, 80 / h)
            if scale > 1.5:
                plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            results = self._ocr.ocr(rgb, cls=False)
            if not results or not results[0]:
                return None

            best_text = ""
            best_conf = 0.0
            for line in results[0]:
                text = line[1][0]
                conf = line[1][1]
                clean = "".join(ch for ch in text if ch.isalnum()).upper()
                if conf > best_conf and len(clean) >= 4:
                    best_conf = conf
                    best_text = clean

            if best_text and best_conf >= self.min_confidence:
                return best_text
        except Exception as e:
            logger.debug(f"OCR error: {e}")
        return None

    def recognize(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[str]:
        """Detect plate in vehicle bbox and run OCR. Returns plate text or None."""
        if not self.enabled:
            return None

        self._ensure_ocr()
        if self._ocr is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Step 1: find plate regions
        candidates = self._plate_detector.detect(roi)

        # Fallback: bottom-center heuristic if no YOLO detections
        if not candidates:
            candidates = self._heuristic_plate_candidates(roi)

        # Step 2: OCR each candidate
        for cand in candidates:
            px1, py1, px2, py2 = cand["bbox"]
            plate_crop = roi[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            text = self._ocr_plate(plate_crop)
            if text:
                logger.info(f"Plate OCR: {text} (conf={cand['confidence']:.2f})")
                return text

        return None

    @staticmethod
    def _heuristic_plate_candidates(roi: np.ndarray) -> list[dict]:
        """Contour-based plate detection as fallback."""
        h, w = roi.shape[:2]
        # Focus on bottom 60%
        y_start = int(h * 0.35)
        bottom = roi[y_start:, :]
        if bottom.size == 0:
            return []

        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blur, 50, 200)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, cy, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            area_ratio = (cw * ch) / (w * (h - y_start))
            if 1.5 < aspect < 6.0 and area_ratio > 0.01 and cw > 15 and ch > 8:
                candidates.append({
                    "bbox": (x, y_start + cy, x + cw, y_start + cy + ch),
                    "confidence": 0.5,
                })

        return candidates
