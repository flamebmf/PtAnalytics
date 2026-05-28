from typing import Optional

import cv2
import numpy as np
from loguru import logger


class LPRRecognizer:
    """License Plate Recognition using PaddleOCR."""

    def __init__(self, min_confidence: float = 0.6, enabled: bool = True):
        self.min_confidence = min_confidence
        self.enabled = enabled
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is None and self.enabled:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang="en", use_angle_cls=False)
                logger.info("PaddleOCR initialized for LPR")
            except Exception as e:
                logger.error(f"Failed to init PaddleOCR: {e}")
                self.enabled = False

    def recognize(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[str]:
        """Extract license plate text from vehicle ROI."""
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

        height = roi.shape[0]
        if height < 20:  # too small for plate
            return None

        try:
            resized = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None

        try:
            results = self._ocr.ocr(enhanced, cls=False)
            if not results or not results[0]:
                return None

            candidate = ""
            max_conf = 0.0
            for line in results[0]:
                text = line[1][0]
                conf = line[1][1]
                clean = "".join(ch for ch in text if ch.isalnum()).upper()
                if conf > max_conf and len(clean) >= 4:
                    max_conf = conf
                    candidate = clean

            if candidate and max_conf >= self.min_confidence:
                return candidate
        except Exception as e:
            logger.debug(f"OCR error: {e}")

        return None
