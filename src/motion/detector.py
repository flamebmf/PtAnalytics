import cv2
import numpy as np
from enum import Enum
from loguru import logger


class MotionMethod(str, Enum):
    MOG2 = "mog2"
    FRAME_DIFF = "frame_diff"


class MotionDetector:
    """Detects significant motion between frames to skip static scenes."""

    def __init__(
        self,
        method: MotionMethod = MotionMethod.MOG2,
        threshold: float = 0.15,
        resize_to: tuple[int, int] | None = (640, 360),
    ):
        self.method = method
        self.threshold = threshold
        self.resize_to = resize_to

        if method == MotionMethod.MOG2:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=36, detectShadows=False
            )
        else:
            self._bg_subtractor = None
            self._prev_frame: np.ndarray | None = None

    def has_motion(self, frame: np.ndarray) -> bool:
        """Returns True if significant motion detected."""
        work = frame
        if self.resize_to:
            work = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_NEAREST)

        if self.method == MotionMethod.MOG2:
            fg_mask = self._bg_subtractor.apply(work, learningRate=0.01)
        else:
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self._prev_frame is None:
                self._prev_frame = gray
                return True
            delta = cv2.absdiff(self._prev_frame, gray)
            _, fg_mask = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
            self._prev_frame = gray

        motion_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        return motion_ratio > self.threshold

    def reset(self):
        if self.method == MotionMethod.MOG2:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=36, detectShadows=False
            )
        else:
            self._prev_frame = None
