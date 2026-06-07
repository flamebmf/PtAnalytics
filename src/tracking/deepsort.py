# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
from typing import Optional

import cv2
import numpy as np
from collections import Counter
from loguru import logger


class KalmanBoxTracker:
    """Kalman filter tracker for a single object bounding box."""
    count = 0

    def __init__(self, bbox: tuple[int, int, int, int], class_id: int = -1, class_name: str = "unknown"):
        x1, y1, x2, y2 = bbox
        self.class_id = class_id
        self.class_name = class_name
        self._class_votes: list[int] = [class_id]
        self._class_names: dict[int, str] = {class_id: class_name}
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], np.float32)

        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h
        r = w / max(h, 1)
        self.kf.statePost = np.array([cx, cy, s, r, 0, 0, 0], np.float32)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.history = []
        self.features: Optional[np.ndarray] = None

    def predict(self) -> np.ndarray:
        """Advance Kalman filter and return predicted bounding box [x1,y1,x2,y2]."""
        if (self.kf.statePost[6] + self.kf.statePost[2]) <= 0:
            self.kf.statePost[6] *= 0.0
        prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox(prediction)

    def update(self, bbox: tuple[int, int, int, int]):
        """Correct Kalman filter with measurement [cx, cy, s, r]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        measurement = np.array([
            x1 + w / 2,
            y1 + h / 2,
            w * h,
            w / max(h, 1),
        ], np.float32)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.hits += 1
        self.history.append(bbox)

    def get_state(self) -> np.ndarray:
        """Get current bounding box [x1,y1,x2,y2]."""
        return self._state_to_bbox(self.kf.statePost)

    @staticmethod
    def _state_to_bbox(state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        cx, cy, s, r = (float(state[i]) for i in range(4))
        s = max(s, 1.0)
        r = max(r, 0.1)
        w = np.sqrt(s * r)
        h = s / w
        return np.array([
            max(0, cx - w / 2),
            max(0, cy - h / 2),
            max(0, cx + w / 2),
            max(0, cy + h / 2),
        ], np.float32)


def iou(bb1: tuple, bb2: tuple) -> float:
    """Intersection over Union of two bounding boxes."""
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)


class DeepSortTracker:
    """Multi-object tracker using Kalman filters + IoU matching."""

    def __init__(self, max_age: int = 30, n_init: int = 3, nn_budget: int = 100, iou_threshold: float = 0.5):
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.iou_threshold = iou_threshold or 0.5
        self.trackers: list[KalmanBoxTracker] = []
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Args:
            detections: list of {bbox, confidence, class_id, class_name}

        Returns:
            list of {bbox, track_id, class_id, class_name, confidence, status: 'confirmed'|'tentative'}
        """
        matched, unmatched_dets, unmatched_trks = [], [], []

        for trk in self.trackers:
            trk.predict()

        if detections:
            det_bboxes = [d["bbox"] for d in detections]
            trk_bboxes = [t.get_state() for t in self.trackers]

            # Build per-class IoU sub-matrices so tracks never match wrong class
            used_dets = set()
            used_trks = set()

            for d_idx, det in enumerate(detections):
                if d_idx in used_dets:
                    continue
                det_class = det["class_id"]
                best_idx = -1
                best_iou = self.iou_threshold
                for t_idx, trk in enumerate(self.trackers):
                    if t_idx in used_trks:
                        continue
                    if trk.class_id != det_class:
                        continue
                    val = iou(det["bbox"], tuple(trk_bboxes[t_idx]))
                    if val >= best_iou:
                        best_iou = val
                        best_idx = t_idx
                if best_idx >= 0:
                    matched.append((d_idx, best_idx))
                    used_dets.add(d_idx)
                    used_trks.add(best_idx)

            unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
            unmatched_trks = [i for i in range(len(self.trackers)) if i not in used_trks]

            # Centroid-distance fallback (same-class only)
            if unmatched_dets and unmatched_trks:
                for d_idx in unmatched_dets[:]:
                    det_class = detections[d_idx]["class_id"]
                    dx1, dy1, dx2, dy2 = det_bboxes[d_idx]
                    dcx = (dx1 + dx2) / 2
                    dcy = (dy1 + dy2) / 2
                    dw = max(dx2 - dx1, 1)
                    dh = max(dy2 - dy1, 1)
                    threshold = max(dw, dh) * 1.5
                    best_t = None
                    best_d = float("inf")
                    for t_idx in unmatched_trks:
                        if self.trackers[t_idx].class_id != det_class:
                            continue
                        tx1, ty1, tx2, ty2 = trk_bboxes[t_idx]
                        tcx = (tx1 + tx2) / 2
                        tcy = (ty1 + ty2) / 2
                        dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                        if dist < best_d:
                            best_d = dist
                            best_t = t_idx
                    if best_d < threshold:
                        matched.append((d_idx, best_t))
                        unmatched_dets.remove(d_idx)
                        unmatched_trks.remove(best_t)
        else:
            unmatched_trks = list(range(len(self.trackers)))

        for d_idx, t_idx in matched:
            trk = self.trackers[t_idx]
            trk.update(detections[d_idx]["bbox"])
            det = detections[d_idx]
            trk._class_votes.append(det["class_id"])
            trk._class_names[det["class_id"]] = det["class_name"]

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            trk = KalmanBoxTracker(det["bbox"], class_id=det["class_id"], class_name=det["class_name"])
            self.trackers.append(trk)

        max_missed = max(2, self.max_age // 5)
        results = []
        remaining = []
        for trk in self.trackers:
            if trk.time_since_update < max_missed:
                is_confirmed = trk.hits >= self.n_init
                conf = 0.0
                for det in detections:
                    if iou(det["bbox"], tuple(trk.get_state())) > self.iou_threshold:
                        conf = det["confidence"]
                        break
                if len(trk._class_votes) > 1:
                    top_two = Counter(trk._class_votes).most_common(2)
                    best_id, best_cnt = top_two[0]
                    second_cnt = top_two[1][1] if len(top_two) > 1 else 0
                    if best_cnt >= second_cnt + 2 and best_id != trk.class_id:
                        trk.class_id = best_id
                        trk.class_name = trk._class_names.get(best_id, trk.class_name)
                results.append({
                    "bbox": tuple(int(v) for v in trk.get_state()),
                    "track_id": trk.id,
                    "class_id": trk.class_id,
                    "class_name": trk.class_name,
                    "confidence": conf,
                    "status": "confirmed" if is_confirmed else "tentative",
                })
            if trk.time_since_update <= self.max_age:
                remaining.append(trk)

        self.trackers = remaining
        if self.trackers:
            track_ids = [t.id for t in self.trackers]
            used = set()
            for i, t in enumerate(self.trackers):
                if t.id in used:
                    self.trackers[i].id = max(track_ids) + i + 1
                used.add(t.id)

        return results
