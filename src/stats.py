# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from loguru import logger


@dataclass
class PipelineStats:
    """Per-camera stats counters. Thread-safe via single-writer pattern."""
    camera_id: str = ""
    camera_name: str = ""
    running: bool = False
    started_at: float = 0.0

    frames_captured: int = 0
    frames_skipped: int = 0
    frames_with_motion: int = 0
    frames_processed: int = 0
    detections_total: int = 0
    objects_tracked: int = 0
    objects_stored: int = 0
    capture_fps: float = 0.0
    plates_recognized: int = 0
    faces_detected: int = 0
    events_fired: int = 0
    db_errors: int = 0
    last_frame_at: float = 0.0

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    @property
    def fps_effective(self) -> float:
        """Effective processing FPS (frames with detection)."""
        if self.uptime < 1.0:
            return 0.0
        return self.frames_processed / self.uptime


class StatsCollector:
    """Periodic stats logger + aggregator. Reads stats dict from pipelines."""

    def __init__(self, log_interval: float = 30.0):
        self.log_interval = log_interval
        self._last_log = 0.0
        self._pipelines: dict[str, PipelineStats] = {}

    def register(self, camera_id: str, camera_name: str):
        self._pipelines[camera_id] = PipelineStats(
            camera_id=camera_id,
            camera_name=camera_name,
        )

    def get(self, camera_id: str) -> Optional[PipelineStats]:
        return self._pipelines.get(camera_id)

    def snapshot(self) -> list[dict]:
        return [asdict(s) for s in self._pipelines.values()]

    def summary(self) -> dict:
        total_captured = sum(s.frames_captured for s in self._pipelines.values())
        total_skipped = sum(s.frames_skipped for s in self._pipelines.values())
        total_motion = sum(s.frames_with_motion for s in self._pipelines.values())
        total_frames = sum(s.frames_processed for s in self._pipelines.values())
        total_objects = sum(s.objects_stored for s in self._pipelines.values())
        total_events = sum(s.events_fired for s in self._pipelines.values())
        total_errors = sum(s.db_errors for s in self._pipelines.values())
        return {
            "cameras": len(self._pipelines),
            "cameras_running": sum(1 for s in self._pipelines.values() if s.running),
            "total_frames_captured": total_captured,
            "total_frames_skipped": total_skipped,
            "total_frames_with_motion": total_motion,
            "total_frames_processed": total_frames,
            "total_objects_stored": total_objects,
            "total_events_fired": total_events,
            "total_errors": total_errors,
            "per_camera": [
                {
                    "id": s.camera_id,
                    "name": s.camera_name,
                    "running": s.running,
                    "fps": round(s.fps_effective, 2),
                    "capture_fps": round(s.capture_fps, 2),
                    "uptime_s": round(s.uptime, 1),
                    "captured": s.frames_captured,
                    "skipped": s.frames_skipped,
                    "motion": s.frames_with_motion,
                    "frames": s.frames_processed,
                    "detections": s.detections_total,
                    "objects": s.objects_stored,
                    "events": s.events_fired,
                    "plates": s.plates_recognized,
                    "faces": s.faces_detected,
                }
                for s in self._pipelines.values()
            ],
        }

    def maybe_log(self):
        now = time.time()
        if now - self._last_log >= self.log_interval:
            self._last_log = now
            s = self.summary()
            logger.info(
                f"STATS | cams={s['cameras_running']}/{s['cameras']} "
                f"captured={s['total_frames_captured']} "
                f"motion={s['total_frames_with_motion']} "
                f"processed={s['total_frames_processed']} "
                f"objects={s['total_objects_stored']} "
                f"events={s['total_events_fired']} "
                f"errors={s['total_errors']}"
            )
            for cam in s["per_camera"]:
                logger.info(
                    f"  [{cam['id']}] fps={cam['fps']:.1f} "
                    f"capture_fps={cam['capture_fps']:.1f} "
                    f"captured={cam['captured']} skipped={cam['skipped']} "
                    f"motion={cam['motion']} processed={cam['frames']} "
                    f"det={cam['detections']} objs={cam['objects']} "
                    f"plates={cam['plates']} faces={cam['faces']}"
                )
