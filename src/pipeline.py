import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from .capture import StreamReader
from .motion import MotionDetector, MotionMethod
from .detection import YoloDetector
from .tracking import DeepSortTracker
from .recognition import LPRRecognizer, FaceRecognizer
from .recognition.reid import compute_embedding
from .storage import StorageRepository
from .storage.models import TrackedObject
from .actions import ActionDispatcher
from .stats import PipelineStats


class CameraPipeline:
    """Orchestrates capture→motion→detect→track→recognize→store→act for one camera."""

    def __init__(
        self,
        camera_config: dict,
        settings: dict,
        repository: StorageRepository,
        dispatcher: ActionDispatcher,
    ):
        self.cam_id = camera_config["id"]
        self.cam_name = camera_config["name"]
        self.enabled = camera_config.get("enabled", True)
        self.debug = camera_config.get("debug", settings.get("app", {}).get("debug", False))
        self.roi = camera_config.get("roi", [])

        # Capture
        rtsp_url = os.path.expandvars(camera_config["rtsp_url"])
        fps = camera_config.get("fps", settings.get("app", {}).get("fps", 10))
        self.reader = StreamReader(rtsp_url=rtsp_url, target_fps=fps, debug=self.debug)

        # Motion
        mot_cfg = settings.get("motion", {})
        self.motion_detector = MotionDetector(
            method=MotionMethod(mot_cfg.get("method", "mog2")),
            threshold=camera_config.get("motion_threshold", mot_cfg.get("threshold", 0.15)),
            resize_to=tuple(mot_cfg.get("resize_to", [640, 360])),
        )
        self.motion_skip = camera_config.get("motion_skip_seconds", mot_cfg.get("skip_seconds", 1.0))
        self.motion_enabled = camera_config.get("motion_enabled", mot_cfg.get("enabled", True))

        # Detector (camera config overrides global settings)
        det_cfg = settings.get("detector", {})
        cam_det = camera_config.get("detector", {})
        self.detector = YoloDetector(
            model_path=det_cfg.get("model", "yolo11n.pt"),
            device=det_cfg.get("device", "cpu"),
            confidence=cam_det.get("confidence", det_cfg.get("confidence", 0.4)),
            iou=cam_det.get("iou", det_cfg.get("iou", 0.45)),
            classes=cam_det.get("classes", det_cfg.get("classes")),
            imgsz=cam_det.get("imgsz", det_cfg.get("imgsz", 640)),
            workers=cam_det.get("workers", det_cfg.get("workers")),
            backend=cam_det.get("backend", det_cfg.get("backend", "torch")),
            min_bbox_size=cam_det.get("min_bbox_size", det_cfg.get("min_bbox_size", 20)),
        )

        # Tracker
        trk_cfg = settings.get("tracker", {})
        self.tracker = DeepSortTracker(
            max_age=trk_cfg.get("max_age", 30),
            n_init=trk_cfg.get("n_init", 3),
            nn_budget=trk_cfg.get("nn_budget", 100),
        )

        # Recognizers
        lpr_cfg = settings.get("lpr", {})
        self.lpr = LPRRecognizer(
            min_confidence=lpr_cfg.get("min_confidence", 0.6),
            enabled=lpr_cfg.get("enabled", True),
        )

        face_cfg = settings.get("face", {})
        self.face_recognizer = FaceRecognizer(
            min_confidence=face_cfg.get("min_confidence", 0.5),
            search_threshold=face_cfg.get("search_threshold", 0.6),
            enabled=face_cfg.get("enabled", True),
        )

        self.repo = repository
        self.dispatcher = dispatcher
        self.stats = PipelineStats(camera_id=self.cam_id, camera_name=self.cam_name)
        self._stop = False
        self._last_processed = 0.0
        self._active_tracks: dict[int, dict] = {}
        self._last_obj_by_track: dict[int, TrackedObject] = {}
        self.track_depart_timeout = settings.get("tracker", {}).get("depart_timeout", 3.0)

        # Dataset collection (3 crops per track: entry, mid, exit)
        ds_cfg = settings.get("dataset", {})
        self.crop_enabled = ds_cfg.get("crop_enabled", True)
        self._last_frame: dict[int, np.ndarray] = {}
        self._last_bbox: dict[int, tuple] = {}
        self._last_class_name: dict[int, str] = {}
        self._last_class_id: dict[int, int] = {}
        self._crop_phase: dict[int, int] = {}  # 0=none, 1=entry, 2=mid, 3=exit, 4=done

    async def run(self):
        """Main loop."""
        if not self.enabled:
            logger.info(f"Camera {self.cam_name} disabled, skipping")
            return

        await self.repo.ensure_camera(
            camera_id=self.cam_id,
            name=self.cam_name,
            rtsp_url=self.reader.rtsp_url,
            fps=self.reader.target_fps,
        )
        logger.info(f"Pipeline {self.cam_name}: ensure_camera completed")

        logger.info(f"Pipeline {self.cam_name}: starting reader")
        await self.reader.start_async()
        logger.info(f"Pipeline {self.cam_name}: reader started")

        self.stats.running = True
        self.stats.started_at = time()
        logger.info(f"Pipeline {self.cam_name} started")

        # Save first-frame snapshot for verification
        loop = asyncio.get_running_loop()
        snapshot_frame = await loop.run_in_executor(None, self.reader.read, 5.0)
        if snapshot_frame is not None:
            await loop.run_in_executor(
                None, self.repo.save_snapshot, self.cam_id, snapshot_frame
            )
            h, w = snapshot_frame.shape[:2]
            logger.info(f"Pipeline {self.cam_name}: snapshot saved ({w}x{h})")
        else:
            logger.warning(f"Pipeline {self.cam_name}: no frame within 5s, snapshot skipped")

        empty_count = 0
        try:
            while not self._stop:
                _t_read = time()
                frame: Optional[np.ndarray] = await loop.run_in_executor(
                    None, self.reader.read, 1.0
                )
                _t_read = time() - _t_read
                if self.debug and frame is not None and _t_read > 0.01:
                    logger.info(f"[TIMING {self.cam_name}] read() took {_t_read:.3f}s")
                if frame is None:
                    empty_count += 1
                    if empty_count % 30 == 0:
                        logger.warning(
                            f"Pipeline {self.cam_name}: no frames for {empty_count} iterations"
                        )
                    await asyncio.sleep(0.1)
                    continue
                empty_count = 0

                skip = await loop.run_in_executor(None, self._should_skip)
                if skip:
                    continue

                if self.roi:
                    frame = await loop.run_in_executor(
                        None, self._apply_roi, frame, self.roi
                    )

                self.stats.frames_captured += 1
                self.stats.capture_fps = self.reader.fps

                _t = time()
                motion_detected = True
                if self.motion_enabled:
                    motion_detected = await loop.run_in_executor(
                        None, self.motion_detector.has_motion, frame
                    )
                if not motion_detected:
                    self.stats.frames_skipped += 1
                    continue
                _t_mot = time() - _t

                self.stats.frames_with_motion += 1
                self._last_processed = time()

                _t2 = time()
                detections = await loop.run_in_executor(
                    None, self.detector.detect, frame
                )
                _t_det = time() - _t2
                self.stats.frames_processed += 1
                self.stats.last_frame_at = time()
                detections = self._filter_persons_in_vehicles(detections)
                if not detections:
                    if self.debug:
                        logger.info(f"[TIMING {self.cam_name}] mot={_t_mot:.2f}s det={_t_det:.2f}s post=0s")
                    continue

                self.stats.detections_total += len(detections)

                _t3 = time()
                tracks = await loop.run_in_executor(
                    None, self.tracker.update, detections
                )

                current_ids = {t["track_id"] for t in tracks}

                # Store last-known state for crop saving (entry/mid/exit)
                for track in tracks:
                    tid = track["track_id"]
                    self._last_frame[tid] = frame
                    self._last_bbox[tid] = track["bbox"]
                    self._last_class_name[tid] = track["class_name"]
                    self._last_class_id[tid] = track["class_id"]
                    if tid not in self._crop_phase:
                        self._crop_phase[tid] = 0

                # Process only NEW tracks (appeared or reappeared)
                for track in tracks:
                    tid = track["track_id"]
                    if tid not in self._active_tracks:
                        is_reappeared = tid in self._last_obj_by_track
                        self._active_tracks[tid] = {
                            "class_name": track["class_name"],
                            "first_seen": time(),
                            "missing_since": 0.0,
                        }
                        await self._process_track(
                            frame, track,
                            reappeared=is_reappeared,
                        )
                    else:
                        self._active_tracks[tid]["missing_since"] = 0.0

                # Mid-crop: save when track has been alive ~50% of depart timeout
                if self.crop_enabled:
                    for tid, info in list(self._active_tracks.items()):
                        if self._crop_phase.get(tid, 0) == 1:
                            elapsed = time() - info["first_seen"]
                            if elapsed >= self.track_depart_timeout * 0.5:
                                bbox = self._last_bbox.get(tid)
                                cn = self._last_class_name.get(tid)
                                ci = self._last_class_id.get(tid)
                                lf = self._last_frame.get(tid)
                                if bbox and cn is not None and ci is not None and lf is not None:
                                    await self.repo.save_crop(
                                        self.cam_id, cn, ci, lf, bbox, phase="mid",
                                    )
                                self._crop_phase[tid] = 2

                # Detect departed tracks (disappeared)
                now = time()
                _t_post = now - _t3
                _t_post = now - _t3
                for tid in list(self._active_tracks.keys()):
                    if tid not in current_ids:
                        td = self._active_tracks[tid]
                        if td["missing_since"] == 0.0:
                            td["missing_since"] = now
                        elif now - td["missing_since"] >= self.track_depart_timeout:
                            # Exit crop before cleanup
                            if self.crop_enabled and self._crop_phase.get(tid, 0) in (1, 2):
                                bbox = self._last_bbox.get(tid)
                                cn = self._last_class_name.get(tid)
                                ci = self._last_class_id.get(tid)
                                lf = self._last_frame.get(tid)
                                if bbox and cn is not None and ci is not None and lf is not None:
                                    await self.repo.save_crop(
                                        self.cam_id, cn, ci, lf, bbox, phase="exit",
                                    )
                                self._crop_phase[tid] = 4
                            self._active_tracks.pop(tid)
                            logger.info(
                                f"[{self.cam_name}] Track {tid} ({td['class_name']}) "
                                f"departed after {now - td['first_seen']:.0f}s"
                            )
                            obj = self._last_obj_by_track.pop(tid, None)
                            if obj is not None:
                                try:
                                    await self.dispatcher.evaluate(
                                        obj=obj,
                                        plate_number=None,
                                        face_id=None,
                                        metadata={
                                            "track_id": tid,
                                            "class_name": td["class_name"],
                                            "event": "departed",
                                            "duration_s": round(now - td["first_seen"], 1),
                                        },
                                    )
                                except Exception as e:
                                    logger.error(f"Departure dispatch error: {e}")

                self.stats.objects_tracked = len(self._active_tracks)
                if self.debug:
                    logger.info(
                        f"[TIMING {self.cam_name}] mot={_t_mot:.2f}s "
                        f"det={_t_det:.2f}s track+post={_t_post:.2f}s "
                        f"total={_t_mot+_t_det+_t_post:.2f}s"
                    )

        except Exception as e:
            logger.exception(f"Pipeline {self.cam_name} error: {e}")
        finally:
            self.reader.stop()

    async def _process_track(self, frame: np.ndarray, track: dict, reappeared: bool = False):
        """Process one tracked object: recognize → store → dispatch."""
        bbox = track["bbox"]
        class_name = track["class_name"]
        confidence = track["confidence"]
        track_id = track["track_id"]

        plate_number: Optional[str] = None
        face_id: Optional[str] = None
        face_hash: Optional[str] = None
        embedding: Optional[list[float]] = None

        vehicle_classes = {"car", "truck", "bus", "motorcycle"}

        if class_name in vehicle_classes and self.lpr.enabled:
            plate_number = await self.lpr.recognize(frame, bbox)
            if plate_number:
                logger.info(f"[{self.cam_name}] Plate detected: {plate_number}")
                self.stats.plates_recognized += 1

        if class_name == "person" and self.face_recognizer.enabled:
            faces = self.face_recognizer.detect_faces(frame)
            self.stats.faces_detected += len(faces)
            for face in faces:
                if self._bbox_overlap(face["bbox"], bbox) > 0.3:
                    face_embedding = face["embedding"]
                    embedding = face_embedding.tolist()
                    face_hash = face["face_hash"]

                    existing_obj = await self.repo.search_similar_face(
                        embedding,
                        threshold=self.face_recognizer.search_threshold,
                    )
                    if existing_obj:
                        face_id = str(existing_obj.id)
                        logger.info(f"[{self.cam_name}] Face matched: {existing_obj.id}")
                    else:
                        face_id = face_hash
                    break

        try:
            obj = await self.repo.get_or_create_object(
                camera_id=self.cam_id,
                track_id=track_id,
                class_name=class_name,
                timestamp=datetime.now(timezone.utc),
                embedding=embedding,
                plate_number=plate_number,
                face_hash=face_hash,
                face_id=face_id,
            )
            self.stats.objects_stored += 1
            self._last_obj_by_track[track_id] = obj
        except Exception as e:
            logger.error(f"DB error saving object: {e}")
            self.stats.db_errors += 1
            return

        # Ignored objects: update last_seen but skip frames and triggers
        if obj.ignored:
            return

        try:
            await self.repo.save_frame(
                object_id=obj.id,
                frame=frame,
                bbox=bbox,
                confidence=confidence,
            )
        except Exception as e:
            logger.error(f"DB error saving frame: {e}")

        # Vehicle ReID: auto-link unnamed vehicles across cameras
        if not obj.name and class_name in ("car", "truck", "bus", "motorcycle"):
            try:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.debug(f"[{self.cam_name}] ReID skip: empty crop track {track_id}")
                else:
                    veid = compute_embedding(crop)
                    embed_norm = np.linalg.norm(veid)
                    logger.info(f"[{self.cam_name}] ReID: track {track_id} {class_name} "
                                f"embedding dim={len(veid)} norm={embed_norm:.4f}")
                    await self.repo.update_embedding(obj.id, veid)
                    matches = await self.repo.find_similar_objects(
                        veid, class_name, exclude_object_id=obj.id,
                    )
                    logger.info(f"[{self.cam_name}] ReID: found {len(matches)} matches for track {track_id}")
                    for m, s in matches:
                        logger.info(f"[{self.cam_name}] ReID candidate: {m.id} '{m.name}' "
                                    f"cam={m.camera_id} score={s:.4f}")
                    if matches:
                        best, score = matches[0]
                        name = best.name or f"vehicle-{best.id.hex[:8]}"
                        if not best.name:
                            await self.repo.update_object_name(best.id, name)
                        obj.name = name
                        await self.repo.update_object_name(obj.id, name)
                        logger.info(f"[{self.cam_name}] ReID: track {track_id} → '{name}' ({score:.3f})")
            except Exception as e:
                logger.error(f"ReID error: {e}")

        # Entry crop for dataset collection
        if self.crop_enabled:
            try:
                await self.repo.save_crop(
                    self.cam_id, class_name, track["class_id"],
                    frame, bbox, phase="entry",
                )
                self._crop_phase[track_id] = 1
            except Exception as e:
                logger.error(f"Crop error: {e}")

        try:
            meta = {
                "track_id": track_id,
                "class_name": class_name,
                "confidence": confidence,
                "plate_number": plate_number,
                "face_id": face_id,
            }
            if reappeared:
                meta["event"] = "reappeared"
            action_results = await self.dispatcher.evaluate(
                obj=obj,
                plate_number=plate_number,
                face_id=face_id,
                metadata=meta,
            )

            for ar in action_results:
                await self.repo.log_event(
                    object_id=obj.id,
                    event_type="action_triggered",
                    trigger_name=ar.get("trigger"),
                    action_result=ar.get("result"),
                )
                self.stats.events_fired += 1
        except Exception as e:
            logger.error(f"Action dispatch error: {e}")

    def mark_track_ignored(self, camera_id: str, track_id: int):
        """Mark an active track as ignored so it stops saving frames and firing triggers."""
        if camera_id != self.cam_id:
            return
        if track_id in self._last_obj_by_track:
            self._last_obj_by_track[track_id].ignored = True
            logger.info(f"[{self.cam_name}] Track {track_id} marked ignored in memory")

    VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    @staticmethod
    def _filter_persons_in_vehicles(detections: list) -> list:
        vehicles = [d for d in detections if d["class_id"] in CameraPipeline.VEHICLE_IDS]
        if not vehicles:
            return detections
        result = []
        for d in detections:
            if d["class_id"] != 0:  # not person
                result.append(d)
                continue
            inside = False
            for v in vehicles:
                if CameraPipeline._bbox_overlap(d["bbox"], v["bbox"]) > 0.5:
                    inside = True
                    break
            if not inside:
                result.append(d)
        return result

    def _should_skip(self) -> bool:
        return (time() - self._last_processed) < self.motion_skip

    @staticmethod
    def _apply_roi(frame: np.ndarray, roi: list) -> np.ndarray:
        if not roi:
            return frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    @staticmethod
    def _bbox_overlap(b1: tuple, b2: tuple) -> float:
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return intersection / min(area1, area2) if min(area1, area2) > 0 else 0.0

    def stop(self):
        self._stop = True
        self.reader.stop()
