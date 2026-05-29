import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from sqlalchemy import select, update

from .db import get_session
from .models import Camera, TrackedObject, FrameCapture, CropSample, Event


class StorageRepository:
    """Async persistence layer for tracked objects, frames, and events."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _db_timestamp(ts: Optional[datetime] = None) -> datetime:
        """Return UTC timestamp compatible with TIMESTAMP WITHOUT TIME ZONE columns."""
        ts = ts or datetime.now(timezone.utc)
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    def save_snapshot(self, camera_id: str, frame: np.ndarray) -> str:
        """Save a full-frame snapshot for camera verification."""
        filename = f"snapshot_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.data_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info(f"Snapshot saved: {filepath}")
        return str(filepath)

    async def ensure_camera(self, camera_id: str, name: str, rtsp_url: str, fps: int) -> Camera:
        async with await get_session() as session:
            result = await session.execute(
                select(Camera).where(Camera.id == camera_id)
            )
            cam = result.scalar_one_or_none()
            if cam is None:
                cam = Camera(id=camera_id, name=name, rtsp_url=rtsp_url, fps=fps)
                session.add(cam)
                await session.commit()
                await session.refresh(cam)
            return cam

    async def get_or_create_object(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        embedding: Optional[list[float]] = None,
        plate_number: Optional[str] = None,
        face_hash: Optional[str] = None,
        face_id: Optional[str] = None,
    ) -> TrackedObject:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(
                    TrackedObject.camera_id == camera_id,
                    TrackedObject.track_id == track_id,
                )
            )
            obj = result.scalar_one_or_none()

            if obj:
                obj.last_seen = self._db_timestamp()
                obj.class_name = class_name
                obj.appearance_count = (obj.appearance_count or 0) + 1
                if plate_number:
                    obj.plate_number = plate_number
                if face_hash:
                    obj.face_hash = face_hash
                if face_id:
                    obj.face_id = face_id
                if embedding is not None:
                    obj.embedding = embedding
                await session.commit()
                await session.refresh(obj)
            else:
                obj = TrackedObject(
                    camera_id=camera_id,
                    track_id=track_id,
                    class_name=class_name,
                    embedding=embedding,
                    plate_number=plate_number,
                    face_hash=face_hash,
                    face_id=face_id,
                )
                session.add(obj)
                await session.commit()
                await session.refresh(obj)

            return obj

    async def save_frame(
        self,
        object_id: uuid.UUID,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> FrameCapture:
        x1, y1, x2, y2 = bbox
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        ts = self._db_timestamp(timestamp)
        filename = f"{object_id.hex}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        camera_dir = self.data_dir
        camera_dir.mkdir(parents=True, exist_ok=True)
        filepath = camera_dir / filename
        cv2.imwrite(str(filepath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        async with await get_session() as session:
            fc = FrameCapture(
                object_id=object_id,
                image_path=str(filepath),
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                confidence=confidence,
                timestamp=ts,
            )
            session.add(fc)
            await session.commit()
            await session.refresh(fc)
            return fc

    async def save_crop(
        self,
        camera_id: str,
        class_name: str,
        class_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        phase: str = "entry",
    ) -> Optional[str]:
        """Save clean crop + YOLO label for dataset collection.
        Returns relative path if saved, None if skipped (dedup or empty crop).
        """
        if await self._check_crop_dedup(camera_id, class_name, bbox):
            return None

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid.uuid4().hex[:8]
        img_name = f"{ts}_{suffix}.jpg"
        rel_path = f"{class_name}/{camera_id}/{img_name}"
        full_path = self.data_dir.parent / "crops" / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        label_path = full_path.with_suffix(".txt")
        label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")

        await self._record_crop(camera_id, class_name, bbox, rel_path, phase)
        logger.info(f"[{camera_id}] Crop saved: {rel_path} ({phase})")
        return rel_path

    async def _check_crop_dedup(self, camera_id: str, class_name: str, bbox: tuple) -> bool:
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
        async with await get_session() as session:
            rows = await session.execute(
                select(CropSample).where(
                    CropSample.camera_id == camera_id,
                    CropSample.class_name == class_name,
                    CropSample.timestamp >= cutoff,
                )
            )
            for row in rows.scalars():
                obb = (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2)
                if self._bbox_iou(bbox, obb) > 0.5:
                    return True
        return False

    async def _record_crop(
        self, camera_id: str, class_name: str, bbox: tuple,
        image_path: str, phase: str,
    ):
        async with await get_session() as session:
            session.add(CropSample(
                camera_id=camera_id,
                class_name=class_name,
                bbox_x1=bbox[0], bbox_y1=bbox[1],
                bbox_x2=bbox[2], bbox_y2=bbox[3],
                image_path=image_path,
                phase=phase,
            ))
            await session.commit()

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (a_area + b_area - inter + 1e-6)

    async def log_event(
        self,
        object_id: uuid.UUID,
        event_type: str,
        trigger_name: Optional[str] = None,
        action_result: Optional[dict] = None,
    ) -> Event:
        async with await get_session() as session:
            evt = Event(
                object_id=object_id,
                event_type=event_type,
                trigger_name=trigger_name,
                action_result=action_result,
            )
            session.add(evt)
            await session.commit()
            await session.refresh(evt)
            return evt

    async def search_by_plate(self, plate_number: str) -> Optional[TrackedObject]:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(TrackedObject.plate_number == plate_number)
            )
            return result.scalar_one_or_none()

    async def search_similar_face(
        self, embedding: list[float], threshold: float = 0.6
    ) -> Optional[TrackedObject]:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject)
                .where(TrackedObject.embedding.isnot(None))
                .order_by(TrackedObject.embedding.cosine_distance(embedding))
                .limit(1)
            )
            obj = result.scalar_one_or_none()
            if obj and obj.embedding is not None:
                emb_arr = np.array(obj.embedding, dtype=np.float32)
                query_arr = np.array(embedding, dtype=np.float32)
                similarity = float(np.dot(emb_arr, query_arr))
                if similarity >= threshold:
                    return obj
            return None

    async def get_frame_count(self, object_id: uuid.UUID) -> int:
        async with await get_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(func.count()).where(FrameCapture.object_id == object_id)
            )
            return result.scalar() or 0

    async def list_objects(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        name: Optional[str] = None,
        show_ignored: bool = False,
        limit: int = 50,
        offset: int = 0,
        sort: str = "-last_seen",
    ) -> list[TrackedObject]:
        async with await get_session() as session:
            from sqlalchemy import asc, desc
            order = desc if sort.startswith("-") else asc
            col_name = sort.lstrip("-")
            col = getattr(TrackedObject, col_name, TrackedObject.last_seen)
            query = select(TrackedObject).order_by(order(col))
            if camera_id:
                query = query.where(TrackedObject.camera_id == camera_id)
            if class_name:
                query = query.where(TrackedObject.class_name == class_name)
            if name:
                query = query.where(TrackedObject.name.ilike(f"%{name}%"))
            if not show_ignored:
                query = query.where(TrackedObject.ignored != True)
            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_object_count(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        show_ignored: bool = False,
    ) -> int:
        async with await get_session() as session:
            from sqlalchemy import func
            query = select(func.count()).select_from(TrackedObject)
            if camera_id:
                query = query.where(TrackedObject.camera_id == camera_id)
            if class_name:
                query = query.where(TrackedObject.class_name == class_name)
            if not show_ignored:
                query = query.where(TrackedObject.ignored != True)
            result = await session.execute(query)
            return result.scalar() or 0

    async def delete_frame(self, frame_id: uuid.UUID) -> bool:
        async with await get_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(FrameCapture).where(FrameCapture.id == frame_id)
            )
            fc = result.scalar_one_or_none()
            if fc is None:
                return False
            obj_id = fc.object_id
            try:
                if fc.image_path and Path(fc.image_path).exists():
                    Path(fc.image_path).unlink()
            except Exception:
                pass
            await session.delete(fc)
            # Update parent object's last_seen to max remaining frame timestamp
            ts_result = await session.execute(
                select(func.max(FrameCapture.timestamp)).where(FrameCapture.object_id == obj_id)
            )
            max_ts = ts_result.scalar()
            if max_ts is not None:
                await session.execute(
                    update(TrackedObject)
                    .where(TrackedObject.id == obj_id)
                    .values(last_seen=max_ts)
                )
            await session.commit()
            return True

    async def list_frames(
        self,
        object_id: uuid.UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[FrameCapture]:
        async with await get_session() as session:
            query = (
                select(FrameCapture)
                .where(FrameCapture.object_id == object_id)
                .order_by(FrameCapture.timestamp.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_object_name(self, object_id: uuid.UUID, name: Optional[str]) -> bool:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(TrackedObject.id == object_id)
            )
            obj = result.scalar_one_or_none()
            if obj is None:
                return False
            obj.name = name
            await session.commit()
            return True

    async def delete_object(self, object_id: uuid.UUID) -> bool:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(TrackedObject.id == object_id)
            )
            obj = result.scalar_one_or_none()
            if obj is None:
                return False
            await session.execute(
                Event.__table__.delete().where(Event.object_id == object_id)
            )
            await session.delete(obj)
            await session.commit()
            return True

    async def find_similar_objects(
        self,
        embedding: list[float],
        class_name: str,
        exclude_object_id: uuid.UUID | None = None,
        threshold: float = 0.85,
        max_age_seconds: float = 300,
        limit: int = 5,
    ) -> list[tuple[TrackedObject, float]]:
        from sqlalchemy import func as sqfunc
        from datetime import datetime, timedelta
        vec = np.array(embedding, dtype=np.float32)
        cutoff = self._db_timestamp() - timedelta(seconds=max_age_seconds)
        async with await get_session() as session:
            stmt = (
                select(
                    TrackedObject,
                    TrackedObject.embedding.cosine_distance(vec).label("dist")
                )
                .where(TrackedObject.embedding.isnot(None))
                .where(TrackedObject.class_name == class_name)
                .where(TrackedObject.last_seen >= cutoff)
            )
            if exclude_object_id:
                stmt = stmt.where(TrackedObject.id != exclude_object_id)
            stmt = stmt.order_by("dist").limit(limit)
            result = await session.execute(stmt)
            rows = result.all()
        return [(r[0], 1 - r[1]) for r in rows if (1 - r[1]) >= threshold]

    async def update_embedding(self, object_id: uuid.UUID, embedding: list[float]) -> None:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(TrackedObject.id == object_id)
            )
            obj = result.scalar_one_or_none()
            if obj:
                obj.embedding = embedding
                await session.commit()

    async def list_object_names(self) -> list[dict]:
        async with await get_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(
                    TrackedObject.name,
                    func.count().label("count"),
                    func.array_agg(func.distinct(TrackedObject.camera_id)).label("cameras"),
                )
                .where(TrackedObject.name.isnot(None))
                .where(TrackedObject.name != "")
                .group_by(TrackedObject.name)
                .order_by(func.max(TrackedObject.last_seen).desc())
            )
            return [
                {
                    "name": row[0],
                    "count": row[1],
                    "cameras": row[2] if row[2] else [],
                }
                for row in result.all()
            ]
