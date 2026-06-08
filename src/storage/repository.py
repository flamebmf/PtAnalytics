# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from sqlalchemy import select, update, func, text

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
        timestamp: Optional[datetime] = None,
        vmr_brand: Optional[str] = None,
    ) -> TrackedObject:
        async with await get_session() as session:
            ts = self._db_timestamp(timestamp)
            metadata = {"vmr": vmr_brand} if vmr_brand else None
            obj = TrackedObject(
                camera_id=camera_id,
                track_id=track_id,
                class_name=class_name,
                first_seen=ts,
                last_seen=ts,
                metadata_=metadata,
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
        plate_number: Optional[str] = None,
        face_id: Optional[str] = None,
        vmr_brand: Optional[str] = None,
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
                plate_number=plate_number,
                face_id=face_id,
                vmr_brand=vmr_brand,
                timestamp=ts,
            )
            session.add(fc)
            await session.execute(
                update(TrackedObject)
                .where(TrackedObject.id == object_id)
                .values(last_seen=ts)
            )
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
        min_crop_size: int = 0,
    ) -> Optional[str]:
        """Save clean crop + YOLO label for dataset collection.
        Returns relative path if saved, None if skipped (empty crop, too small).
        When min_crop_size > 0, crops smaller than that are skipped unless they
        touch a frame edge — in which case the bbox is expanded up to 200%.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        frame_h, frame_w = frame.shape[:2]

        if min_crop_size > 0 and (w < min_crop_size or h < min_crop_size):
            touches_edge = x1 <= 0 or y1 <= 0 or x2 >= frame_w or y2 >= frame_h
            if touches_edge:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                nw = max(int(w * 2), min_crop_size)
                nh = max(int(h * 2), min_crop_size)
                x1 = max(0, int(cx - nw / 2))
                y1 = max(0, int(cy - nh / 2))
                x2 = min(frame_w, int(cx + nw / 2))
                y2 = min(frame_h, int(cy + nh / 2))
                if x2 - x1 < min_crop_size or y2 - y1 < min_crop_size:
                    return None
            else:
                return None

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
                .where(TrackedObject.class_name == "person")
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

    async def list_grouped(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        show_ignored: bool = False,
        sort: str = "-last_seen",
    ) -> dict:
        async with await get_session() as session:
            from sqlalchemy import asc, desc, func as sqfunc
            # Named groups
            named = select(
                TrackedObject.name,
                sqfunc.count(TrackedObject.id).label("count"),
                sqfunc.array_agg(TrackedObject.id).label("obj_ids"),
                sqfunc.array_agg(TrackedObject.camera_id).label("cameras"),
                sqfunc.array_agg(TrackedObject.class_name).label("classes"),
                sqfunc.max(TrackedObject.last_seen).label("last_seen"),
                sqfunc.bool_or(TrackedObject.ignored).label("any_ignored"),
            ).where(
                TrackedObject.name.isnot(None),
                TrackedObject.name != "",
            )
            if camera_id:
                named = named.where(TrackedObject.camera_id == camera_id)
            if class_name:
                named = named.where(TrackedObject.class_name == class_name)
            if not show_ignored:
                named = named.where(TrackedObject.ignored != True)
            order_fn = desc if sort.startswith("-") else asc
            col_name = sort.lstrip("-")
            order_col = sqfunc.max(getattr(TrackedObject, col_name, TrackedObject.last_seen))
            named = named.group_by(TrackedObject.name).order_by(order_fn(order_col))
            result = await session.execute(named)
            groups = []
            for row in result.all():
                cameras = list(dict.fromkeys(row.cameras))
                classes = list(dict.fromkeys(row.classes))
                obj_ids = [str(oid) for oid in row.obj_ids]
                groups.append({
                    "name": row.name,
                    "count": row.count,
                    "camera_ids": cameras,
                    "class_names": classes,
                    "last_seen": row.last_seen.isoformat() if row.last_seen else None,
                    "obj_ids": obj_ids,
                    "any_ignored": row.any_ignored,
                })

            # Unnamed items
            unnamed = select(TrackedObject).where(
                (TrackedObject.name.is_(None)) | (TrackedObject.name == "")
            )
            if camera_id:
                unnamed = unnamed.where(TrackedObject.camera_id == camera_id)
            if class_name:
                unnamed = unnamed.where(TrackedObject.class_name == class_name)
            if not show_ignored:
                unnamed = unnamed.where(TrackedObject.ignored != True)
            order_fn2 = desc if sort.startswith("-") else asc
            unnamed = unnamed.order_by(order_fn2(getattr(TrackedObject, col_name, TrackedObject.last_seen)))
            unnamed = unnamed.limit(200)
            result2 = await session.execute(unnamed)
            items = []
            for obj in result2.scalars():
                items.append({
                    "id": str(obj.id),
                    "camera_id": obj.camera_id,
                    "class_name": obj.class_name,
                    "name": obj.name,
                    "ignored": obj.ignored,
                    "last_seen": obj.last_seen.isoformat() if obj.last_seen else None,
                })

            return {"groups": groups, "items": items}

    async def reclassify_frame(self, frame_id: uuid.UUID, new_class: str) -> Optional[dict]:
        """Move frame to a new or existing TrackedObject with the given class."""
        async with await get_session() as session:
            result = await session.execute(
                select(FrameCapture).where(FrameCapture.id == frame_id)
            )
            fc = result.scalar_one_or_none()
            if fc is None:
                return None

            old_obj = await session.execute(
                select(TrackedObject).where(TrackedObject.id == fc.object_id)
            )
            old_obj = old_obj.scalar_one_or_none()
            if old_obj is None:
                return None

            # Find or create a target object for the new class
            target = await session.execute(
                select(TrackedObject).where(
                    TrackedObject.camera_id == old_obj.camera_id,
                    TrackedObject.track_id == old_obj.track_id,
                    TrackedObject.class_name == new_class,
                )
            )
            target_obj = target.scalars().first()
            if target_obj is None:
                target_obj = TrackedObject(
                    camera_id=old_obj.camera_id,
                    track_id=old_obj.track_id,
                    class_name=new_class,
                    name=old_obj.name,
                    first_seen=self._db_timestamp(fc.timestamp),
                    last_seen=self._db_timestamp(fc.timestamp),
                    appearance_count=1,
                )
                session.add(target_obj)
                await session.flush()

            fc.object_id = target_obj.id
            if old_obj.name and not target_obj.name:
                target_obj.name = old_obj.name
            fc_ts = self._db_timestamp(fc.timestamp)
            target_obj.last_seen = fc_ts
            target_obj.appearance_count = (target_obj.appearance_count or 0) + 1

            # Recalc old object
            from sqlalchemy import func
            cnt = await session.execute(
                select(func.count(FrameCapture.id)).where(FrameCapture.object_id == old_obj.id)
            )
            remaining = cnt.scalar() or 0
            old_obj.appearance_count = remaining
            if remaining == 0:
                old_obj.name = None
            else:
                max_ts = await session.execute(
                    select(func.max(FrameCapture.timestamp)).where(FrameCapture.object_id == old_obj.id)
                )
                max_val = max_ts.scalar()
                old_obj.last_seen = self._db_timestamp(max_val) if max_val else old_obj.last_seen

            await session.commit()
            return {"frame_id": str(fc.id), "new_object_id": str(target_obj.id), "class_name": new_class}

    async def move_frame_to_name(self, frame_id: uuid.UUID, target_name: str, class_name: Optional[str] = None) -> Optional[dict]:
        """Move frame to an object with the given name (create if needed)."""
        async with await get_session() as session:
            result = await session.execute(
                select(FrameCapture).where(FrameCapture.id == frame_id)
            )
            fc = result.scalar_one_or_none()
            if fc is None:
                return None
            old_obj = await session.execute(
                select(TrackedObject).where(TrackedObject.id == fc.object_id)
            )
            old_obj = old_obj.scalar_one_or_none()
            if old_obj is None:
                return None
            # Find existing object with target name on same camera
            target = await session.execute(
                select(TrackedObject).where(
                    TrackedObject.name == target_name,
                    TrackedObject.camera_id == old_obj.camera_id,
                ).limit(1)
            )
            target_obj = target.scalar_one_or_none()
            if target_obj is None:
                # Determine class: explicit > dominant for name > old class
                if not class_name:
                    dom = await session.execute(
                        select(TrackedObject.class_name, func.count(TrackedObject.id).label("cnt"))
                        .where(TrackedObject.name == target_name)
                        .where(TrackedObject.class_name != "")
                        .group_by(TrackedObject.class_name)
                        .order_by(text("cnt DESC"))
                        .limit(1)
                    )
                    dom_row = dom.first()
                    class_name = dom_row[0] if dom_row else None
                target_obj = TrackedObject(
                    camera_id=old_obj.camera_id,
                    track_id=old_obj.track_id,
                    class_name=class_name or old_obj.class_name,
                    name=target_name,
                    face_id=old_obj.face_id,
                    face_hash=old_obj.face_hash,
                    plate_number=old_obj.plate_number,
                    embedding=old_obj.embedding,
                    first_seen=self._db_timestamp(fc.timestamp),
                    last_seen=self._db_timestamp(fc.timestamp),
                    appearance_count=1,
                )
                session.add(target_obj)
                await session.flush()
            fc.object_id = target_obj.id
            # Copy metadata from old object if target doesn't have it
            if old_obj.face_id and not target_obj.face_id:
                target_obj.face_id = old_obj.face_id
            if old_obj.plate_number and not target_obj.plate_number:
                target_obj.plate_number = old_obj.plate_number
            fc_ts = self._db_timestamp(fc.timestamp)
            target_obj.last_seen = fc_ts
            target_obj.appearance_count = (target_obj.appearance_count or 0) + 1
            # Recalc old
            from sqlalchemy import func
            cnt = await session.execute(select(func.count(FrameCapture.id)).where(FrameCapture.object_id == old_obj.id))
            remaining = cnt.scalar() or 0
            old_obj.appearance_count = remaining
            if remaining == 0:
                old_obj.name = None
            else:
                max_ts = await session.execute(select(func.max(FrameCapture.timestamp)).where(FrameCapture.object_id == old_obj.id))
                max_val = max_ts.scalar()
                old_obj.last_seen = self._db_timestamp(max_val) if max_val else old_obj.last_seen
            await session.commit()
            return {"frame_id": str(fc.id), "target_name": target_name, "target_object_id": str(target_obj.id), "class_name": target_obj.class_name}

    async def reclassify_group(self, group_name: str, new_class: str) -> dict:
        """Change class_name for all objects with the given name."""
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(TrackedObject.name == group_name)
            )
            objs = result.scalars().all()
            updated = 0
            for obj in objs:
                if obj.class_name == new_class:
                    continue
                obj.class_name = new_class
                updated += 1
            await session.commit()
            return {"updated": updated, "total": len(objs), "class_name": new_class, "name": group_name}

    async def reclassify_by_ids(self, obj_ids: list[str], new_class: str) -> dict:
        """Change class_name for specific object IDs."""
        from uuid import UUID
        uuids = [UUID(i) for i in obj_ids]
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject).where(TrackedObject.id.in_(uuids))
            )
            objs = result.scalars().all()
            updated = 0
            for obj in objs:
                if obj.class_name == new_class:
                    continue
                obj.class_name = new_class
                updated += 1
            await session.commit()
            return {"updated": updated, "total": len(objs), "class_name": new_class}

    async def get_latest_frame(self) -> Optional[dict]:
        """Return the most recently saved frame with its object info."""
        async with await get_session() as session:
            result = await session.execute(
                select(FrameCapture, TrackedObject)
                .join(TrackedObject, FrameCapture.object_id == TrackedObject.id)
                .order_by(FrameCapture.timestamp.desc())
                .limit(1)
            )
            row = result.first()
            if row is None:
                return None
            fc, obj = row
            return {
                "id": str(fc.id),
                "image": f"/frames/{Path(fc.image_path).name}",
                "camera_id": obj.camera_id,
                "class_name": obj.class_name,
                "name": obj.name,
                "confidence": fc.confidence,
                "timestamp": fc.timestamp.isoformat() if fc.timestamp else None,
                "face_id": obj.face_id,
                "plate_number": obj.plate_number,
                "vmr_brand": (obj.metadata_ or {}).get("vmr"),
            }

    async def list_objects(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        name: Optional[str] = None,
        unnamed: bool = False,
        show_ignored: bool = False,
        limit: int = 50,
        offset: int = 0,
        sort: str = "-last_seen",
        date: Optional[str] = None,
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
            if unnamed:
                query = query.where(
                    (TrackedObject.name.is_(None)) | (TrackedObject.name == "")
                )
            if not show_ignored:
                query = query.where(TrackedObject.ignored != True)
            if date:
                from datetime import datetime, timedelta
                dt = datetime.strptime(date, "%Y-%m-%d")
                msk_start = dt - timedelta(hours=3)
                msk_end = dt + timedelta(days=1) - timedelta(hours=3)
                query = query.where(TrackedObject.last_seen >= msk_start)
                query = query.where(TrackedObject.last_seen < msk_end)
            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_object_count(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        unnamed: bool = False,
        show_ignored: bool = False,
        date: Optional[str] = None,
    ) -> int:
        async with await get_session() as session:
            from sqlalchemy import func
            query = select(func.count()).select_from(TrackedObject)
            if camera_id:
                query = query.where(TrackedObject.camera_id == camera_id)
            if class_name:
                query = query.where(TrackedObject.class_name == class_name)
            if unnamed:
                query = query.where(
                    (TrackedObject.name.is_(None)) | (TrackedObject.name == "")
                )
            if not show_ignored:
                query = query.where(TrackedObject.ignored != True)
            if date:
                from datetime import datetime, timedelta
                dt = datetime.strptime(date, "%Y-%m-%d")
                msk_start = dt - timedelta(hours=3)
                msk_end = dt + timedelta(days=1) - timedelta(hours=3)
                query = query.where(TrackedObject.last_seen >= msk_start)
                query = query.where(TrackedObject.last_seen < msk_end)
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
                    .values(last_seen=self._db_timestamp(max_ts))
                )
            await session.commit()
            return True

    async def list_frames(
        self,
        object_id: uuid.UUID,
        limit: int = 0,
        offset: int = 0,
    ) -> list[FrameCapture]:
        async with await get_session() as session:
            query = (
                select(FrameCapture)
                .where(FrameCapture.object_id == object_id)
                .order_by(FrameCapture.timestamp.desc())
                .offset(offset)
            )
            if limit:
                query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_best_frame(self, object_id: uuid.UUID) -> Optional[FrameCapture]:
        async with await get_session() as session:
            from sqlalchemy import literal_column
            area = (FrameCapture.bbox_x2 - FrameCapture.bbox_x1) * (FrameCapture.bbox_y2 - FrameCapture.bbox_y1)
            query = (
                select(FrameCapture)
                .where(FrameCapture.object_id == object_id)
                .order_by(area.desc())
                .limit(1)
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

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

    async def backup_all_objects(self) -> list[dict]:
        async with await get_session() as session:
            result = await session.execute(select(TrackedObject))
            rows = []
            for obj in result.scalars().all():
                rows.append({
                    "id": str(obj.id),
                    "camera_id": obj.camera_id,
                    "track_id": obj.track_id,
                    "class_name": obj.class_name,
                    "name": obj.name,
                    "ignored": obj.ignored,
                    "plate_number": obj.plate_number,
                    "face_id": obj.face_id,
                    "metadata_": obj.metadata_,
                    "first_seen": obj.first_seen.isoformat() if obj.first_seen else None,
                    "last_seen": obj.last_seen.isoformat() if obj.last_seen else None,
                    "appearance_count": obj.appearance_count,
                })
            return rows

    async def restore_objects(self, objects_data: list[dict]) -> int:
        count = 0
        async with await get_session() as session:
            for data in objects_data:
                result = await session.execute(
                    select(TrackedObject).where(TrackedObject.id == uuid.UUID(data["id"]))
                )
                obj = result.scalar_one_or_none()
                if obj is None:
                    continue
                obj.name = data.get("name")
                obj.class_name = data.get("class_name", obj.class_name)
                obj.ignored = data.get("ignored", obj.ignored)
                obj.plate_number = data.get("plate_number")
                obj.face_id = data.get("face_id")
                obj.metadata_ = data.get("metadata_")
                count += 1
            await session.commit()
        return count

    async def get_available_dates(self) -> list[dict]:
        async with await get_session() as session:
            from sqlalchemy import func, cast, Date
            from datetime import timedelta
            msk_ts = TrackedObject.last_seen + timedelta(hours=3)
            result = await session.execute(
                select(
                    cast(msk_ts, Date).label("date"),
                    func.count().label("count"),
                )
                .where(TrackedObject.ignored != True)
                .group_by(cast(msk_ts, Date))
                .order_by(cast(msk_ts, Date).desc())
                .limit(365)
            )
            return [{"date": str(row[0]), "count": row[1]} for row in result.all()]

    async def get_unnamed_objects(self) -> list[TrackedObject]:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject)
                .where(
                    (TrackedObject.name.is_(None)) | (TrackedObject.name == "")
                )
                .where(TrackedObject.ignored != True)
                .order_by(TrackedObject.last_seen.desc())
            )
            return list(result.scalars().all())

    async def get_named_objects(self) -> list[TrackedObject]:
        async with await get_session() as session:
            result = await session.execute(
                select(TrackedObject)
                .where(TrackedObject.name.isnot(None))
                .where(TrackedObject.name != "")
                .where(TrackedObject.ignored != True)
                .order_by(TrackedObject.name)
            )
            return list(result.scalars().all())

    async def auto_assign_names(self, assignments: dict[str, str]) -> int:
        count = 0
        async with await get_session() as session:
            for obj_id_str, name in assignments.items():
                try:
                    result = await session.execute(
                        select(TrackedObject).where(TrackedObject.id == uuid.UUID(obj_id_str))
                    )
                    obj = result.scalar_one_or_none()
                    if obj is None or not name:
                        continue
                    obj.name = name
                    count += 1
                except Exception:
                    pass
            await session.commit()
        return count

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
