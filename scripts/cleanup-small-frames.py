# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
"""Clean up frames with bbox smaller than min_bbox_size (w < 40 or h < 40)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from sqlalchemy import select, func

from src.storage import init_db, get_session, close_db
from src.storage.models import FrameCapture, TrackedObject
from src.config import load_settings


async def main():
    config_dir = Path("config")
    settings = load_settings(config_dir)
    db = settings.get("database", {})
    await init_db(
        host=db.get("host", "localhost"),
        port=db.get("port", 5432),
        user=db.get("user", "cam"),
        password=db.get("password", "cam"),
        database=db.get("database", "cam"),
    )

    min_size = settings.get("detector", {}).get("min_bbox_size", 40)

    async with await get_session() as session:
        # Find all small frames
        rows = await session.execute(
            select(FrameCapture).where(
                (FrameCapture.bbox_x2 - FrameCapture.bbox_x1 < min_size)
                | (FrameCapture.bbox_y2 - FrameCapture.bbox_y1 < min_size)
            )
        )
        frames = rows.scalars().all()
        logger.info(f"Found {len(frames)} frames with bbox < {min_size}px")

        affected_objects = set()
        deleted_files = 0
        for fc in frames:
            affected_objects.add(fc.object_id)
            try:
                if fc.image_path and Path(fc.image_path).exists():
                    Path(fc.image_path).unlink()
                    deleted_files += 1
            except Exception:
                pass
            await session.delete(fc)

        # Update last_seen for affected objects
        for obj_id in affected_objects:
            ts_result = await session.execute(
                select(func.max(FrameCapture.timestamp)).where(FrameCapture.object_id == obj_id)
            )
            max_ts = ts_result.scalar()
            if max_ts is not None:
                # make naive for TIMESTAMP WITHOUT TIME ZONE column
                if max_ts.tzinfo is not None:
                    max_ts = max_ts.astimezone(tz=None).replace(tzinfo=None)
                await session.execute(
                    TrackedObject.__table__.update()
                    .where(TrackedObject.id == obj_id)
                    .values(last_seen=max_ts)
                )

        await session.commit()
        logger.info(f"Deleted {len(frames)} frames, {deleted_files} files cleaned")

        # Find and report orphaned objects
        orphans = await session.execute(
            select(TrackedObject).where(
                TrackedObject.id.in_(affected_objects)
            )
        )
        for obj in orphans.scalars().all():
            cnt = await session.execute(
                select(func.count()).select_from(FrameCapture).where(FrameCapture.object_id == obj.id)
            )
            remaining = cnt.scalar() or 0
            if remaining == 0:
                logger.warning(f"Object {obj.id} ({obj.camera_id}) now has 0 frames")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
