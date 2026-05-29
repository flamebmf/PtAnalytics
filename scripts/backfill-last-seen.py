"""Set last_seen for all tracked objects to max(frame_captures.timestamp)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from sqlalchemy import select, func, update

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

    async with await get_session() as session:
        objs = await session.execute(select(TrackedObject.id, TrackedObject.last_seen))
        rows = objs.all()
        fixed = 0

        for obj_id, curr_seen in rows:
            ts_result = await session.execute(
                select(func.max(FrameCapture.timestamp)).where(FrameCapture.object_id == obj_id)
            )
            max_ts = ts_result.scalar()
            if max_ts is None:
                continue
            if max_ts.tzinfo is not None:
                max_ts = max_ts.astimezone(None).replace(tzinfo=None)
            if curr_seen != max_ts:
                await session.execute(
                    update(TrackedObject)
                    .where(TrackedObject.id == obj_id)
                    .values(last_seen=max_ts)
                )
                fixed += 1

        await session.commit()
        logger.info(f"Fixed {fixed} objects, skipped {len(rows) - fixed} (already correct or no frames)")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
