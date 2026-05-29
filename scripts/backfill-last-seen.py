"""Recalculate first_seen/last_seen from actual frame timestamps for all objects."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from sqlalchemy import select, func, update

from src.storage import init_db, get_session, close_db
from src.storage.models import FrameCapture, TrackedObject
from src.config import load_settings


def _unlocal(ts):
    """Make datetime naive (assume UTC) for TIMESTAMP WITHOUT TIME ZONE column."""
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(None).replace(tzinfo=None)
    return ts


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
        objs = await session.execute(select(TrackedObject.id, TrackedObject.last_seen, TrackedObject.first_seen))
        rows = objs.all()
        fixed_last = 0
        fixed_first = 0

        for obj_id, curr_last, curr_first in rows:
            ts_result = await session.execute(
                select(
                    func.min(FrameCapture.timestamp),
                    func.max(FrameCapture.timestamp),
                ).where(FrameCapture.object_id == obj_id)
            )
            row = ts_result.one_or_none()
            if row is None or row[0] is None:
                continue
            min_ts, max_ts = _unlocal(row[0]), _unlocal(row[1])
            vals = {}
            if max_ts is not None and _unlocal(curr_last) != max_ts:
                vals["last_seen"] = max_ts
                fixed_last += 1
            if min_ts is not None and _unlocal(curr_first) != min_ts:
                vals["first_seen"] = min_ts
                fixed_first += 1
            if vals:
                await session.execute(
                    update(TrackedObject).where(TrackedObject.id == obj_id).values(**vals)
                )

        await session.commit()
        logger.info(f"Fixed last_seen for {fixed_last} objects, first_seen for {fixed_first} objects")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
