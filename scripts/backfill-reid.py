# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""
backfill-reid.py — Compute embeddings for existing unnamed vehicles and
find cross-camera matches. Run after deploy to process historical data.

Usage:
    python scripts/backfill-reid.py [--threshold 0.85] [--limit 200]

Requires DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD env vars.
"""
import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from sqlalchemy import select, text as sqtext

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import init_db, get_session, init_pgvector, init_schema
from src.storage.models import TrackedObject, FrameCapture
from src.recognition.reid import compute_embedding


async def main():
    parser = argparse.ArgumentParser(description="Backfill ReID embeddings")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    db_cfg = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", 5432)),
        "user": os.environ.get("DB_USER", "cam"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", "cam"),
    }

    await init_db(**db_cfg)
    await init_pgvector()
    await init_schema()

    vehicle_classes = ("car", "truck", "bus", "motorcycle")

    # Find unnamed vehicles without embeddings, most recent first
    async with await get_session() as session:
        result = await session.execute(
            select(TrackedObject)
            .where(TrackedObject.class_name.in_(vehicle_classes))
            .where(TrackedObject.embedding.is_(None))
            .where(TrackedObject.name.is_(None))
            .order_by(TrackedObject.last_seen.desc())
            .limit(args.limit)
        )
        objects = list(result.scalars().all())

    logger.info(f"Found {len(objects)} unnamed vehicles without embeddings")

    matched = 0
    for obj in objects:
        # Load the most recent frame
        async with await get_session() as session:
            frame_result = await session.execute(
                select(FrameCapture)
                .where(FrameCapture.object_id == obj.id)
                .order_by(FrameCapture.timestamp.desc())
                .limit(1)
            )
            fc = frame_result.scalar_one_or_none()

        if fc is None:
            logger.debug(f"No frame for {obj.id}, skipping")
            continue

        img = cv2.imread(fc.image_path)
        if img is None:
            logger.debug(f"Cannot read {fc.image_path}, skipping")
            continue

        bbox = (fc.bbox_x1, fc.bbox_y1, fc.bbox_x2, fc.bbox_y2)
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0:
            continue

        vec = compute_embedding(crop)
        if not vec or len(vec) != 512:
            continue

        from src.storage.repository import StorageRepository
        repo = StorageRepository(Path("/data/frames"))

        await repo.update_embedding(obj.id, vec)
        matches = await repo.find_similar_objects(
            vec, obj.class_name, exclude_object_id=obj.id,
            threshold=args.threshold,
        )

        if matches:
            best, score = matches[0]
            name = best.name or f"vehicle-{best.id.hex[:8]}"
            if not best.name:
                await repo.update_object_name(best.id, name)
            await repo.update_object_name(obj.id, name)
            matched += 1
            logger.info(f"ReID: {obj.id} ({obj.camera_id}) → '{name}' "
                       f"score={score:.3f} match_with={best.id} ({best.camera_id})")
        else:
            logger.info(f"ReID: {obj.id} ({obj.camera_id}) no matches above {args.threshold}")

    logger.info(f"Done. Processed {len(objects)}, auto-named {matched}")


if __name__ == "__main__":
    asyncio.run(main())
