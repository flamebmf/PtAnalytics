# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from loguru import logger


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory: async_sessionmaker | None = None


async def init_db(host: str, port: int, user: str, password: str, database: str):
    global _engine, _session_factory
    url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    _engine = create_async_engine(url, pool_size=10, max_overflow=5, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info(f"Database connected: {host}:{port}/{database}")


async def get_session() -> AsyncSession:
    if _session_factory is None:
        raise RuntimeError("DB not initialized. Call init_db() first.")
    return _session_factory()


async def close_db():
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database disconnected")


async def init_pgvector():
    """Ensure pgvector extension is enabled."""
    async with await get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()


_crop_samples_ddl = """
CREATE TABLE IF NOT EXISTS crop_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id VARCHAR(64) NOT NULL,
    class_name VARCHAR(32) NOT NULL,
    bbox_x1 INTEGER NOT NULL, bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL, bbox_y2 INTEGER NOT NULL,
    image_path VARCHAR(512) NOT NULL,
    phase VARCHAR(16) DEFAULT 'entry',
    is_val BOOLEAN DEFAULT false,
    timestamp TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_crop_samples_lookup
    ON crop_samples(camera_id, class_name, timestamp);
"""


async def init_schema():
    """Create application tables that may not exist (for deployments without init SQL)."""
    async with await get_session() as session:
        for stmt in _crop_samples_ddl.split(";"):
            stripped = stmt.strip()
            if stripped:
                await session.execute(text(stripped + ";"))
        await session.commit()
        logger.info("Schema check complete (crop_samples)")
