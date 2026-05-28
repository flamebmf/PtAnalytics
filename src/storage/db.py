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
