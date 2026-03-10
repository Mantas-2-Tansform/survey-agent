"""
backend/database/db.py

Async SQLAlchemy engine + session factory.

LOCAL DEV  --> SQLite   (aiosqlite, zero config, creates survey.db automatically)
PRODUCTION --> PostgreSQL (asyncpg, set DATABASE_URL env var)

DATABASE_URL examples:
  SQLite (default): sqlite+aiosqlite:///./survey.db
  PostgreSQL:       postgresql+asyncpg://user:pass@localhost:5432/survey_db
"""
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, StaticPool

from database.models import Base

# ---------------------------------------------------------------------------
# Database URL — defaults to local SQLite file (no setup required)
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./survey.db",
)

_is_sqlite = DATABASE_URL.startswith("sqlite")

# ---------------------------------------------------------------------------
# Engine
# SQLite requires StaticPool + check_same_thread=False for async.
# PostgreSQL uses NullPool (safe for multi-process / serverless).
# ---------------------------------------------------------------------------
if _is_sqlite:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,
    )

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a transactional async DB session.

    Usage in a route:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Schema initialisation helper
# Dev: called automatically at startup via main.py lifespan.
# Prod: use `alembic upgrade head` instead and set AUTO_MIGRATE=false.
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """
    Create all tables defined in models.py if they don't exist,
    then apply any missing column migrations for existing tables.

    SQLAlchemy's create_all() only creates MISSING TABLES — it never
    adds new columns to existing ones. We handle that here with an
    explicit ALTER TABLE fallback so the app works against existing
    survey.db files without requiring a manual migration step.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _apply_migrations(conn)


async def _apply_migrations(conn) -> None:
    """
    Apply any missing column / schema changes to EXISTING tables.
    Each migration is idempotent — safe to run on every startup.
    """
    import logging
    log = logging.getLogger(__name__)

    migrations = [
        # (table, column, sql_type, default_value)
        # Added in v2.1 — branching/logic rules per question
        ("questions", "question_logic", "JSON", None),
    ]

    for table, column, col_type, default in migrations:
        # Check if column already exists
        try:
            result = await conn.execute(
                __import__("sqlalchemy", fromlist=["text"]).text(
                    f"SELECT {column} FROM {table} LIMIT 1"
                )
            )
            # Column exists — nothing to do
        except Exception:
            # Column missing — add it
            try:
                default_clause = (
                    f" DEFAULT {default}" if default is not None else ""
                )
                await conn.execute(
                    __import__("sqlalchemy", fromlist=["text"]).text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
                    )
                )
                log.info("Migration applied: ALTER TABLE %s ADD COLUMN %s", table, column)
            except Exception as e:
                log.warning("Migration failed for %s.%s: %s", table, column, e)