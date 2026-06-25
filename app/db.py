from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings

# SQLite needs a shared connection so in-memory tests keep their tables across sessions.
connect_args: dict = {}
engine_kwargs: dict = {}
if settings.database_url.startswith("sqlite"):
    from sqlalchemy.pool import StaticPool

    connect_args = {"check_same_thread": False}
    engine_kwargs = {"poolclass": StaticPool}
else:
    # Postgres / any server DB: a real bounded QueuePool. These bound how many connections EACH
    # worker process holds; total at peak ~= (workers) x (pool_size + max_overflow). That must stay
    # under Postgres max_connections (raise it, or run pgbouncer) when scaling workers wide.
    engine_kwargs = {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_timeout": settings.db_pool_timeout,
        "pool_recycle": settings.db_pool_recycle,
    }

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
    **engine_kwargs,
)
# Dev/prod parity: make SQLite enforce foreign keys the way Postgres always does, so FK-ordering
# bugs (e.g. inserting a child before its parent) fail loudly in tests instead of silently passing
# on SQLite and then exploding on Postgres in production.
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_fk_on(dbapi_connection, _record):  # pragma: no cover
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Phase-0 schema bootstrap. Dev-only: real migrations move to Alembic once the schema
    stabilises (Phase 1). create_all is intentionally chosen here so the skeleton runs with
    zero migration friction."""
    from . import models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=engine)
