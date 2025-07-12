"""
Database Module

Provides database connection management and session handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import asyncpg
from asyncpg.pool import Pool

from .config import settings
from .observability import get_logger

logger = get_logger(__name__)

# Base class for all models
Base = declarative_base()

# Global engine and session factory for SQLAlchemy
_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker] = None

# Global asyncpg pool for direct queries
_db_pool: Optional[Pool] = None


def get_database_url() -> str:
    """Get database URL from settings"""
    # Convert PostgresDsn to string and ensure it's async
    db_url = str(settings.database_url)
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return db_url


async def init_database() -> None:
    """Initialize database engine and session factory"""
    global _engine, _async_session_factory
    
    if _engine is not None:
        logger.warning("Database already initialized")
        return
    
    # Create engine
    database_url = get_database_url()
    
    if settings.is_production:
        # Production settings
        _engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout,
            pool_pre_ping=True,  # Verify connections before using
        )
    else:
        # Development settings
        _engine = create_async_engine(
            database_url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    
    # Create session factory
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )
    
    logger.info("Database initialized successfully")


async def close_database() -> None:
    """Close database connections"""
    global _engine, _async_session_factory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database connections closed")


def get_engine() -> AsyncEngine:
    """Get database engine"""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Get session factory"""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session.
    
    This is typically used as a dependency in FastAPI:
    
    @router.get("/items")
    async def get_items(session: AsyncSession = Depends(get_session)):
        ...
    """
    if _async_session_factory is None:
        await init_database()
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    
    Usage:
        async with get_db_session() as session:
            # Use session here
            ...
    """
    if _async_session_factory is None:
        await init_database()
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Legacy asyncpg pool support (for backward compatibility)
async def init_db_pool(
    database_url: Optional[str] = None,
    min_size: int = 10,
    max_size: int = 20
) -> Pool:
    """
    Initialize asyncpg connection pool (legacy support)
    
    Args:
        database_url: PostgreSQL connection URL
        min_size: Minimum number of connections
        max_size: Maximum number of connections
        
    Returns:
        Database connection pool
    """
    global _db_pool
    
    if _db_pool is not None:
        return _db_pool
    
    # Get database URL from environment if not provided
    if database_url is None:
        database_url = str(settings.database_url)
    
    try:
        # Create connection pool
        _db_pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
            server_settings={
                'application_name': 'quantum_trading_platform',
                'jit': 'off'
            }
        )
        
        # Test connection
        async with _db_pool.acquire() as conn:
            version = await conn.fetchval('SELECT version()')
            logger.info(f"Connected to PostgreSQL: {version}")
            
        return _db_pool
        
    except Exception as e:
        logger.error(f"Failed to create database pool: {str(e)}")
        raise


async def close_db_pool() -> None:
    """Close asyncpg connection pool"""
    global _db_pool
    
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None
        logger.info("Database pool closed")


async def get_db_pool() -> Pool:
    """
    Get asyncpg connection pool (legacy support)
    
    FastAPI dependency for getting database pool.
    """
    global _db_pool
    
    if _db_pool is None:
        _db_pool = await init_db_pool()
        
    return _db_pool


async def get_db_connection():
    """
    Get database connection from pool (legacy support)
    
    FastAPI dependency for getting a database connection.
    Usage:
        async def endpoint(conn = Depends(get_db_connection)):
            ...
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        yield connection


async def create_tables() -> None:
    """Create all tables in the database"""
    engine = get_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created")


async def drop_tables() -> None:
    """Drop all tables from the database"""
    engine = get_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("Database tables dropped")


async def check_connection() -> bool:
    """Check if database connection is working"""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False


# Transaction decorator
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def transactional(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to run function in a database transaction.
    
    Usage:
        @transactional
        async def create_user(user_data: dict) -> User:
            # This will run in a transaction
            ...
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with get_db_session() as session:
            # Inject session as first argument if it's not already there
            if "session" not in kwargs and (
                not args or not isinstance(args[0], AsyncSession)
            ):
                kwargs["session"] = session
            
            return await func(*args, **kwargs)
    
    return wrapper