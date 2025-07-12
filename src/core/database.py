"""
Database Connection Management

Manages PostgreSQL connection pool and provides database dependencies.
"""

import os
from typing import Optional
import asyncpg
from asyncpg.pool import Pool

from .observability import get_logger

logger = get_logger(__name__)

# Global database pool
_db_pool: Optional[Pool] = None


async def init_db_pool(
    database_url: Optional[str] = None,
    min_size: int = 10,
    max_size: int = 20
) -> Pool:
    """
    Initialize database connection pool
    
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
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/quantum_trading"
        )
    
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
    """Close database connection pool"""
    global _db_pool
    
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None
        logger.info("Database pool closed")


async def get_db_pool() -> Pool:
    """
    Get database connection pool
    
    FastAPI dependency for getting database pool.
    """
    global _db_pool
    
    if _db_pool is None:
        _db_pool = await init_db_pool()
        
    return _db_pool


async def get_db_connection():
    """
    Get database connection from pool
    
    FastAPI dependency for getting a database connection.
    Usage:
        async def endpoint(conn = Depends(get_db_connection)):
            ...
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        yield connection