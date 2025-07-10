"""
Pytest configuration and fixtures.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.api import create_app
from src.core.config import settings
from src.core.database import Base


# Override settings for testing
settings.testing = True
settings.database_url = "postgresql+asyncpg://postgres:test@localhost:5432/test_quantum_trading"
settings.redis_url = "redis://localhost:6379/1"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_db():
    """Create test database."""
    engine = create_async_engine(
        settings.database_url.replace("/test_quantum_trading", "/postgres")
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: sync_conn.execute("DROP DATABASE IF EXISTS test_quantum_trading"))
        await conn.run_sync(lambda sync_conn: sync_conn.execute("CREATE DATABASE test_quantum_trading"))
    
    await engine.dispose()
    
    # Create tables
    test_engine = create_async_engine(settings.database_url)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Cleanup
    await test_engine.dispose()
    engine = create_async_engine(
        settings.database_url.replace("/test_quantum_trading", "/postgres")
    )
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: sync_conn.execute("DROP DATABASE IF EXISTS test_quantum_trading"))
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for tests."""
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()
    
    await engine.dispose()


@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[Redis, None]:
    """Create Redis client for tests."""
    client = await Redis.from_url(str(settings.redis_url))
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
def app():
    """Create FastAPI app for tests."""
    return create_app()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Get authentication headers for tests."""
    # In a real app, this would generate a test JWT token
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def mock_market_data():
    """Mock market data for tests."""
    return {
        "symbol": "BTC/USDT",
        "bid": 45000.0,
        "ask": 45010.0,
        "last": 45005.0,
        "volume": 1234.56,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_order():
    """Mock order for tests."""
    return {
        "id": "test-order-123",
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "quantity": 0.1,
        "price": 45000.0,
        "status": "open",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_signal():
    """Mock trading signal for tests."""
    return {
        "symbol": "BTC/USDT",
        "side": "buy",
        "confidence": 0.85,
        "entry_price": 45000.0,
        "stop_loss": 44000.0,
        "take_profit": 46000.0,
        "strategy": "momentum_ai"
    }