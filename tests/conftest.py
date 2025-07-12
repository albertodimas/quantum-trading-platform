"""
Pytest configuration and shared fixtures
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from faker import Faker

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.architecture import DIContainer, Scope
from src.core.messaging import InMemoryBroker, Event
from src.models.trading import Order, Trade, OrderType, OrderSide, OrderStatus
from src.models.exchange import ExchangeCredentials


# Initialize faker
fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def di_container():
    """Provide a fresh DI container for each test"""
    container = DIContainer()
    yield container
    # Cleanup
    asyncio.create_task(container.dispose())


@pytest.fixture
def mock_exchange():
    """Mock exchange for testing"""
    exchange = AsyncMock()
    exchange.id = "mock_exchange"
    exchange.connect = AsyncMock()
    exchange.disconnect = AsyncMock()
    exchange.place_order = AsyncMock(return_value="order_123")
    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.get_balance = AsyncMock(return_value={"USDT": 10000.0})
    exchange.get_ticker = AsyncMock(return_value={"bid": 99.5, "ask": 100.5})
    return exchange


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing"""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.unsubscribe = AsyncMock()
    return bus


@pytest.fixture
def sample_order():
    """Create a sample order"""
    return Order(
        id="test_order_123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.1,
        price=50000.0,
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_trade():
    """Create a sample trade"""
    return Trade(
        id="test_trade_123",
        order_id="test_order_123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        entry_price=50000.0,
        entry_time=datetime.utcnow(),
        strategy_id="test_strategy",
        exchange_id="binance"
    )


@pytest.fixture
def sample_trades():
    """Create a list of sample trades"""
    trades = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(100):
        trade = Trade(
            id=f"trade_{i}",
            order_id=f"order_{i}",
            symbol=fake.random_element(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
            side=fake.random_element([OrderSide.BUY, OrderSide.SELL]),
            quantity=fake.random.uniform(0.01, 1.0),
            entry_price=fake.random.uniform(1000, 60000),
            entry_time=base_time + timedelta(hours=i),
            strategy_id=fake.random_element(["strategy_1", "strategy_2", "strategy_3"]),
            exchange_id=fake.random_element(["binance", "coinbase"])
        )
        
        # Add exit for some trades
        if i % 3 == 0:
            trade.exit_price = trade.entry_price * fake.random.uniform(0.95, 1.05)
            trade.exit_time = trade.entry_time + timedelta(hours=fake.random.randint(1, 24))
            trade.status = OrderStatus.FILLED
            trade.calculate_pnl()
        
        trades.append(trade)
    
    return trades


@pytest.fixture
async def inmemory_broker():
    """Provide InMemory message broker"""
    broker = InMemoryBroker()
    await broker.connect()
    yield broker
    await broker.disconnect()


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('redis.asyncio.Redis') as mock:
        client = AsyncMock()
        client.ping = AsyncMock(return_value=True)
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.expire = AsyncMock(return_value=True)
        client.publish = AsyncMock(return_value=1)
        client.subscribe = AsyncMock()
        client.xadd = AsyncMock(return_value=b"stream-id")
        client.xread = AsyncMock(return_value=[])
        mock.from_url.return_value = client
        yield client


@pytest.fixture
def mock_database():
    """Mock database connection"""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock()
    db.transaction = AsyncMock()
    return db


@pytest.fixture
def exchange_credentials():
    """Sample exchange credentials"""
    return ExchangeCredentials(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase"
    )


@pytest.fixture
def market_data():
    """Sample market data"""
    return {
        "symbol": "BTC/USDT",
        "bid": 50000.0,
        "ask": 50010.0,
        "last": 50005.0,
        "volume": 1000.0,
        "timestamp": datetime.utcnow()
    }


@pytest.fixture
def performance_snapshots():
    """Generate sample performance snapshots"""
    snapshots = []
    equity = 100000.0
    
    for i in range(100):
        # Simulate returns
        daily_return = fake.random.uniform(-0.02, 0.02)
        equity *= (1 + daily_return)
        
        snapshot = Mock()
        snapshot.timestamp = datetime.utcnow() - timedelta(days=100-i)
        snapshot.equity = equity
        snapshot.daily_return = daily_return
        snapshot.trades_count = fake.random.randint(0, 10)
        
        snapshots.append(snapshot)
    
    return snapshots


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    with patch('src.core.logger.get_logger') as mock:
        logger = Mock()
        logger.debug = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        logger.critical = Mock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "exchange": {
            "default": "binance",
            "rate_limit": 100,
            "timeout": 30
        },
        "risk": {
            "max_position_size": 10000,
            "max_daily_loss": 1000,
            "max_leverage": 3.0
        },
        "strategy": {
            "default_timeframe": "1h",
            "max_concurrent": 5
        },
        "database": {
            "url": "postgresql://test:test@localhost:5432/test"
        },
        "redis": {
            "url": "redis://localhost:6379/0"
        }
    }


# Async fixtures
@pytest_asyncio.fixture
async def async_client():
    """Async HTTP client for testing"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        yield session


# Markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis"
    )
    config.addinivalue_line(
        "markers", "requires_postgres: mark test as requiring PostgreSQL"
    )
    config.addinivalue_line(
        "markers", "requires_rabbitmq: mark test as requiring RabbitMQ"
    )
    config.addinivalue_line(
        "markers", "live_exchange: mark test as requiring live exchange connection"
    )


# Test data generators
class DataGenerator:
    """Generate test data"""
    
    @staticmethod
    def generate_ohlcv(days: int = 30) -> List[dict]:
        """Generate OHLCV data"""
        data = []
        base_price = 50000.0
        base_time = datetime.utcnow() - timedelta(days=days)
        
        for i in range(days * 24):  # Hourly data
            open_price = base_price
            high = open_price * fake.random.uniform(1.0, 1.02)
            low = open_price * fake.random.uniform(0.98, 1.0)
            close = fake.random.uniform(low, high)
            volume = fake.random.uniform(100, 1000)
            
            data.append({
                "timestamp": base_time + timedelta(hours=i),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
            
            base_price = close
        
        return data
    
    @staticmethod
    def generate_orderbook(depth: int = 10) -> dict:
        """Generate orderbook data"""
        mid_price = 50000.0
        
        bids = []
        asks = []
        
        for i in range(depth):
            bid_price = mid_price - (i + 1) * 10
            ask_price = mid_price + (i + 1) * 10
            
            bids.append([
                bid_price,
                fake.random.uniform(0.1, 1.0)
            ])
            
            asks.append([
                ask_price,
                fake.random.uniform(0.1, 1.0)
            ])
        
        return {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.utcnow()
        }


@pytest.fixture
def data_generator():
    """Provide data generator"""
    return DataGenerator()


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")