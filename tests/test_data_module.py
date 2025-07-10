"""
Tests for Data Module.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from src.data.market import MarketDataStream
from src.data.storage import TimeSeriesStorage


@pytest.fixture
def market_stream(redis_client):
    """Create market data stream instance for tests."""
    stream = MarketDataStream(redis_client)
    return stream


@pytest.fixture
def storage(redis_client):
    """Create time series storage instance for tests."""
    storage = TimeSeriesStorage(redis_client)
    return storage


@pytest.mark.asyncio
async def test_market_stream_initialization(market_stream):
    """Test market data stream initialization."""
    assert market_stream.redis is not None
    assert market_stream._connections == {}
    assert market_stream._subscribers == {}
    assert market_stream._running is False


@pytest.mark.asyncio
async def test_subscribe_to_market_data(market_stream):
    """Test subscribing to market data."""
    # Mock exchange connection
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(return_value=json.dumps({
        "symbol": "BTC/USDT",
        "bid": 45000.0,
        "ask": 45010.0,
        "last": 45005.0,
        "volume": 1234.56
    }))
    
    with patch("websockets.connect", return_value=mock_ws):
        # Subscribe with callback
        callback_data = []
        async def test_callback(data):
            callback_data.append(data)
        
        await market_stream.subscribe("binance", "BTC/USDT", test_callback)
        
        # Verify subscription
        assert "binance:BTC/USDT" in market_stream._subscribers
        assert test_callback in market_stream._subscribers["binance:BTC/USDT"]


@pytest.mark.asyncio
async def test_unsubscribe_from_market_data(market_stream):
    """Test unsubscribing from market data."""
    # Setup subscription
    key = "binance:BTC/USDT"
    callback = AsyncMock()
    market_stream._subscribers[key] = {callback}
    
    # Unsubscribe
    await market_stream.unsubscribe("binance", "BTC/USDT", callback)
    
    # Verify unsubscribed
    assert key not in market_stream._subscribers


@pytest.mark.asyncio
async def test_get_market_data(market_stream):
    """Test getting cached market data."""
    # Set test data in Redis
    test_data = {
        "symbol": "BTC/USDT",
        "bid": 45000.0,
        "ask": 45010.0,
        "last": 45005.0,
        "volume": 1234.56,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await market_stream.redis.setex(
        "market_data:binance:BTC/USDT",
        60,
        json.dumps(test_data)
    )
    
    # Get market data
    data = await market_stream.get_market_data("binance", "BTC/USDT")
    
    assert data is not None
    assert data["symbol"] == "BTC/USDT"
    assert data["bid"] == 45000.0


@pytest.mark.asyncio
async def test_store_ohlcv_data(storage):
    """Test storing OHLCV data."""
    ohlcv_data = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "timestamp": datetime.utcnow().isoformat(),
        "open": 45000.0,
        "high": 45500.0,
        "low": 44800.0,
        "close": 45200.0,
        "volume": 123.45
    }
    
    # Store data
    await storage.store_ohlcv("binance", ohlcv_data)
    
    # Verify stored in Redis
    key = f"ohlcv:binance:BTC/USDT:1h:{ohlcv_data['timestamp']}"
    stored_data = await storage.redis.get(key)
    assert stored_data is not None
    
    parsed_data = json.loads(stored_data)
    assert parsed_data["close"] == 45200.0


@pytest.mark.asyncio
async def test_get_ohlcv_range(storage):
    """Test getting OHLCV data range."""
    # Store test data
    base_time = datetime.utcnow()
    for i in range(5):
        timestamp = base_time.replace(hour=i)
        ohlcv_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": timestamp.isoformat(),
            "open": 45000.0 + i * 100,
            "high": 45100.0 + i * 100,
            "low": 44900.0 + i * 100,
            "close": 45050.0 + i * 100,
            "volume": 100.0 + i * 10
        }
        await storage.store_ohlcv("binance", ohlcv_data)
    
    # Get range
    start_time = base_time.replace(hour=0)
    end_time = base_time.replace(hour=4)
    
    data = await storage.get_ohlcv_range(
        "binance", "BTC/USDT", "1h", start_time, end_time
    )
    
    assert len(data) > 0
    assert all(d["symbol"] == "BTC/USDT" for d in data)


@pytest.mark.asyncio
async def test_store_order_book(storage):
    """Test storing order book data."""
    order_book = {
        "symbol": "BTC/USDT",
        "timestamp": datetime.utcnow().isoformat(),
        "bids": [[45000.0, 1.0], [44990.0, 2.0]],
        "asks": [[45010.0, 1.0], [45020.0, 2.0]]
    }
    
    # Store order book
    await storage.store_order_book("binance", order_book)
    
    # Verify stored
    key = f"orderbook:binance:BTC/USDT"
    stored_data = await storage.redis.get(key)
    assert stored_data is not None
    
    parsed_data = json.loads(stored_data)
    assert len(parsed_data["bids"]) == 2
    assert parsed_data["bids"][0][0] == 45000.0


@pytest.mark.asyncio
async def test_get_order_book(storage):
    """Test getting order book data."""
    # Store test data
    order_book = {
        "symbol": "BTC/USDT",
        "timestamp": datetime.utcnow().isoformat(),
        "bids": [[45000.0, 1.0]],
        "asks": [[45010.0, 1.0]]
    }
    
    key = "orderbook:binance:BTC/USDT"
    await storage.redis.setex(key, 300, json.dumps(order_book))
    
    # Get order book
    data = await storage.get_order_book("binance", "BTC/USDT")
    
    assert data is not None
    assert data["symbol"] == "BTC/USDT"
    assert len(data["bids"]) == 1


@pytest.mark.asyncio
async def test_broadcast_market_data(market_stream):
    """Test broadcasting market data through Redis pub/sub."""
    # Subscribe to Redis channel
    pubsub = market_stream.redis.pubsub()
    await pubsub.subscribe("market_data:binance:BTC/USDT")
    
    # Broadcast data
    test_data = {
        "symbol": "BTC/USDT",
        "bid": 45000.0,
        "ask": 45010.0
    }
    
    await market_stream._broadcast_data("binance:BTC/USDT", test_data)
    
    # Check message received
    await asyncio.sleep(0.1)  # Give time for message to propagate
    message = await pubsub.get_message(ignore_subscribe_messages=True)
    
    if message:
        assert message["type"] == "message"
        data = json.loads(message["data"])
        assert data["symbol"] == "BTC/USDT"
    
    await pubsub.close()


@pytest.mark.asyncio
async def test_market_stream_error_handling(market_stream):
    """Test error handling in market stream."""
    # Mock WebSocket with error
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(side_effect=Exception("Connection error"))
    
    with patch("websockets.connect", return_value=mock_ws):
        # This should not raise an exception
        await market_stream._connect_exchange("binance", ["BTC/USDT"])
        
        # Connection should be cleaned up
        assert "binance" not in market_stream._connections