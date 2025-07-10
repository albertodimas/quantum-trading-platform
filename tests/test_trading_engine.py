"""
Tests for Trading Engine module.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.engine import TradingEngine
from src.trading.models import Order, OrderSide, OrderStatus, OrderType, Signal


@pytest.fixture
def trading_engine():
    """Create trading engine instance for tests."""
    engine = TradingEngine("binance")
    # Mock the exchange connector
    engine.exchange = MagicMock()
    engine.exchange.create_order = AsyncMock(return_value={"id": "exchange-order-123"})
    engine.exchange.cancel_order = AsyncMock(return_value=True)
    return engine


@pytest.mark.asyncio
async def test_trading_engine_initialization(trading_engine):
    """Test trading engine initialization."""
    assert trading_engine.exchange_name == "binance"
    assert trading_engine._running is False
    assert trading_engine.order_manager is not None
    assert trading_engine.position_manager is not None
    assert trading_engine.risk_manager is not None


@pytest.mark.asyncio
async def test_process_signal_valid(trading_engine):
    """Test processing a valid trading signal."""
    signal = Signal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        confidence=0.85,
        entry_price=45000.0,
        stop_loss=44000.0,
        take_profit=46000.0,
        strategy="test_strategy"
    )
    
    # Mock risk check to pass
    trading_engine.risk_manager.check_signal = AsyncMock(return_value=True)
    
    # Mock order creation
    mock_order = Order(
        id="test-order-123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.1,
        price=45000.0,
        status=OrderStatus.PENDING
    )
    trading_engine.order_manager.create_order = AsyncMock(return_value=mock_order)
    
    # Process signal
    order_id = await trading_engine.process_signal(signal)
    
    assert order_id == "test-order-123"
    trading_engine.risk_manager.check_signal.assert_called_once_with(signal)
    trading_engine.order_manager.create_order.assert_called_once()


@pytest.mark.asyncio
async def test_process_signal_rejected_by_risk(trading_engine):
    """Test signal rejected by risk management."""
    signal = Signal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        confidence=0.3,  # Low confidence
        entry_price=45000.0,
        strategy="test_strategy"
    )
    
    # Mock risk check to fail
    trading_engine.risk_manager.check_signal = AsyncMock(return_value=False)
    
    # Process signal
    order_id = await trading_engine.process_signal(signal)
    
    assert order_id is None
    trading_engine.risk_manager.check_signal.assert_called_once_with(signal)


@pytest.mark.asyncio
async def test_cancel_order(trading_engine):
    """Test order cancellation."""
    order_id = "test-order-123"
    
    # Mock order exists
    mock_order = Order(
        id=order_id,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.1,
        price=45000.0,
        status=OrderStatus.OPEN
    )
    trading_engine.order_manager.get_order = AsyncMock(return_value=mock_order)
    trading_engine.order_manager.update_order_status = AsyncMock()
    
    # Cancel order
    result = await trading_engine.cancel_order(order_id)
    
    assert result is True
    trading_engine.exchange.cancel_order.assert_called_once_with(
        order_id, "BTC/USDT"
    )
    trading_engine.order_manager.update_order_status.assert_called_once_with(
        order_id, OrderStatus.CANCELLED
    )


@pytest.mark.asyncio
async def test_get_portfolio_status(trading_engine):
    """Test getting portfolio status."""
    # Mock portfolio data
    trading_engine.position_manager.get_all_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 0.1,
                "entry_price": 45000.0,
                "current_price": 46000.0,
                "pnl": 100.0
            }
        ]
    )
    
    # Get portfolio status
    status = await trading_engine.get_portfolio_status()
    
    assert status["timestamp"] is not None
    assert status["position_count"] == 1
    assert status["positions"][0]["symbol"] == "BTC/USDT"
    assert status["total_pnl"] == 100.0


@pytest.mark.asyncio
async def test_start_stop_engine(trading_engine):
    """Test starting and stopping the trading engine."""
    # Start engine
    await trading_engine.start()
    assert trading_engine._running is True
    assert len(trading_engine._tasks) > 0
    
    # Stop engine
    await trading_engine.stop()
    assert trading_engine._running is False
    assert len(trading_engine._tasks) == 0


@pytest.mark.asyncio
async def test_submit_order(trading_engine):
    """Test submitting an order to exchange."""
    order = Order(
        id="test-order-123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.1,
        price=45000.0,
        status=OrderStatus.PENDING
    )
    
    # Mock order manager update
    trading_engine.order_manager.update_order_status = AsyncMock()
    
    # Submit order
    await trading_engine._submit_order(order)
    
    # Verify exchange call
    trading_engine.exchange.create_order.assert_called_once_with(
        symbol="BTC/USDT",
        type="limit",
        side="buy",
        amount=0.1,
        price=45000.0
    )
    
    # Verify status update
    trading_engine.order_manager.update_order_status.assert_called_with(
        "test-order-123", OrderStatus.OPEN
    )


@pytest.mark.asyncio
async def test_close_all_positions(trading_engine):
    """Test closing all positions."""
    # Mock open positions
    positions = [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 0.1
        },
        {
            "symbol": "ETH/USDT",
            "side": "short",
            "quantity": 1.0
        }
    ]
    trading_engine.position_manager.get_all_positions = AsyncMock(
        return_value=positions
    )
    
    # Mock order creation
    trading_engine.order_manager.create_order = AsyncMock(
        side_effect=[
            Order(id="close-1", symbol="BTC/USDT", side=OrderSide.SELL, 
                  type=OrderType.MARKET, quantity=0.1, status=OrderStatus.PENDING),
            Order(id="close-2", symbol="ETH/USDT", side=OrderSide.BUY, 
                  type=OrderType.MARKET, quantity=1.0, status=OrderStatus.PENDING)
        ]
    )
    
    # Close all positions
    await trading_engine._close_all_positions()
    
    # Verify orders created
    assert trading_engine.order_manager.create_order.call_count == 2
    assert trading_engine.exchange.create_order.call_count == 2