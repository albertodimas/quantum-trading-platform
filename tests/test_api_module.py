"""
Tests for API Module.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

from src.api.websocket import ConnectionManager, WebSocketManager
from src.trading.models import OrderSide, OrderType


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_trading_signal_endpoint(client, auth_headers, mock_signal):
    """Test trading signal processing endpoint."""
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine") as mock_engine:
            # Mock trading engine
            mock_engine.return_value.process_signal = AsyncMock(return_value="order-123")
            
            response = client.post(
                "/api/v1/trading/signal",
                json=mock_signal,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["order_id"] == "order-123"
            assert data["status"] == "accepted"


def test_trading_signal_rejected(client, auth_headers, mock_signal):
    """Test rejected trading signal."""
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine") as mock_engine:
            # Mock trading engine to reject signal
            mock_engine.return_value.process_signal = AsyncMock(return_value=None)
            
            response = client.post(
                "/api/v1/trading/signal",
                json=mock_signal,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["order_id"] is None
            assert data["status"] == "rejected"


def test_create_order_endpoint(client, auth_headers):
    """Test direct order creation endpoint."""
    order_request = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "order_type": "limit",
        "quantity": 0.1,
        "price": 45000.0
    }
    
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine") as mock_engine:
            # Mock order creation
            mock_order = MagicMock(
                id="order-123",
                status="pending",
                created_at=datetime.utcnow()
            )
            mock_engine.return_value.order_manager.create_order = AsyncMock(
                return_value=mock_order
            )
            mock_engine.return_value._submit_order = AsyncMock()
            
            response = client.post(
                "/api/v1/trading/order",
                json=order_request,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["order_id"] == "order-123"
            assert data["status"] == "pending"


def test_cancel_order_endpoint(client, auth_headers):
    """Test order cancellation endpoint."""
    order_id = "order-123"
    
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine") as mock_engine:
            mock_engine.return_value.cancel_order = AsyncMock(return_value=True)
            
            response = client.delete(
                f"/api/v1/trading/order/{order_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"
            assert data["order_id"] == order_id


def test_portfolio_status_endpoint(client, auth_headers):
    """Test portfolio status endpoint."""
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine") as mock_engine:
            mock_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_value": 10000.0,
                "total_pnl": 500.0,
                "position_count": 2,
                "positions": [
                    {"symbol": "BTC/USDT", "pnl": 300.0},
                    {"symbol": "ETH/USDT", "pnl": 200.0}
                ],
                "metrics": {"win_rate": 0.65}
            }
            mock_engine.return_value.get_portfolio_status = AsyncMock(
                return_value=mock_status
            )
            
            response = client.get(
                "/api/v1/trading/portfolio",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_value"] == 10000.0
            assert data["position_count"] == 2


@pytest.mark.asyncio
async def test_websocket_connection_manager():
    """Test WebSocket connection manager."""
    manager = ConnectionManager()
    
    # Mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.accept = AsyncMock()
    mock_ws.send_text = AsyncMock()
    
    # Connect client
    await manager.connect(mock_ws, "client-1")
    assert "client-1" in manager.active_connections
    assert "client-1" in manager.subscriptions
    
    # Subscribe to channel
    manager.subscribe("client-1", "market:BTC/USDT")
    assert "market:BTC/USDT" in manager.subscriptions["client-1"]
    
    # Send personal message
    await manager.send_personal_message("test message", "client-1")
    mock_ws.send_text.assert_called_once_with("test message")
    
    # Broadcast to channel
    await manager.broadcast("broadcast message", "market:BTC/USDT")
    mock_ws.send_text.assert_called_with("broadcast message")
    
    # Disconnect
    manager.disconnect("client-1")
    assert "client-1" not in manager.active_connections


@pytest.mark.asyncio
async def test_websocket_manager_handle_connection(redis_client):
    """Test WebSocket manager connection handling."""
    ws_manager = WebSocketManager(redis_client)
    
    # Mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.accept = AsyncMock()
    mock_ws.send_json = AsyncMock()
    mock_ws.receive_text = AsyncMock(side_effect=[
        json.dumps({"type": "subscribe", "channels": ["market:BTC/USDT"]}),
        json.dumps({"type": "ping"}),
        Exception("Disconnect")  # Simulate disconnect
    ])
    
    # Handle connection
    try:
        await ws_manager.handle_connection(mock_ws, "client-1")
    except:
        pass
    
    # Verify welcome message sent
    mock_ws.send_json.assert_any_call({
        "type": "connection",
        "status": "connected",
        "timestamp": pytest.approx(datetime.utcnow().isoformat(), rel=1),
        "client_id": "client-1"
    })
    
    # Verify subscription confirmed
    sent_messages = [call[0][0] for call in mock_ws.send_text.call_args_list]
    assert any("subscribed" in str(msg) for msg in sent_messages)


def test_market_data_endpoint(client, auth_headers, mock_market_data):
    """Test market data endpoint."""
    with patch("src.api.dependencies.get_market_stream") as mock_stream:
        mock_stream.return_value.get_market_data = AsyncMock(
            return_value=mock_market_data
        )
        
        response = client.get(
            "/api/v1/market/data?exchange=binance&symbol=BTC/USDT",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert data["bid"] == 45000.0


def test_performance_metrics_endpoint(client, auth_headers):
    """Test performance metrics endpoint."""
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine"):
            response = client.get(
                "/api/v1/trading/performance?days=30",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["period_days"] == 30
            assert "win_rate" in data
            assert "sharpe_ratio" in data


def test_close_all_positions_endpoint(client, auth_headers):
    """Test close all positions endpoint."""
    with patch("src.api.dependencies.get_current_user", return_value=MagicMock(id="test-user")):
        with patch("src.api.dependencies.get_trading_engine") as mock_engine:
            mock_engine.return_value._close_all_positions = AsyncMock()
            
            response = client.post(
                "/api/v1/trading/close-all",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "All positions closed" in data["message"]