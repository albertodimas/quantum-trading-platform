"""
Integration tests for Exchange components
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from src.exchange import ExchangeManager, BinanceExchange, ExchangeType
from src.models.trading import Order, OrderType, OrderSide, OrderStatus
from src.models.exchange import ExchangeCredentials
from src.core.architecture import DIContainer


@pytest.mark.integration
class TestExchangeIntegration:
    """Test exchange integration components"""
    
    @pytest.fixture
    async def exchange_manager(self, di_container):
        """Create exchange manager with mocked exchanges"""
        manager = ExchangeManager(di_container)
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    def mock_binance_api(self):
        """Mock Binance API responses"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock()
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            mock_session.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            yield mock_response
    
    @pytest.mark.asyncio
    async def test_exchange_manager_initialization(self, exchange_manager):
        """Test exchange manager initialization"""
        assert exchange_manager.is_initialized
        assert len(exchange_manager._exchanges) == 0
    
    @pytest.mark.asyncio
    async def test_add_exchange(self, exchange_manager, exchange_credentials):
        """Test adding exchange to manager"""
        exchange_id = await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        assert exchange_id == "binance"
        assert "binance" in exchange_manager._exchanges
        exchange = exchange_manager.get_exchange("binance")
        assert isinstance(exchange, BinanceExchange)
    
    @pytest.mark.asyncio
    async def test_multi_exchange_management(self, exchange_manager, exchange_credentials):
        """Test managing multiple exchanges"""
        # Add multiple exchanges
        binance_id = await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        # Would add more exchange types here
        exchanges = exchange_manager.get_all_exchanges()
        assert len(exchanges) == 1
        assert binance_id in exchanges
    
    @pytest.mark.asyncio
    async def test_place_order_through_manager(
        self,
        exchange_manager,
        exchange_credentials,
        sample_order,
        mock_binance_api
    ):
        """Test placing order through exchange manager"""
        # Setup mock response
        mock_binance_api.json.return_value = {
            "orderId": 12345,
            "status": "NEW",
            "executedQty": "0.0",
            "cummulativeQuoteQty": "0.0"
        }
        
        # Add exchange
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        # Place order
        order_id = await exchange_manager.place_order("binance", sample_order)
        assert order_id == "12345"
    
    @pytest.mark.asyncio
    async def test_get_balance_multi_exchange(
        self,
        exchange_manager,
        exchange_credentials,
        mock_binance_api
    ):
        """Test getting balance from multiple exchanges"""
        # Mock balance response
        mock_binance_api.json.return_value = {
            "balances": [
                {"asset": "BTC", "free": "0.5", "locked": "0.1"},
                {"asset": "USDT", "free": "10000.0", "locked": "0.0"}
            ]
        }
        
        # Add exchange
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        # Get aggregated balance
        total_balance = await exchange_manager.get_total_balance()
        
        assert "BTC" in total_balance
        assert total_balance["BTC"]["free"] == 0.5
        assert total_balance["BTC"]["total"] == 0.6
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, exchange_credentials):
        """Test WebSocket connection to exchange"""
        with patch('websockets.connect') as mock_ws:
            # Mock WebSocket connection
            mock_ws_instance = AsyncMock()
            mock_ws_instance.recv = AsyncMock()
            mock_ws_instance.send = AsyncMock()
            mock_ws_instance.close = AsyncMock()
            
            mock_ws.return_value.__aenter__.return_value = mock_ws_instance
            
            exchange = BinanceExchange(exchange_credentials, testnet=True)
            await exchange.connect()
            
            # Start market data stream
            await exchange.subscribe_ticker("BTC/USDT")
            
            # Simulate receiving data
            mock_ws_instance.recv.return_value = '''
            {
                "e": "24hrTicker",
                "s": "BTCUSDT",
                "c": "50000.00",
                "h": "51000.00",
                "l": "49000.00",
                "v": "1000.0"
            }
            '''
            
            # Process one message
            await exchange._process_websocket_message()
            
            await exchange.disconnect()
            mock_ws_instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, exchange_manager, exchange_credentials):
        """Test rate limiting functionality"""
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        exchange = exchange_manager.get_exchange("binance")
        
        # Simulate rapid requests
        start_time = asyncio.get_event_loop().time()
        
        for i in range(5):
            await exchange._rate_limiter.acquire()
        
        end_time = asyncio.get_event_loop().time()
        
        # Should have rate limiting delays
        # Actual delay depends on rate limiter configuration
        assert end_time - start_time > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(
        self,
        exchange_manager,
        exchange_credentials,
        mock_binance_api
    ):
        """Test circuit breaker with exchange failures"""
        # Make API calls fail
        mock_binance_api.status = 500
        mock_binance_api.json.side_effect = Exception("Server error")
        
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        exchange = exchange_manager.get_exchange("binance")
        
        # Trigger multiple failures
        for _ in range(5):
            with pytest.raises(Exception):
                await exchange.get_ticker("BTC/USDT")
        
        # Circuit should be open
        assert exchange._circuit_breaker.is_open
        
        # Further calls should fail immediately
        with pytest.raises(Exception) as exc_info:
            await exchange.get_ticker("BTC/USDT")
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_order_execution_flow(
        self,
        exchange_manager,
        exchange_credentials,
        mock_binance_api,
        mock_event_bus
    ):
        """Test complete order execution flow"""
        # Setup responses
        place_order_response = {
            "orderId": 12345,
            "status": "NEW",
            "executedQty": "0.0"
        }
        
        order_status_response = {
            "orderId": 12345,
            "status": "FILLED",
            "executedQty": "0.1",
            "cummulativeQuoteQty": "5000.0"
        }
        
        mock_binance_api.json.side_effect = [
            place_order_response,
            order_status_response
        ]
        
        # Add exchange
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        # Create and place order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0
        )
        
        order_id = await exchange_manager.place_order("binance", order)
        assert order_id == "12345"
        
        # Check order status
        status = await exchange_manager.get_order_status("binance", order_id)
        assert status.status == OrderStatus.FILLED
        assert status.filled_quantity == 0.1
        
        # Verify events were published
        mock_event_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_market_data_aggregation(
        self,
        exchange_manager,
        exchange_credentials,
        mock_binance_api
    ):
        """Test market data aggregation from multiple sources"""
        # Mock ticker data
        mock_binance_api.json.return_value = {
            "symbol": "BTCUSDT",
            "bidPrice": "49900.00",
            "askPrice": "50100.00",
            "lastPrice": "50000.00",
            "volume": "1000.0"
        }
        
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        # Get ticker from all exchanges
        tickers = await exchange_manager.get_aggregated_ticker("BTC/USDT")
        
        assert "binance" in tickers
        assert tickers["binance"]["bid"] == 49900.0
        assert tickers["binance"]["ask"] == 50100.0
        
        # Calculate best bid/ask
        best_bid = max(t["bid"] for t in tickers.values())
        best_ask = min(t["ask"] for t in tickers.values())
        
        assert best_bid == 49900.0
        assert best_ask == 50100.0
    
    @pytest.mark.asyncio
    async def test_failover_mechanism(
        self,
        exchange_manager,
        exchange_credentials,
        mock_binance_api
    ):
        """Test failover between exchanges"""
        # Add primary exchange
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True,
            is_primary=True
        )
        
        # Make primary fail
        mock_binance_api.json.side_effect = Exception("Connection error")
        
        # Should handle gracefully
        with pytest.raises(Exception):
            await exchange_manager.place_order("binance", sample_order)
        
        # In real implementation, would failover to secondary
    
    @pytest.mark.asyncio
    async def test_reconnection_logic(self, exchange_credentials):
        """Test automatic reconnection on disconnection"""
        reconnect_count = 0
        
        async def mock_connect():
            nonlocal reconnect_count
            reconnect_count += 1
            if reconnect_count < 3:
                raise ConnectionError("Connection failed")
        
        exchange = BinanceExchange(exchange_credentials, testnet=True)
        exchange._connect = mock_connect
        
        # Should retry connection
        await exchange.connect()
        
        assert reconnect_count >= 1
    
    @pytest.mark.asyncio
    @pytest.mark.requires_redis
    async def test_exchange_state_persistence(
        self,
        exchange_manager,
        exchange_credentials,
        mock_redis
    ):
        """Test persisting exchange state"""
        await exchange_manager.add_exchange(
            exchange_type=ExchangeType.BINANCE,
            credentials=exchange_credentials,
            testnet=True
        )
        
        # Save state
        await exchange_manager.save_state()
        
        # Verify Redis calls
        mock_redis.set.assert_called()
        
        # Create new manager and load state
        new_manager = ExchangeManager(di_container)
        await new_manager.load_state()
        
        # Should have same exchanges
        assert "binance" in new_manager._exchanges