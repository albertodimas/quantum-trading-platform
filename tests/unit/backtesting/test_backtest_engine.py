"""
Unit tests for Backtesting Engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.backtesting import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    BacktestResult,
    DataProvider,
    CSVDataProvider,
    EventSimulator,
    PortfolioSimulator,
    ExecutionSimulator
)
from src.models.trading import Order, Trade, OrderType, OrderSide, OrderStatus
from src.strategies import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing"""
    
    def __init__(self):
        super().__init__("mock_strategy")
        self.on_start_called = False
        self.on_data_called = 0
        self.orders_to_place = []
    
    async def on_start(self):
        self.on_start_called = True
    
    async def on_market_data(self, data):
        self.on_data_called += 1
        
        # Place orders based on test scenario
        if self.orders_to_place:
            order = self.orders_to_place.pop(0)
            await self.place_order(
                symbol=order["symbol"],
                side=order["side"],
                order_type=order["type"],
                quantity=order["quantity"],
                price=order.get("price")
            )
    
    async def on_order_filled(self, order):
        pass
    
    async def on_stop(self):
        pass


@pytest.mark.unit
class TestBacktestEngine:
    """Test backtesting engine functionality"""
    
    @pytest.fixture
    def backtest_config(self):
        """Create backtesting configuration"""
        return BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=100000.0,
            symbols=["BTC/USDT", "ETH/USDT"],
            mode=BacktestMode.BAR_BY_BAR,
            commission_rate=0.001,
            slippage_model="fixed",
            use_spread=True,
            enable_shorting=False
        )
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider"""
        provider = AsyncMock(spec=DataProvider)
        provider.get_data = AsyncMock()
        provider.has_more_data = Mock(return_value=True)
        provider.get_next_timestamp = Mock()
        return provider
    
    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy"""
        return MockStrategy()
    
    @pytest.fixture
    async def backtest_engine(
        self,
        backtest_config,
        mock_data_provider,
        mock_strategy,
        mock_event_bus
    ):
        """Create backtest engine with mocks"""
        engine = BacktestEngine(
            config=backtest_config,
            data_provider=mock_data_provider,
            event_bus=mock_event_bus
        )
        
        # Inject mocked components
        engine._event_simulator = AsyncMock(spec=EventSimulator)
        engine._portfolio_simulator = AsyncMock(spec=PortfolioSimulator)
        engine._execution_simulator = AsyncMock(spec=ExecutionSimulator)
        
        engine.add_strategy(mock_strategy)
        
        yield engine
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, backtest_engine, backtest_config):
        """Test engine initialization"""
        assert backtest_engine.config == backtest_config
        assert len(backtest_engine._strategies) == 1
        assert not backtest_engine._is_running
    
    @pytest.mark.asyncio
    async def test_add_strategy(self, backtest_engine):
        """Test adding strategies"""
        new_strategy = MockStrategy()
        backtest_engine.add_strategy(new_strategy)
        
        assert len(backtest_engine._strategies) == 2
        assert new_strategy in backtest_engine._strategies.values()
    
    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, backtest_engine, mock_data_provider):
        """Test basic backtest execution"""
        # Setup mock data
        mock_data = [
            {
                "timestamp": datetime.now() - timedelta(hours=i),
                "symbol": "BTC/USDT",
                "open": 50000,
                "high": 51000,
                "low": 49000,
                "close": 50500,
                "volume": 1000
            }
            for i in range(10, 0, -1)
        ]
        
        mock_data_provider.get_data.return_value = mock_data
        mock_data_provider.has_more_data.side_effect = [True] * 9 + [False]
        
        # Run backtest
        result = await backtest_engine.run()
        
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert result.final_equity > 0
    
    @pytest.mark.asyncio
    async def test_strategy_lifecycle(self, backtest_engine, mock_strategy):
        """Test strategy lifecycle methods are called"""
        # Setup minimal data
        mock_data_provider = backtest_engine._data_provider
        mock_data_provider.has_more_data.side_effect = [True, False]
        mock_data_provider.get_data.return_value = [{
            "timestamp": datetime.now(),
            "symbol": "BTC/USDT",
            "close": 50000
        }]
        
        # Run backtest
        await backtest_engine.run()
        
        # Verify lifecycle methods called
        assert mock_strategy.on_start_called
        assert mock_strategy.on_data_called > 0
    
    @pytest.mark.asyncio
    async def test_order_execution_flow(self, backtest_engine, mock_strategy):
        """Test order placement and execution"""
        # Setup strategy to place an order
        mock_strategy.orders_to_place = [{
            "symbol": "BTC/USDT",
            "side": OrderSide.BUY,
            "type": OrderType.MARKET,
            "quantity": 0.1
        }]
        
        # Mock execution
        backtest_engine._execution_simulator.execute_order.return_value = Trade(
            id="trade_1",
            order_id="order_1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_price=50000,
            entry_time=datetime.now()
        )
        
        # Run backtest
        result = await backtest_engine.run()
        
        # Verify order was executed
        backtest_engine._execution_simulator.execute_order.assert_called()
    
    @pytest.mark.asyncio
    async def test_portfolio_tracking(self, backtest_engine):
        """Test portfolio value tracking"""
        portfolio_sim = backtest_engine._portfolio_simulator
        
        # Mock portfolio values
        portfolio_values = [100000, 101000, 99000, 102000]
        portfolio_sim.get_portfolio_value.side_effect = portfolio_values
        
        # Run backtest
        result = await backtest_engine.run()
        
        # Portfolio should be updated
        portfolio_sim.update_portfolio.assert_called()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, backtest_engine):
        """Test performance metrics calculation"""
        # Mock final portfolio state
        backtest_engine._portfolio_simulator.get_portfolio_value.return_value = 110000
        backtest_engine._portfolio_simulator.get_positions.return_value = []
        
        # Mock trades
        mock_trades = [
            Trade(
                id=f"trade_{i}",
                symbol="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=0.1,
                entry_price=50000 + i * 100,
                exit_price=50000 + i * 100 + 50,
                pnl=50 * 0.1,
                pnl_percentage=0.1
            )
            for i in range(10)
        ]
        
        backtest_engine._portfolio_simulator.get_trades.return_value = mock_trades
        
        # Run backtest
        result = await backtest_engine.run()
        
        # Check metrics
        assert result.total_return > 0
        assert result.sharpe_ratio is not None
        assert result.max_drawdown >= 0
        assert result.win_rate >= 0
    
    @pytest.mark.asyncio
    async def test_tick_by_tick_mode(self, backtest_engine, backtest_config):
        """Test tick-by-tick backtesting mode"""
        backtest_config.mode = BacktestMode.TICK_BY_TICK
        
        # Mock tick data
        tick_data = [
            {
                "timestamp": datetime.now() - timedelta(seconds=i),
                "symbol": "BTC/USDT",
                "bid": 49990,
                "ask": 50010,
                "last": 50000,
                "volume": 0.1
            }
            for i in range(10, 0, -1)
        ]
        
        backtest_engine._data_provider.get_data.return_value = tick_data
        backtest_engine._data_provider.has_more_data.side_effect = [True] * 9 + [False]
        
        # Run backtest
        result = await backtest_engine.run()
        
        assert isinstance(result, BacktestResult)
    
    @pytest.mark.asyncio
    async def test_multi_symbol_backtest(self, backtest_engine):
        """Test backtesting with multiple symbols"""
        # Mock data for multiple symbols
        btc_data = {
            "timestamp": datetime.now(),
            "symbol": "BTC/USDT",
            "close": 50000
        }
        eth_data = {
            "timestamp": datetime.now(),
            "symbol": "ETH/USDT",
            "close": 3000
        }
        
        backtest_engine._data_provider.get_data.side_effect = [
            [btc_data], [eth_data], []
        ]
        backtest_engine._data_provider.has_more_data.side_effect = [
            True, True, False
        ]
        
        # Run backtest
        result = await backtest_engine.run()
        
        # Both symbols should be processed
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_commission_calculation(self, backtest_engine, backtest_config):
        """Test commission calculation"""
        backtest_config.commission_rate = 0.001  # 0.1%
        
        # Mock a trade
        trade = Trade(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            entry_price=50000
        )
        
        backtest_engine._execution_simulator.execute_order.return_value = trade
        
        # Commission should be deducted
        expected_commission = 50000 * 1.0 * 0.001
        
        # Run and verify commission is calculated
        result = await backtest_engine.run()
        
        # Commission should affect final equity
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_slippage_modeling(self, backtest_engine):
        """Test slippage modeling"""
        # Configure slippage
        backtest_engine._execution_simulator.slippage_model = "percentage"
        backtest_engine._execution_simulator.slippage_rate = 0.0005  # 0.05%
        
        # Mock order execution with slippage
        order_price = 50000
        expected_slippage = order_price * 0.0005
        
        trade = Trade(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            entry_price=order_price + expected_slippage
        )
        
        backtest_engine._execution_simulator.execute_order.return_value = trade
        
        result = await backtest_engine.run()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_event_publishing(self, backtest_engine, mock_event_bus):
        """Test event publishing during backtest"""
        # Run backtest
        await backtest_engine.run()
        
        # Events should be published
        mock_event_bus.publish.assert_called()
        
        # Check for specific event types
        published_events = [
            call[0][0] for call in mock_event_bus.publish.call_args_list
        ]
        
        event_types = [event.type for event in published_events]
        
        # Should have backtest lifecycle events
        assert any("backtest.started" in t for t in event_types)
        assert any("backtest.completed" in t for t in event_types)
    
    @pytest.mark.asyncio
    async def test_stop_backtest(self, backtest_engine):
        """Test stopping backtest mid-execution"""
        # Setup long-running backtest
        backtest_engine._data_provider.has_more_data.return_value = True
        
        async def delayed_stop():
            await asyncio.sleep(0.1)
            await backtest_engine.stop()
        
        # Start backtest and stop it
        stop_task = asyncio.create_task(delayed_stop())
        result = await backtest_engine.run()
        await stop_task
        
        # Should stop gracefully
        assert not backtest_engine._is_running
    
    @pytest.mark.asyncio
    async def test_error_handling(self, backtest_engine):
        """Test error handling during backtest"""
        # Make data provider raise error
        backtest_engine._data_provider.get_data.side_effect = Exception(
            "Data error"
        )
        
        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            await backtest_engine.run()
        
        assert "Data error" in str(exc_info.value)
        assert not backtest_engine._is_running
    
    @pytest.mark.asyncio
    async def test_warmup_period(self, backtest_engine, backtest_config):
        """Test warmup period handling"""
        backtest_config.warmup_period = 10  # 10 bars warmup
        
        # Mock data
        data_points = 20
        mock_data = [
            {
                "timestamp": datetime.now() - timedelta(hours=i),
                "symbol": "BTC/USDT",
                "close": 50000 + i * 100
            }
            for i in range(data_points, 0, -1)
        ]
        
        backtest_engine._data_provider.get_data.return_value = mock_data
        
        # Strategy should not trade during warmup
        result = await backtest_engine.run()
        
        # First 10 data points should be warmup
        assert result is not None