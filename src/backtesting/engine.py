"""
Backtesting Engine Implementation

Provides the core backtesting engine that orchestrates historical simulations
of trading strategies with realistic market conditions.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Type
from enum import Enum
import pandas as pd
import numpy as np

from ..core.architecture import EventBus, injectable, inject
from ..core.observability import get_logger, MetricsCollector
from ..core.cache import CacheManager
from ..strategies.base_strategy import BaseStrategy
from ..orders.order_manager import OrderManager, Order
from ..positions.position_tracker import PositionTracker
from ..risk.risk_manager import RiskManager
from .data_provider import DataProvider
from .event_simulator import EventSimulator, MarketEvent
from .portfolio_simulator import PortfolioSimulator
from .execution_simulator import ExecutionSimulator
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics

logger = get_logger(__name__)class BacktestMode(Enum):
    """Backtesting execution modes"""
    TICK_BY_TICK = "tick_by_tick"
    BAR_BY_BAR = "bar_by_bar"
    EVENT_DRIVEN = "event_driven"
    VECTORIZED = "vectorized"

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    mode: BacktestMode = BacktestMode.BAR_BY_BAR
    timeframe: str = "1m"  # 1m, 5m, 15m, 1h, 1d
    commission: float = 0.001  # 0.1%
    slippage_model: str = "linear"  # linear, square_root, market_impact
    use_spread: bool = True
    spread_multiplier: float = 1.0
    allow_shorting: bool = True
    margin_ratio: float = 0.5  # 2:1 leverage
    use_historical_volatility: bool = True
    random_seed: Optional[int] = None
    warm_up_period: int = 100  # bars for indicator warm-up
    rebalance_frequency: Optional[str] = None  # daily, weekly, monthly
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 2% annual risk-free rate@dataclass
class BacktestResult:
    """Results from backtesting"""
    performance_metrics: PerformanceMetrics
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    drawdowns: pd.DataFrame
    returns: pd.DataFrame
    portfolio_values: pd.DataFrame
    execution_details: Dict[str, Any]
    strategy_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    benchmark_comparison: Optional[pd.DataFrame] = None

@injectable
class BacktestEngine:
    """
    Core backtesting engine that orchestrates historical simulations
    """
    
    @inject
    def __init__(
        self,
        event_bus: EventBus,
        metrics_collector: MetricsCollector,
        cache_manager: CacheManager
    ):
        self.event_bus = event_bus
        self.metrics = metrics_collector
        self.cache = cache_manager
        self.logger = logger        
        # Component instances
        self.data_provider: Optional[DataProvider] = None
        self.event_simulator: Optional[EventSimulator] = None
        self.portfolio_simulator: Optional[PortfolioSimulator] = None
        self.execution_simulator: Optional[ExecutionSimulator] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        
        # State
        self.config: Optional[BacktestConfig] = None
        self.strategy: Optional[BaseStrategy] = None
        self.is_running = False
        self.current_timestamp: Optional[datetime] = None
        self.processed_bars = 0
        self.total_bars = 0
        
    async def initialize(
        self,
        config: BacktestConfig,
        strategy: BaseStrategy,
        data_provider: DataProvider
    ):
        """Initialize backtesting components"""
        self.config = config
        self.strategy = strategy
        self.data_provider = data_provider
        
        # Set random seed for reproducibility
        if config.random_seed:
            np.random.seed(config.random_seed)
            
        # Initialize components
        self.event_simulator = EventSimulator(
            start_date=config.start_date,
            end_date=config.end_date,
            mode=config.mode
        )        
        self.portfolio_simulator = PortfolioSimulator(
            initial_capital=config.initial_capital,
            commission=config.commission,
            allow_shorting=config.allow_shorting,
            margin_ratio=config.margin_ratio
        )
        
        self.execution_simulator = ExecutionSimulator(
            slippage_model=config.slippage_model,
            use_spread=config.use_spread,
            spread_multiplier=config.spread_multiplier
        )
        
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Subscribe to events
        await self._setup_event_handlers()
        
        # Load historical data
        await self._load_historical_data()
        
        self.logger.info(
            f"Initialized backtest engine for {len(config.symbols)} symbols "
            f"from {config.start_date} to {config.end_date}"
        )
        
    async def _setup_event_handlers(self):
        """Setup event handlers for backtesting"""
        await self.event_bus.subscribe(
            "backtest.market_update",
            self._handle_market_update
        )        
        await self.event_bus.subscribe(
            "backtest.order_filled",
            self._handle_order_filled
        )
        
        await self.event_bus.subscribe(
            "backtest.position_updated",
            self._handle_position_updated
        )
        
    async def _load_historical_data(self):
        """Load and prepare historical data"""
        self.logger.info("Loading historical data...")
        
        # Load data for all symbols
        for symbol in self.config.symbols:
            await self.data_provider.load_data(
                symbol=symbol,
                start_date=self.config.start_date - timedelta(days=self.config.warm_up_period),
                end_date=self.config.end_date,
                timeframe=self.config.timeframe
            )
            
        # Calculate total bars to process
        self.total_bars = await self.data_provider.get_total_bars()
        
        # Load benchmark data if specified
        if self.config.benchmark_symbol:
            await self.data_provider.load_data(
                symbol=self.config.benchmark_symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                timeframe=self.config.timeframe
            )            
    async def run(self) -> BacktestResult:
        """Run the backtest simulation"""
        if not all([self.config, self.strategy, self.data_provider]):
            raise ValueError("Engine not properly initialized")
            
        self.is_running = True
        self.logger.info("Starting backtest simulation...")
        
        try:
            # Initialize strategy
            await self.strategy.initialize()
            
            # Run simulation based on mode
            if self.config.mode == BacktestMode.TICK_BY_TICK:
                await self._run_tick_simulation()
            elif self.config.mode == BacktestMode.BAR_BY_BAR:
                await self._run_bar_simulation()
            elif self.config.mode == BacktestMode.EVENT_DRIVEN:
                await self._run_event_simulation()
            elif self.config.mode == BacktestMode.VECTORIZED:
                await self._run_vectorized_simulation()
                
            # Finalize positions
            await self._close_all_positions()
            
            # Calculate performance metrics
            result = await self._generate_results()
            
            self.logger.info("Backtest simulation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise        finally:
            self.is_running = False
            
    async def _run_bar_simulation(self):
        """Run bar-by-bar simulation"""
        # Get data iterator
        async for timestamp, market_data in self.data_provider.iter_bars():
            self.current_timestamp = timestamp
            self.processed_bars += 1
            
            # Skip warm-up period
            if self.processed_bars <= self.config.warm_up_period:
                continue
                
            # Update portfolio with latest prices
            await self.portfolio_simulator.update_market_data(market_data)
            
            # Generate market event
            event = MarketEvent(
                timestamp=timestamp,
                data=market_data,
                symbols=self.config.symbols
            )
            
            # Process event through strategy
            signals = await self.strategy.on_market_data(event)
            
            # Execute signals
            if signals:
                orders = await self._convert_signals_to_orders(signals)
                for order in orders:
                    await self._execute_order(order, market_data)
                    
            # Check for rebalancing
            if self.config.rebalance_frequency:
                await self._check_rebalancing(timestamp)                    
            # Record portfolio snapshot
            await self.portfolio_simulator.record_snapshot(timestamp)
            
            # Emit progress
            if self.processed_bars % 1000 == 0:
                progress = (self.processed_bars / self.total_bars) * 100
                await self.event_bus.publish("backtest.progress", {
                    "progress": progress,
                    "processed_bars": self.processed_bars,
                    "total_bars": self.total_bars,
                    "current_timestamp": timestamp
                })
                
    async def _execute_order(self, order: Order, market_data: Dict[str, Any]):
        """Execute order with simulated fills"""
        # Apply execution simulation
        fill = await self.execution_simulator.simulate_fill(
            order=order,
            market_data=market_data,
            timestamp=self.current_timestamp
        )
        
        if fill:
            # Update portfolio
            await self.portfolio_simulator.process_fill(fill)
            
            # Update position tracker
            await self.event_bus.publish("backtest.order_filled", fill)
            
            # Record trade
            await self._record_trade(fill)            
    async def _generate_results(self) -> BacktestResult:
        """Generate comprehensive backtest results"""
        # Get performance metrics
        performance_metrics = await self.performance_analyzer.calculate_metrics(
            portfolio=self.portfolio_simulator.get_portfolio_history(),
            benchmark=await self._get_benchmark_returns() if self.config.benchmark_symbol else None
        )
        
        # Get trade history
        trades_df = self.portfolio_simulator.get_trades_dataframe()
        
        # Get position history
        positions_df = self.portfolio_simulator.get_positions_dataframe()
        
        # Calculate equity curve
        equity_curve = self.portfolio_simulator.get_equity_curve()
        
        # Calculate returns
        returns = equity_curve['total_value'].pct_change().fillna(0)
        
        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(equity_curve['total_value'])
        
        # Get portfolio values over time
        portfolio_values = self.portfolio_simulator.get_portfolio_values()
        
        # Execution statistics
        execution_details = self.execution_simulator.get_statistics()
        
        # Strategy-specific metrics
        strategy_metrics = await self.strategy.get_metrics()        
        # Risk metrics
        risk_metrics = await self._calculate_risk_metrics(returns, positions_df)
        
        # Benchmark comparison if available
        benchmark_comparison = None
        if self.config.benchmark_symbol:
            benchmark_comparison = await self._generate_benchmark_comparison(
                returns, 
                await self._get_benchmark_returns()
            )
            
        return BacktestResult(
            performance_metrics=performance_metrics,
            equity_curve=equity_curve,
            trades=trades_df,
            positions=positions_df,
            drawdowns=drawdowns,
            returns=pd.DataFrame({'returns': returns}),
            portfolio_values=portfolio_values,
            execution_details=execution_details,
            strategy_metrics=strategy_metrics,
            risk_metrics=risk_metrics,
            benchmark_comparison=benchmark_comparison
        )
        
    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate drawdown statistics"""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max        
        # Find drawdown periods
        drawdown_starts = []
        drawdown_ends = []
        in_drawdown = False
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0 and not in_drawdown:
                drawdown_starts.append(i)
                in_drawdown = True
            elif drawdown.iloc[i] == 0 and in_drawdown:
                drawdown_ends.append(i)
                in_drawdown = False
                
        # If still in drawdown at end
        if in_drawdown:
            drawdown_ends.append(len(drawdown) - 1)
            
        # Create drawdown dataframe
        drawdown_periods = []
        for start, end in zip(drawdown_starts, drawdown_ends):
            period_drawdown = drawdown.iloc[start:end+1]
            drawdown_periods.append({
                'start_date': equity_curve.index[start],
                'end_date': equity_curve.index[end],
                'duration_days': (equity_curve.index[end] - equity_curve.index[start]).days,
                'max_drawdown': period_drawdown.min(),
                'recovery_days': end - start
            })
            
        return pd.DataFrame(drawdown_periods)    
    async def _calculate_risk_metrics(
        self, 
        returns: pd.Series, 
        positions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        # Basic statistics
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'win_rate': self._calculate_win_rate(positions),
            'profit_factor': self._calculate_profit_factor(positions),
            'average_win': positions[positions['pnl'] > 0]['pnl'].mean() if len(positions[positions['pnl'] > 0]) > 0 else 0,
            'average_loss': positions[positions['pnl'] < 0]['pnl'].mean() if len(positions[positions['pnl'] < 0]) > 0 else 0,
            'largest_win': positions['pnl'].max() if len(positions) > 0 else 0,
            'largest_loss': positions['pnl'].min() if len(positions) > 0 else 0,
            'avg_holding_period': self._calculate_avg_holding_period(positions),
            'trades_per_day': len(positions) / ((self.config.end_date - self.config.start_date).days),
        }
        
        return metrics    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
            
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
            
        excess_return = returns.mean() - (self.config.risk_free_rate / 252)
        return np.sqrt(252) * excess_return / downside_std
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
        
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        max_dd = abs(self._calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return 0.0
            
        return annual_return / max_dd        
    def _calculate_win_rate(self, positions: pd.DataFrame) -> float:
        """Calculate win rate from positions"""
        if positions.empty or 'pnl' not in positions.columns:
            return 0.0
            
        winning_positions = positions[positions['pnl'] > 0]
        return len(winning_positions) / len(positions) if len(positions) > 0 else 0.0
        
    def _calculate_profit_factor(self, positions: pd.DataFrame) -> float:
        """Calculate profit factor"""
        if positions.empty or 'pnl' not in positions.columns:
            return 0.0
            
        gross_profit = positions[positions['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(positions[positions['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
        
    def _calculate_avg_holding_period(self, positions: pd.DataFrame) -> float:
        """Calculate average holding period in days"""
        if positions.empty:
            return 0.0
            
        if 'opened_at' in positions.columns and 'closed_at' in positions.columns:
            closed_positions = positions[positions['closed_at'].notna()]
            if not closed_positions.empty:
                holding_periods = (closed_positions['closed_at'] - closed_positions['opened_at']).dt.days
                return holding_periods.mean()
                
        return 0.0    
    async def _run_tick_simulation(self):
        """Run tick-by-tick simulation - placeholder"""
        # This would implement tick-level simulation
        # For now, redirect to bar simulation
        await self._run_bar_simulation()
        
    async def _run_event_simulation(self):
        """Run event-driven simulation - placeholder"""
        # This would implement event-driven simulation
        # For now, redirect to bar simulation
        await self._run_bar_simulation()
        
    async def _run_vectorized_simulation(self):
        """Run vectorized simulation - placeholder"""
        # This would implement vectorized backtesting
        # For now, redirect to bar simulation
        await self._run_bar_simulation()
        
    async def _convert_signals_to_orders(self, signals: List[Dict[str, Any]]) -> List[Order]:
        """Convert strategy signals to orders"""
        orders = []
        
        for signal in signals:
            # Create order from signal
            order = Order(
                symbol=signal['symbol'],
                side=signal['side'],
                quantity=signal['quantity'],
                type=signal.get('order_type', OrderType.MARKET),
                price=signal.get('price'),
                stop_price=signal.get('stop_price')
            )
            orders.append(order)
            
        return orders        
    async def _record_trade(self, fill: Dict[str, Any]):
        """Record executed trade"""
        # This would normally record to a trade database
        # For now, just log it
        self.logger.info(f"Trade executed: {fill}")
        
    async def _check_rebalancing(self, timestamp: datetime):
        """Check if portfolio needs rebalancing"""
        if not self.config.rebalance_frequency:
            return
            
        # Check if it's time to rebalance
        should_rebalance = False
        
        if self.config.rebalance_frequency == 'daily':
            should_rebalance = True
        elif self.config.rebalance_frequency == 'weekly' and timestamp.weekday() == 0:
            should_rebalance = True
        elif self.config.rebalance_frequency == 'monthly' and timestamp.day == 1:
            should_rebalance = True
            
        if should_rebalance:
            await self.event_bus.publish("backtest.rebalance", {
                'timestamp': timestamp,
                'portfolio_value': self.portfolio_simulator.get_total_value()
            })
            
    async def _close_all_positions(self):
        """Close all open positions at end of backtest"""
        # Get current positions from portfolio simulator
        positions = self.portfolio_simulator.portfolio.positions
        
        for symbol, position in positions.items():
            if position.quantity != 0:
                # Create closing order
                order = Order(
                    symbol=symbol,
                    side='sell' if position.quantity > 0 else 'buy',
                    quantity=abs(position.quantity),
                    type=OrderType.MARKET
                )                
                # Execute at last known price
                last_price = position.current_price
                await self._execute_order(order, {symbol: {'close': last_price}})
                
    async def _get_benchmark_returns(self) -> pd.Series:
        """Get benchmark returns if available"""
        if not self.config.benchmark_symbol:
            return pd.Series()
            
        # Get benchmark data from data provider
        benchmark_data = self.data_provider.data_cache.get(self.config.benchmark_symbol)
        
        if benchmark_data is None or benchmark_data.empty:
            return pd.Series()
            
        # Calculate returns
        returns = benchmark_data['close'].pct_change().dropna()
        return returns
        
    async def _generate_benchmark_comparison(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> pd.DataFrame:
        """Generate benchmark comparison statistics"""
        # Align returns
        aligned_returns = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if aligned_returns.empty:
            return pd.DataFrame()
            
        # Calculate relative metrics
        relative_returns = aligned_returns['strategy'] - aligned_returns['benchmark']
        tracking_error = relative_returns.std() * np.sqrt(252)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'strategy_return': (1 + aligned_returns['strategy']).cumprod() - 1,
            'benchmark_return': (1 + aligned_returns['benchmark']).cumprod() - 1,
            'relative_return': (1 + relative_returns).cumprod() - 1,
            'tracking_error': tracking_error
        })
        
        return comparison    
    async def _handle_market_update(self, event: Dict[str, Any]):
        """Handle market update events"""
        # Process market data update
        pass
        
    async def _handle_order_filled(self, event: Dict[str, Any]):
        """Handle order filled events"""
        # Process order fill
        pass
        
    async def _handle_position_updated(self, event: Dict[str, Any]):
        """Handle position update events"""
        # Process position update
        pass