"""
Performance Analyzer Module

Core module for analyzing trading performance with support for multiple
timeframes, strategy comparison, and comprehensive metrics calculation.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

from ..core.interfaces import Injectable
from ..core.decorators import injectable
from ..core.logger import get_logger
from ..core.event_bus import EventBus, Event
from ..models.trading import Trade, Position, Order
from ..models.portfolio import Portfolio
from .metrics_calculator import MetricsCalculator, PerformanceMetrics
from .portfolio_analytics import PortfolioAnalytics
from .trade_analytics import TradeAnalytics
from .risk_analytics import RiskAnalytics

logger = get_logger(__name__)


class AnalysisTimeframe(Enum):
    """Supported analysis timeframes"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


class AnalysisType(Enum):
    """Types of analysis available"""
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    SYMBOL = "symbol"
    EXCHANGE = "exchange"
    TIME_OF_DAY = "time_of_day"
    MARKET_REGIME = "market_regime"


@dataclass
class AnalysisConfig:
    """Configuration for performance analysis"""
    analyze_strategies: bool = True
    analyze_symbols: bool = True
    analyze_exchanges: bool = True
    analyze_time_patterns: bool = True
    calculate_attribution: bool = True
    calculate_benchmarks: bool = True
    benchmark_symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "SPY"])
    risk_free_rate: float = 0.02  # 2% annual
    confidence_level: float = 0.95
    lookback_periods: Dict[AnalysisTimeframe, int] = field(default_factory=lambda: {
        AnalysisTimeframe.DAILY: 30,
        AnalysisTimeframe.WEEKLY: 52,
        AnalysisTimeframe.MONTHLY: 12,
        AnalysisTimeframe.QUARTERLY: 8,
        AnalysisTimeframe.YEARLY: 5
    })


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    high_water_mark: float
    volatility: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    trades_count: int
    open_positions: int


@dataclass
class StrategyPerformance:
    """Performance metrics for a specific strategy"""
    strategy_id: str
    strategy_name: str
    metrics: PerformanceMetrics
    equity_curve: List[PerformanceSnapshot]
    trade_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    attribution: Dict[str, float]


@injectable
class PerformanceAnalyzer(Injectable):
    """Analyzes trading performance across multiple dimensions"""
    
    def __init__(
        self,
        config: AnalysisConfig = None,
        metrics_calculator: MetricsCalculator = None,
        portfolio_analytics: PortfolioAnalytics = None,
        trade_analytics: TradeAnalytics = None,
        risk_analytics: RiskAnalytics = None,
        event_bus: EventBus = None
    ):
        self.config = config or AnalysisConfig()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.portfolio_analytics = portfolio_analytics or PortfolioAnalytics()
        self.trade_analytics = trade_analytics or TradeAnalytics()
        self.risk_analytics = risk_analytics or RiskAnalytics()
        self.event_bus = event_bus
        
        # Performance data storage
        self._snapshots: List[PerformanceSnapshot] = []
        self._trades: List[Trade] = []
        self._positions: Dict[str, Position] = {}
        self._portfolios: List[Portfolio] = []
        
        # Analysis results cache
        self._analysis_cache: Dict[str, Any] = {}
        self._last_analysis_time: Optional[datetime] = None
        
        # Real-time tracking
        self._high_water_mark = 0.0
        self._current_drawdown = 0.0
        self._daily_returns: List[float] = []
        
    async def analyze(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: AnalysisTimeframe = AnalysisTimeframe.DAILY,
        analysis_types: List[AnalysisType] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            timeframe: Analysis timeframe
            analysis_types: Specific analyses to perform
            
        Returns:
            Comprehensive analysis results
        """
        if not analysis_types:
            analysis_types = list(AnalysisType)
        
        # Filter data by date range
        filtered_trades = self._filter_trades_by_date(start_date, end_date)
        filtered_snapshots = self._filter_snapshots_by_date(start_date, end_date)
        
        # Perform requested analyses
        results = {
            "summary": await self._analyze_summary(filtered_trades, filtered_snapshots),
            "timeframe": timeframe.value,
            "period": {
                "start": start_date or (filtered_trades[0].timestamp if filtered_trades else None),
                "end": end_date or datetime.utcnow()
            }
        }
        
        # Portfolio analysis
        if AnalysisType.PORTFOLIO in analysis_types:
            results["portfolio"] = await self._analyze_portfolio(
                filtered_trades, filtered_snapshots, timeframe
            )
        
        # Strategy analysis
        if AnalysisType.STRATEGY in analysis_types and self.config.analyze_strategies:
            results["strategies"] = await self._analyze_strategies(
                filtered_trades, filtered_snapshots
            )
        
        # Symbol analysis
        if AnalysisType.SYMBOL in analysis_types and self.config.analyze_symbols:
            results["symbols"] = await self._analyze_symbols(
                filtered_trades, filtered_snapshots
            )
        
        # Exchange analysis
        if AnalysisType.EXCHANGE in analysis_types and self.config.analyze_exchanges:
            results["exchanges"] = await self._analyze_exchanges(
                filtered_trades, filtered_snapshots
            )
        
        # Time pattern analysis
        if AnalysisType.TIME_OF_DAY in analysis_types and self.config.analyze_time_patterns:
            results["time_patterns"] = await self._analyze_time_patterns(
                filtered_trades
            )
        
        # Market regime analysis
        if AnalysisType.MARKET_REGIME in analysis_types:
            results["market_regimes"] = await self._analyze_market_regimes(
                filtered_snapshots
            )
        
        # Performance attribution
        if self.config.calculate_attribution:
            results["attribution"] = await self._calculate_attribution(
                filtered_trades, filtered_snapshots
            )
        
        # Benchmark comparison
        if self.config.calculate_benchmarks:
            results["benchmarks"] = await self._compare_benchmarks(
                filtered_snapshots
            )
        
        # Cache results
        self._analysis_cache[self._get_cache_key(start_date, end_date, timeframe)] = results
        self._last_analysis_time = datetime.utcnow()
        
        # Publish analysis event
        if self.event_bus:
            await self.event_bus.publish(Event(
                type="performance.analysis.completed",
                data={"timeframe": timeframe.value, "types": [t.value for t in analysis_types]}
            ))
        
        return results
    
    async def add_trade(self, trade: Trade):
        """Add trade to analysis"""
        self._trades.append(trade)
        
        # Update real-time metrics
        await self._update_real_time_metrics(trade)
        
        # Invalidate cache
        self._analysis_cache.clear()
    
    async def update_position(self, position: Position):
        """Update position for analysis"""
        self._positions[position.position_id] = position
    
    async def snapshot_performance(self, portfolio: Portfolio) -> PerformanceSnapshot:
        """Create performance snapshot from portfolio state"""
        snapshot = await self._create_snapshot(portfolio)
        self._snapshots.append(snapshot)
        
        # Update high water mark and drawdown
        if snapshot.equity > self._high_water_mark:
            self._high_water_mark = snapshot.equity
        
        self._current_drawdown = (self._high_water_mark - snapshot.equity) / self._high_water_mark
        
        return snapshot
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        if not self._snapshots:
            return {}
        
        latest_snapshot = self._snapshots[-1]
        
        return {
            "equity": latest_snapshot.equity,
            "unrealized_pnl": latest_snapshot.unrealized_pnl,
            "realized_pnl": latest_snapshot.realized_pnl,
            "daily_return": latest_snapshot.daily_return,
            "cumulative_return": latest_snapshot.cumulative_return,
            "drawdown": self._current_drawdown,
            "sharpe_ratio": latest_snapshot.sharpe_ratio,
            "win_rate": latest_snapshot.win_rate,
            "trades_today": self._count_trades_today(),
            "open_positions": latest_snapshot.open_positions
        }
    
    async def _analyze_summary(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Generate summary analysis"""
        if not snapshots:
            return {}
        
        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]
        
        # Calculate overall metrics
        total_return = (last_snapshot.equity - first_snapshot.equity) / first_snapshot.equity
        
        returns = [s.daily_return for s in snapshots if s.daily_return is not None]
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        sharpe = self.metrics_calculator.calculate_sharpe_ratio(
            returns, self.config.risk_free_rate
        )
        
        max_drawdown = max(s.drawdown for s in snapshots) if snapshots else 0
        
        # Trade statistics
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.realized_pnl) for t in losing_trades]) if losing_trades else 0
        
        profit_factor = (
            sum(t.realized_pnl for t in winning_trades) / 
            abs(sum(t.realized_pnl for t in losing_trades))
        ) if losing_trades else float('inf')
        
        return {
            "total_return": total_return,
            "total_pnl": last_snapshot.realized_pnl + last_snapshot.unrealized_pnl,
            "realized_pnl": last_snapshot.realized_pnl,
            "unrealized_pnl": last_snapshot.unrealized_pnl,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "trades_count": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": np.mean([t.realized_pnl for t in trades]) if trades else 0,
            "best_trade": max(t.realized_pnl for t in trades) if trades else 0,
            "worst_trade": min(t.realized_pnl for t in trades) if trades else 0
        }
    
    async def _analyze_portfolio(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot],
        timeframe: AnalysisTimeframe
    ) -> Dict[str, Any]:
        """Analyze portfolio performance"""
        # Delegate to portfolio analytics
        return await self.portfolio_analytics.analyze(
            trades, snapshots, timeframe, self.config
        )
    
    async def _analyze_strategies(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, StrategyPerformance]:
        """Analyze performance by strategy"""
        strategies = {}
        
        # Group trades by strategy
        trades_by_strategy = defaultdict(list)
        for trade in trades:
            trades_by_strategy[trade.strategy_id].append(trade)
        
        # Analyze each strategy
        for strategy_id, strategy_trades in trades_by_strategy.items():
            # Calculate strategy-specific metrics
            metrics = await self.metrics_calculator.calculate_metrics(strategy_trades)
            
            # Get trade analysis
            trade_analysis = await self.trade_analytics.analyze_trades(strategy_trades)
            
            # Get risk analysis
            risk_analysis = await self.risk_analytics.analyze_risk(
                strategy_trades, snapshots
            )
            
            # Calculate attribution
            attribution = await self._calculate_strategy_attribution(
                strategy_id, strategy_trades, snapshots
            )
            
            strategies[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                strategy_name=f"Strategy {strategy_id}",  # TODO: Get from strategy registry
                metrics=metrics,
                equity_curve=[],  # TODO: Build strategy-specific equity curve
                trade_analysis=trade_analysis,
                risk_analysis=risk_analysis,
                attribution=attribution
            )
        
        return strategies
    
    async def _analyze_symbols(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by symbol"""
        symbols = {}
        
        # Group trades by symbol
        trades_by_symbol = defaultdict(list)
        for trade in trades:
            trades_by_symbol[trade.symbol].append(trade)
        
        # Analyze each symbol
        for symbol, symbol_trades in trades_by_symbol.items():
            metrics = await self.metrics_calculator.calculate_metrics(symbol_trades)
            
            symbols[symbol] = {
                "metrics": metrics.__dict__,
                "trades_count": len(symbol_trades),
                "total_volume": sum(t.quantity for t in symbol_trades),
                "avg_trade_size": np.mean([t.quantity for t in symbol_trades]),
                "avg_holding_time": self._calculate_avg_holding_time(symbol_trades)
            }
        
        return symbols
    
    async def _analyze_exchanges(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by exchange"""
        exchanges = {}
        
        # Group trades by exchange
        trades_by_exchange = defaultdict(list)
        for trade in trades:
            trades_by_exchange[trade.exchange].append(trade)
        
        # Analyze each exchange
        for exchange, exchange_trades in trades_by_exchange.items():
            metrics = await self.metrics_calculator.calculate_metrics(exchange_trades)
            
            exchanges[exchange] = {
                "metrics": metrics.__dict__,
                "trades_count": len(exchange_trades),
                "total_fees": sum(t.fees for t in exchange_trades),
                "avg_execution_time": self._calculate_avg_execution_time(exchange_trades),
                "slippage_analysis": await self._analyze_slippage(exchange_trades)
            }
        
        return exchanges
    
    async def _analyze_time_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance patterns by time"""
        patterns = {
            "hour_of_day": defaultdict(list),
            "day_of_week": defaultdict(list),
            "day_of_month": defaultdict(list),
            "month_of_year": defaultdict(list)
        }
        
        # Group trades by time periods
        for trade in trades:
            patterns["hour_of_day"][trade.timestamp.hour].append(trade.realized_pnl)
            patterns["day_of_week"][trade.timestamp.weekday()].append(trade.realized_pnl)
            patterns["day_of_month"][trade.timestamp.day].append(trade.realized_pnl)
            patterns["month_of_year"][trade.timestamp.month].append(trade.realized_pnl)
        
        # Calculate statistics for each period
        results = {}
        for pattern_type, pattern_data in patterns.items():
            results[pattern_type] = {}
            for period, pnls in pattern_data.items():
                results[pattern_type][period] = {
                    "avg_pnl": np.mean(pnls),
                    "total_pnl": sum(pnls),
                    "trades_count": len(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
                }
        
        return results
    
    async def _analyze_market_regimes(
        self,
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze performance in different market regimes"""
        if not snapshots:
            return {}
        
        # Simple regime detection based on volatility
        volatilities = [s.volatility for s in snapshots if s.volatility]
        if not volatilities:
            return {}
        
        low_vol_threshold = np.percentile(volatilities, 33)
        high_vol_threshold = np.percentile(volatilities, 67)
        
        regimes = {
            "low_volatility": [],
            "medium_volatility": [],
            "high_volatility": []
        }
        
        # Classify snapshots by regime
        for snapshot in snapshots:
            if snapshot.volatility <= low_vol_threshold:
                regimes["low_volatility"].append(snapshot)
            elif snapshot.volatility >= high_vol_threshold:
                regimes["high_volatility"].append(snapshot)
            else:
                regimes["medium_volatility"].append(snapshot)
        
        # Analyze performance in each regime
        results = {}
        for regime, regime_snapshots in regimes.items():
            if regime_snapshots:
                returns = [s.daily_return for s in regime_snapshots if s.daily_return]
                results[regime] = {
                    "avg_return": np.mean(returns) if returns else 0,
                    "volatility": np.std(returns) if returns else 0,
                    "sharpe_ratio": self.metrics_calculator.calculate_sharpe_ratio(
                        returns, self.config.risk_free_rate
                    ),
                    "days_count": len(regime_snapshots),
                    "percentage_of_time": len(regime_snapshots) / len(snapshots)
                }
        
        return results
    
    async def _calculate_attribution(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, float]:
        """Calculate performance attribution"""
        total_pnl = sum(t.realized_pnl for t in trades)
        
        attribution = {
            "selection": 0.0,  # Stock/asset selection
            "timing": 0.0,     # Market timing
            "sizing": 0.0,     # Position sizing
            "costs": 0.0       # Transaction costs
        }
        
        # Simple attribution model
        if total_pnl != 0:
            # Selection: Performance from choosing specific assets
            winning_symbols = defaultdict(float)
            for trade in trades:
                winning_symbols[trade.symbol] += trade.realized_pnl
            
            best_symbols = sorted(winning_symbols.values(), reverse=True)[:3]
            attribution["selection"] = sum(best_symbols) / total_pnl if best_symbols else 0
            
            # Timing: Performance from entry/exit timing
            well_timed_trades = [
                t for t in trades 
                if t.realized_pnl > 0 and t.holding_time < timedelta(days=1)
            ]
            attribution["timing"] = (
                sum(t.realized_pnl for t in well_timed_trades) / total_pnl
            ) if well_timed_trades else 0
            
            # Costs: Impact of transaction costs
            total_fees = sum(t.fees for t in trades)
            attribution["costs"] = -total_fees / total_pnl if total_fees else 0
            
            # Sizing: Remainder
            attribution["sizing"] = 1.0 - sum(attribution.values())
        
        return attribution
    
    async def _compare_benchmarks(
        self,
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare performance against benchmarks"""
        benchmarks = {}
        
        # TODO: Fetch benchmark data for comparison
        # For now, return placeholder
        for symbol in self.config.benchmark_symbols:
            benchmarks[symbol] = {
                "correlation": 0.0,
                "beta": 0.0,
                "alpha": 0.0,
                "tracking_error": 0.0,
                "information_ratio": 0.0
            }
        
        return benchmarks
    
    async def _create_snapshot(self, portfolio: Portfolio) -> PerformanceSnapshot:
        """Create performance snapshot from portfolio"""
        # Calculate returns
        daily_return = 0.0
        if self._snapshots:
            prev_equity = self._snapshots[-1].equity
            daily_return = (portfolio.total_value - prev_equity) / prev_equity
            self._daily_returns.append(daily_return)
        
        # Calculate cumulative return
        initial_equity = self._snapshots[0].equity if self._snapshots else portfolio.total_value
        cumulative_return = (portfolio.total_value - initial_equity) / initial_equity
        
        # Calculate volatility (annualized)
        volatility = 0.0
        if len(self._daily_returns) > 1:
            volatility = np.std(self._daily_returns) * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = 0.0
        if volatility > 0:
            excess_return = (
                np.mean(self._daily_returns) * 252 - self.config.risk_free_rate
            )
            sharpe_ratio = excess_return / volatility
        
        # Calculate win rate and profit factor
        recent_trades = self._trades[-100:]  # Last 100 trades
        winning_trades = [t for t in recent_trades if t.realized_pnl > 0]
        win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0
        
        profit_factor = 1.0
        if recent_trades:
            gross_profit = sum(t.realized_pnl for t in winning_trades)
            gross_loss = abs(sum(t.realized_pnl for t in recent_trades if t.realized_pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            equity=portfolio.total_value,
            cash=portfolio.cash,
            positions_value=portfolio.positions_value,
            unrealized_pnl=portfolio.unrealized_pnl,
            realized_pnl=portfolio.realized_pnl,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=self._current_drawdown,
            high_water_mark=self._high_water_mark,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=len(self._trades),
            open_positions=len(portfolio.positions)
        )
    
    async def _update_real_time_metrics(self, trade: Trade):
        """Update real-time performance metrics"""
        # TODO: Implement real-time metric updates
        pass
    
    def _filter_trades_by_date(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Trade]:
        """Filter trades by date range"""
        filtered = self._trades
        
        if start_date:
            filtered = [t for t in filtered if t.timestamp >= start_date]
        
        if end_date:
            filtered = [t for t in filtered if t.timestamp <= end_date]
        
        return filtered
    
    def _filter_snapshots_by_date(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[PerformanceSnapshot]:
        """Filter snapshots by date range"""
        filtered = self._snapshots
        
        if start_date:
            filtered = [s for s in filtered if s.timestamp >= start_date]
        
        if end_date:
            filtered = [s for s in filtered if s.timestamp <= end_date]
        
        return filtered
    
    def _calculate_avg_holding_time(self, trades: List[Trade]) -> float:
        """Calculate average holding time in hours"""
        holding_times = [t.holding_time.total_seconds() / 3600 for t in trades if t.holding_time]
        return np.mean(holding_times) if holding_times else 0
    
    def _calculate_avg_execution_time(self, trades: List[Trade]) -> float:
        """Calculate average execution time in milliseconds"""
        # TODO: Implement based on order execution times
        return 0.0
    
    async def _analyze_slippage(self, trades: List[Trade]) -> Dict[str, float]:
        """Analyze slippage statistics"""
        # TODO: Implement slippage analysis
        return {
            "avg_slippage": 0.0,
            "max_slippage": 0.0,
            "slippage_cost": 0.0
        }
    
    async def _calculate_strategy_attribution(
        self,
        strategy_id: str,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, float]:
        """Calculate attribution for a specific strategy"""
        # TODO: Implement strategy-specific attribution
        return {
            "selection": 0.0,
            "timing": 0.0,
            "sizing": 0.0,
            "costs": 0.0
        }
    
    def _count_trades_today(self) -> int:
        """Count trades executed today"""
        today = datetime.utcnow().date()
        return sum(1 for t in self._trades if t.timestamp.date() == today)
    
    def _get_cache_key(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        timeframe: AnalysisTimeframe
    ) -> str:
        """Generate cache key for analysis results"""
        start = start_date.isoformat() if start_date else "all"
        end = end_date.isoformat() if end_date else "now"
        return f"{start}_{end}_{timeframe.value}"