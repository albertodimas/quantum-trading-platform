"""
Metrics Calculator Module

Calculates comprehensive trading performance metrics including returns,
risk measures, and efficiency ratios.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum

from ..core.logger import get_logger
from ..models.trading import Trade

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    RETURNS = "returns"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    DRAWDOWN = "drawdown"
    TRADE_ANALYSIS = "trade_analysis"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    monthly_return: float = 0.0
    daily_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    value_at_risk: float = 0.0
    conditional_var: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Trade analysis
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Efficiency metrics
    avg_trade_duration: float = 0.0
    trades_per_day: float = 0.0
    commission_ratio: float = 0.0
    slippage_ratio: float = 0.0
    
    # Statistical metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    hit_ratio: float = 0.0
    payoff_ratio: float = 0.0
    
    # Recovery metrics
    recovery_factor: float = 0.0
    avg_recovery_time: float = 0.0
    underwater_time: float = 0.0


class MetricsCalculator:
    """Calculates trading performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self._cache: Dict[str, Any] = {}
    
    async def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: Optional[List[float]] = None,
        benchmark_returns: Optional[List[float]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: List of completed trades
            equity_curve: Optional equity curve values
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Comprehensive performance metrics
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        # Calculate trade-based metrics
        self._calculate_trade_metrics(trades, metrics)
        
        # Calculate return metrics
        if equity_curve:
            self._calculate_return_metrics(equity_curve, metrics)
            self._calculate_risk_metrics(equity_curve, metrics)
            self._calculate_risk_adjusted_metrics(equity_curve, metrics, benchmark_returns)
            self._calculate_drawdown_metrics(equity_curve, metrics)
            self._calculate_statistical_metrics(equity_curve, metrics)
        
        # Calculate efficiency metrics
        self._calculate_efficiency_metrics(trades, metrics)
        
        return metrics
    
    def _calculate_trade_metrics(self, trades: List[Trade], metrics: PerformanceMetrics):
        """Calculate trade-based metrics"""
        metrics.total_trades = len(trades)
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        # Win rate
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # Average win/loss
        metrics.avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
        metrics.avg_loss = np.mean([abs(t.realized_pnl) for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.realized_pnl for t in winning_trades)
        total_losses = abs(sum(t.realized_pnl for t in losing_trades))
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win - 
            (1 - metrics.win_rate) * metrics.avg_loss
        )
        
        # Payoff ratio
        metrics.payoff_ratio = metrics.avg_win / metrics.avg_loss if metrics.avg_loss > 0 else float('inf')
        
        # Hit ratio (same as win rate but sometimes calculated differently)
        metrics.hit_ratio = metrics.win_rate
    
    def _calculate_return_metrics(self, equity_curve: List[float], metrics: PerformanceMetrics):
        """Calculate return-based metrics"""
        if len(equity_curve) < 2:
            return
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Total return
        metrics.total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Annualized return (assuming daily data)
        days = len(equity_curve) - 1
        years = days / 252
        metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Average returns
        metrics.daily_return = np.mean(returns)
        metrics.monthly_return = metrics.daily_return * 21  # Approximate trading days per month
    
    def _calculate_risk_metrics(self, equity_curve: List[float], metrics: PerformanceMetrics):
        """Calculate risk metrics"""
        if len(equity_curve) < 2:
            return
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Volatility (annualized)
        metrics.volatility = np.std(returns) * np.sqrt(252)
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        metrics.downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # Value at Risk (95% confidence)
        metrics.value_at_risk = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Conditional VaR (Expected Shortfall)
        var_threshold = metrics.value_at_risk
        tail_returns = returns[returns <= var_threshold]
        metrics.conditional_var = np.mean(tail_returns) if len(tail_returns) > 0 else 0
    
    def _calculate_risk_adjusted_metrics(
        self,
        equity_curve: List[float],
        metrics: PerformanceMetrics,
        benchmark_returns: Optional[List[float]] = None
    ):
        """Calculate risk-adjusted return metrics"""
        if len(equity_curve) < 2:
            return
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / 252
        metrics.sharpe_ratio = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        ) if np.std(excess_returns) > 0 else 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        metrics.sortino_ratio = (
            (np.mean(returns) - self.risk_free_rate / 252) / downside_std * np.sqrt(252)
        ) if downside_std > 0 else 0
        
        # Calmar Ratio
        metrics.calmar_ratio = (
            metrics.annualized_return / abs(metrics.max_drawdown)
        ) if metrics.max_drawdown != 0 else 0
        
        # Information Ratio (if benchmark provided)
        if benchmark_returns and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(252)
            metrics.information_ratio = (
                np.mean(active_returns) * 252 / tracking_error
            ) if tracking_error > 0 else 0
            
            # Treynor Ratio (using beta)
            if np.std(benchmark_returns) > 0:
                beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                metrics.treynor_ratio = (
                    (metrics.annualized_return - self.risk_free_rate) / beta
                ) if beta != 0 else 0
    
    def _calculate_drawdown_metrics(self, equity_curve: List[float], metrics: PerformanceMetrics):
        """Calculate drawdown metrics"""
        if len(equity_curve) < 2:
            return
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdowns
        drawdowns = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        metrics.max_drawdown = abs(np.min(drawdowns))
        
        # Maximum drawdown duration
        underwater_periods = self._get_underwater_periods(equity_curve)
        if underwater_periods:
            metrics.max_drawdown_duration = max(len(period) for period in underwater_periods)
            metrics.avg_recovery_time = np.mean([len(period) for period in underwater_periods])
        
        # Underwater time (percentage of time in drawdown)
        underwater_days = sum(len(period) for period in underwater_periods)
        metrics.underwater_time = underwater_days / len(equity_curve) if len(equity_curve) > 0 else 0
        
        # Recovery factor
        total_return = equity_curve[-1] - equity_curve[0]
        metrics.recovery_factor = total_return / abs(metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
    
    def _calculate_statistical_metrics(self, equity_curve: List[float], metrics: PerformanceMetrics):
        """Calculate statistical metrics"""
        if len(equity_curve) < 2:
            return
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Skewness
        metrics.skewness = stats.skew(returns)
        
        # Kurtosis
        metrics.kurtosis = stats.kurtosis(returns)
    
    def _calculate_efficiency_metrics(self, trades: List[Trade], metrics: PerformanceMetrics):
        """Calculate trading efficiency metrics"""
        if not trades:
            return
        
        # Average trade duration
        durations = []
        for trade in trades:
            if trade.holding_time:
                durations.append(trade.holding_time.total_seconds() / 3600)  # Hours
        
        metrics.avg_trade_duration = np.mean(durations) if durations else 0
        
        # Trades per day
        if trades:
            first_trade = min(trades, key=lambda t: t.timestamp)
            last_trade = max(trades, key=lambda t: t.timestamp)
            trading_days = (last_trade.timestamp - first_trade.timestamp).days + 1
            metrics.trades_per_day = len(trades) / trading_days if trading_days > 0 else 0
        
        # Commission ratio
        total_commission = sum(t.fees for t in trades)
        total_pnl = sum(t.realized_pnl for t in trades)
        metrics.commission_ratio = total_commission / abs(total_pnl) if total_pnl != 0 else 0
        
        # Slippage ratio
        total_slippage = sum(getattr(t, 'slippage', 0) for t in trades)
        metrics.slippage_ratio = total_slippage / abs(total_pnl) if total_pnl != 0 else 0
    
    def _get_underwater_periods(self, equity_curve: List[float]) -> List[List[int]]:
        """Get periods where equity is below running maximum"""
        running_max = np.maximum.accumulate(equity_curve)
        underwater = equity_curve < running_max
        
        periods = []
        current_period = []
        
        for i, is_underwater in enumerate(underwater):
            if is_underwater:
                current_period.append(i)
            else:
                if current_period:
                    periods.append(current_period)
                    current_period = []
        
        if current_period:
            periods.append(current_period)
        
        return periods
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float, target_return: float = 0) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return float('inf')
        
        return (np.mean(excess_returns) - target_return) / downside_deviation * np.sqrt(252)
    
    def calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if not returns or max_drawdown == 0:
            return 0.0
        
        annualized_return = self._annualize_return(returns)
        return annualized_return / abs(max_drawdown)
    
    def calculate_information_ratio(self, returns: List[float], benchmark_returns: List[float]) -> float:
        """Calculate Information ratio"""
        if not returns or not benchmark_returns or len(returns) != len(benchmark_returns):
            return 0.0
        
        active_returns = np.array(returns) - np.array(benchmark_returns)
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(active_returns) / tracking_error * np.sqrt(252)
    
    def calculate_maximum_drawdown(self, equity_curve: List[float]) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration
        
        Returns:
            Tuple of (max_drawdown, max_duration_days)
        """
        if len(equity_curve) < 2:
            return 0.0, 0
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        max_drawdown = abs(np.min(drawdowns))
        
        # Calculate duration
        underwater_periods = self._get_underwater_periods(equity_curve)
        max_duration = max(len(period) for period in underwater_periods) if underwater_periods else 0
        
        return max_drawdown, max_duration
    
    def _annualize_return(self, returns: List[float], periods_per_year: int = 252) -> float:
        """Annualize returns"""
        if not returns:
            return 0.0
        
        total_return = np.prod(1 + np.array(returns)) - 1
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0.0
        
        return (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    def calculate_rolling_metrics(
        self,
        returns: List[float],
        window: int = 252,
        metric: str = "sharpe"
    ) -> List[float]:
        """Calculate rolling metrics"""
        if len(returns) < window:
            return []
        
        rolling_values = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            
            if metric == "sharpe":
                value = self.calculate_sharpe_ratio(window_returns, self.risk_free_rate)
            elif metric == "volatility":
                value = np.std(window_returns) * np.sqrt(252)
            elif metric == "return":
                value = np.mean(window_returns) * 252
            else:
                value = 0.0
            
            rolling_values.append(value)
        
        return rolling_values