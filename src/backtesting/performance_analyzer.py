"""
Performance Analyzer for Backtesting

Calculates comprehensive performance metrics including returns, risk metrics,
drawdowns, and strategy-specific statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import scipy.stats as stats

from ..core.observability import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float    
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Statistical metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Consistency metrics
    positive_months: int = 0
    negative_months: int = 0
    best_month: float = 0.0
    worst_month: float = 0.0
    recovery_factor: float = 0.0
    profit_consistency: float = 0.0

class PerformanceAnalyzer:
    """
    Analyzes backtest results and calculates performance metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logger
        
    async def calculate_metrics(
        self,
        portfolio: List[Any],
        benchmark: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Convert portfolio history to DataFrame
        equity_curve = self._extract_equity_curve(portfolio)
        
        if equity_curve.empty:
            return self._empty_metrics()            
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Risk metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        drawdown_stats = self._calculate_drawdown_statistics(equity_curve)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(drawdown_stats['max_drawdown']) if drawdown_stats['max_drawdown'] != 0 else 0        
        # Trade analysis
        trade_stats = self._analyze_trades(portfolio)
        
        # Risk-adjusted metrics
        info_ratio = 0.0
        if benchmark is not None:
            tracking_error = (returns - benchmark).std() * np.sqrt(252)
            info_ratio = (annual_return - benchmark.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
        # Statistical metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Omega ratio
        threshold = self.risk_free_rate / 252
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')
        
        # Monthly statistics
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()        
        # Recovery factor
        recovery_factor = total_return / abs(drawdown_stats['max_drawdown']) if drawdown_stats['max_drawdown'] != 0 else 0
        
        # Profit consistency
        rolling_returns = returns.rolling(window=20).mean()
        profit_consistency = (rolling_returns > 0).sum() / len(rolling_returns) if len(rolling_returns) > 0 else 0
        
        return PerformanceMetrics(
            # Return metrics
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_returns.mean(),
            daily_return=returns.mean(),
            
            # Risk metrics
            volatility=annual_vol,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=drawdown_stats['max_drawdown'],
            max_drawdown_duration=drawdown_stats['max_duration'],
            
            # Trade metrics
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            average_win=trade_stats['average_win'],
            average_loss=trade_stats['average_loss'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            avg_trade_duration=trade_stats['avg_duration'],            
            # Risk-adjusted metrics
            information_ratio=info_ratio,
            omega_ratio=omega_ratio,
            
            # Statistical metrics
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            
            # Consistency metrics
            positive_months=positive_months,
            negative_months=negative_months,
            best_month=best_month,
            worst_month=worst_month,
            recovery_factor=recovery_factor,
            profit_consistency=profit_consistency
        )
        
    def _extract_equity_curve(self, portfolio: List[Any]) -> pd.Series:
        """Extract equity curve from portfolio history"""
        if not portfolio:
            return pd.Series()
            
        # Extract timestamps and values
        timestamps = []
        values = []
        
        for snapshot in portfolio:
            timestamps.append(snapshot.timestamp)
            values.append(snapshot.total_value)
            
        if not timestamps:
            return pd.Series()
            
        return pd.Series(values, index=timestamps)        
    def _calculate_drawdown_statistics(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown statistics"""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        
        # Calculate drawdown durations
        drawdown_starts = []
        drawdown_ends = []
        current_dd_start = None
        
        for i in range(len(in_drawdown)):
            if in_drawdown.iloc[i] and current_dd_start is None:
                current_dd_start = i
            elif not in_drawdown.iloc[i] and current_dd_start is not None:
                drawdown_starts.append(current_dd_start)
                drawdown_ends.append(i)
                current_dd_start = None
                
        # Handle ongoing drawdown
        if current_dd_start is not None:
            drawdown_starts.append(current_dd_start)
            drawdown_ends.append(len(in_drawdown) - 1)            
        # Calculate maximum duration
        max_duration = 0
        for start, end in zip(drawdown_starts, drawdown_ends):
            duration = (equity_curve.index[end] - equity_curve.index[start]).days
            max_duration = max(max_duration, duration)
            
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Number of drawdowns
        num_drawdowns = len(drawdown_starts)
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'avg_drawdown': avg_drawdown,
            'num_drawdowns': num_drawdowns
        }
        
    def _analyze_trades(self, portfolio: List[Any]) -> Dict[str, Any]:
        """Analyze trade statistics from portfolio history"""
        # Extract trades from portfolio
        trades = []
        
        # This would normally extract from portfolio trade history
        # For now, return placeholder statistics
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration': 0.0
            }            
        # Calculate trade statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'average_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'average_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'avg_duration': np.mean([t['duration'] for t in trades]) if trades else 0
        }
        
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no data available"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, monthly_return=0, daily_return=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            profit_factor=0, average_win=0, average_loss=0,
            largest_win=0, largest_loss=0, avg_trade_duration=0
        )