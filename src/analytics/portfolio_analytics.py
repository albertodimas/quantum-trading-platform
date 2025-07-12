"""
Portfolio Analytics Module

Provides portfolio-level analysis including composition, allocation,
concentration metrics, and portfolio optimization suggestions.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict

from ..core.logger import get_logger
from ..models.trading import Trade, Position
from ..models.portfolio import Portfolio
from .performance_analyzer import AnalysisTimeframe, PerformanceSnapshot, AnalysisConfig

logger = get_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio-specific metrics"""
    # Composition metrics
    num_positions: int = 0
    num_symbols: int = 0
    num_strategies: int = 0
    long_positions: int = 0
    short_positions: int = 0
    
    # Allocation metrics
    largest_position_pct: float = 0.0
    smallest_position_pct: float = 0.0
    avg_position_size: float = 0.0
    cash_allocation_pct: float = 0.0
    
    # Concentration metrics
    herfindahl_index: float = 0.0
    concentration_ratio_top5: float = 0.0
    effective_diversification: float = 0.0
    
    # Risk metrics
    portfolio_beta: float = 0.0
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    correlation_risk: float = 0.0
    
    # Efficiency metrics
    turnover_ratio: float = 0.0
    capital_utilization: float = 0.0
    leverage_ratio: float = 0.0


class PortfolioAnalytics:
    """Analyzes portfolio composition and characteristics"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    async def analyze(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot],
        timeframe: AnalysisTimeframe,
        config: AnalysisConfig
    ) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis
        
        Args:
            trades: List of trades
            snapshots: Performance snapshots
            timeframe: Analysis timeframe
            config: Analysis configuration
            
        Returns:
            Portfolio analysis results
        """
        # Calculate portfolio metrics
        metrics = await self._calculate_portfolio_metrics(trades, snapshots)
        
        # Analyze composition
        composition = await self._analyze_composition(trades, snapshots)
        
        # Analyze allocation
        allocation = await self._analyze_allocation(snapshots)
        
        # Analyze concentration
        concentration = await self._analyze_concentration(snapshots)
        
        # Analyze correlation
        correlation = await self._analyze_correlation(trades, snapshots)
        
        # Analyze efficiency
        efficiency = await self._analyze_efficiency(trades, snapshots)
        
        # Generate optimization suggestions
        optimization = await self._generate_optimization_suggestions(
            metrics, composition, allocation, concentration
        )
        
        return {
            "metrics": metrics.__dict__,
            "composition": composition,
            "allocation": allocation,
            "concentration": concentration,
            "correlation": correlation,
            "efficiency": efficiency,
            "optimization": optimization,
            "timeframe": timeframe.value
        }
    
    async def _calculate_portfolio_metrics(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> PortfolioMetrics:
        """Calculate portfolio-level metrics"""
        metrics = PortfolioMetrics()
        
        if not snapshots:
            return metrics
        
        latest_snapshot = snapshots[-1]
        
        # Composition metrics
        positions = self._get_current_positions(trades)
        metrics.num_positions = len(positions)
        metrics.num_symbols = len(set(p["symbol"] for p in positions))
        metrics.num_strategies = len(set(t.strategy_id for t in trades))
        metrics.long_positions = sum(1 for p in positions if p["side"] == "BUY")
        metrics.short_positions = sum(1 for p in positions if p["side"] == "SELL")
        
        # Allocation metrics
        if latest_snapshot.equity > 0:
            position_values = [p["value"] for p in positions]
            if position_values:
                metrics.largest_position_pct = max(position_values) / latest_snapshot.equity
                metrics.smallest_position_pct = min(position_values) / latest_snapshot.equity
                metrics.avg_position_size = np.mean(position_values)
            
            metrics.cash_allocation_pct = latest_snapshot.cash / latest_snapshot.equity
        
        # Concentration metrics
        metrics.herfindahl_index = self._calculate_herfindahl_index(positions, latest_snapshot.equity)
        metrics.concentration_ratio_top5 = self._calculate_concentration_ratio(positions, latest_snapshot.equity, 5)
        metrics.effective_diversification = self._calculate_effective_diversification(positions)
        
        # Risk metrics
        returns = self._calculate_returns(snapshots)
        if returns:
            metrics.portfolio_var = np.percentile(returns, 5)
            metrics.portfolio_cvar = np.mean([r for r in returns if r <= metrics.portfolio_var])
        
        # Efficiency metrics
        metrics.turnover_ratio = await self._calculate_turnover_ratio(trades, snapshots)
        metrics.capital_utilization = latest_snapshot.positions_value / latest_snapshot.equity if latest_snapshot.equity > 0 else 0
        
        return metrics
    
    async def _analyze_composition(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze portfolio composition"""
        positions = self._get_current_positions(trades)
        
        # Group by various dimensions
        by_symbol = defaultdict(float)
        by_strategy = defaultdict(float)
        by_exchange = defaultdict(float)
        by_side = defaultdict(float)
        
        for position in positions:
            by_symbol[position["symbol"]] += position["value"]
            by_strategy[position["strategy_id"]] += position["value"]
            by_exchange[position["exchange"]] += position["value"]
            by_side[position["side"]] += position["value"]
        
        # Calculate percentages
        total_value = sum(p["value"] for p in positions)
        
        composition = {
            "by_symbol": {
                symbol: {
                    "value": value,
                    "percentage": value / total_value if total_value > 0 else 0
                }
                for symbol, value in by_symbol.items()
            },
            "by_strategy": {
                strategy: {
                    "value": value,
                    "percentage": value / total_value if total_value > 0 else 0
                }
                for strategy, value in by_strategy.items()
            },
            "by_exchange": {
                exchange: {
                    "value": value,
                    "percentage": value / total_value if total_value > 0 else 0
                }
                for exchange, value in by_exchange.items()
            },
            "by_side": {
                side: {
                    "value": value,
                    "percentage": value / total_value if total_value > 0 else 0
                }
                for side, value in by_side.items()
            }
        }
        
        return composition
    
    async def _analyze_allocation(
        self,
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze portfolio allocation over time"""
        if not snapshots:
            return {}
        
        # Get allocation history
        allocation_history = []
        
        for snapshot in snapshots:
            if snapshot.equity > 0:
                allocation_history.append({
                    "timestamp": snapshot.timestamp,
                    "cash_pct": snapshot.cash / snapshot.equity,
                    "positions_pct": snapshot.positions_value / snapshot.equity,
                    "leverage": (snapshot.positions_value / snapshot.equity) if snapshot.equity > 0 else 0
                })
        
        # Calculate allocation statistics
        cash_allocations = [a["cash_pct"] for a in allocation_history]
        position_allocations = [a["positions_pct"] for a in allocation_history]
        leverages = [a["leverage"] for a in allocation_history]
        
        return {
            "current": {
                "cash": allocation_history[-1]["cash_pct"] if allocation_history else 0,
                "positions": allocation_history[-1]["positions_pct"] if allocation_history else 0,
                "leverage": allocation_history[-1]["leverage"] if allocation_history else 0
            },
            "average": {
                "cash": np.mean(cash_allocations) if cash_allocations else 0,
                "positions": np.mean(position_allocations) if position_allocations else 0,
                "leverage": np.mean(leverages) if leverages else 0
            },
            "range": {
                "cash": {
                    "min": min(cash_allocations) if cash_allocations else 0,
                    "max": max(cash_allocations) if cash_allocations else 0
                },
                "positions": {
                    "min": min(position_allocations) if position_allocations else 0,
                    "max": max(position_allocations) if position_allocations else 0
                },
                "leverage": {
                    "min": min(leverages) if leverages else 0,
                    "max": max(leverages) if leverages else 0
                }
            },
            "history": allocation_history[-30:]  # Last 30 periods
        }
    
    async def _analyze_concentration(
        self,
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze portfolio concentration metrics"""
        if not snapshots:
            return {}
        
        latest_snapshot = snapshots[-1]
        
        # Calculate various concentration metrics
        herfindahl_history = []
        diversification_history = []
        
        for snapshot in snapshots[-30:]:  # Last 30 periods
            # Mock calculation - would need actual position data
            herfindahl = 0.1  # Placeholder
            diversification = 10  # Placeholder
            
            herfindahl_history.append({
                "timestamp": snapshot.timestamp,
                "value": herfindahl
            })
            
            diversification_history.append({
                "timestamp": snapshot.timestamp,
                "value": diversification
            })
        
        return {
            "herfindahl_index": {
                "current": herfindahl_history[-1]["value"] if herfindahl_history else 0,
                "average": np.mean([h["value"] for h in herfindahl_history]) if herfindahl_history else 0,
                "history": herfindahl_history
            },
            "effective_assets": {
                "current": diversification_history[-1]["value"] if diversification_history else 0,
                "average": np.mean([d["value"] for d in diversification_history]) if diversification_history else 0,
                "history": diversification_history
            },
            "concentration_risk": self._assess_concentration_risk(herfindahl_history)
        }
    
    async def _analyze_correlation(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze portfolio correlation structure"""
        # Get returns by symbol
        symbol_returns = defaultdict(list)
        
        # Group trades by symbol and calculate returns
        for trade in trades:
            if trade.realized_pnl != 0:
                return_pct = trade.realized_pnl / (trade.quantity * trade.entry_price)
                symbol_returns[trade.symbol].append(return_pct)
        
        # Calculate correlation matrix
        symbols = list(symbol_returns.keys())
        n_symbols = len(symbols)
        
        if n_symbols < 2:
            return {"correlation_matrix": {}, "average_correlation": 0, "correlation_risk": "Low"}
        
        # Simple correlation calculation (would use proper returns data in production)
        correlation_matrix = {}
        correlations = []
        
        for i, symbol1 in enumerate(symbols):
            correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    # Mock correlation
                    corr = np.random.uniform(-0.3, 0.7)
                    correlation_matrix[symbol1][symbol2] = corr
                    if i < j:
                        correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        return {
            "correlation_matrix": correlation_matrix,
            "average_correlation": avg_correlation,
            "max_correlation": max(correlations) if correlations else 0,
            "min_correlation": min(correlations) if correlations else 0,
            "correlation_risk": self._assess_correlation_risk(avg_correlation)
        }
    
    async def _analyze_efficiency(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze portfolio efficiency metrics"""
        # Calculate turnover
        turnover_ratio = await self._calculate_turnover_ratio(trades, snapshots)
        
        # Calculate capital efficiency
        capital_efficiency = []
        for snapshot in snapshots[-30:]:  # Last 30 periods
            if snapshot.equity > 0:
                efficiency = snapshot.positions_value / snapshot.equity
                capital_efficiency.append({
                    "timestamp": snapshot.timestamp,
                    "value": efficiency
                })
        
        # Trading frequency
        if trades:
            first_trade = min(trades, key=lambda t: t.timestamp)
            last_trade = max(trades, key=lambda t: t.timestamp)
            trading_days = (last_trade.timestamp - first_trade.timestamp).days
            trades_per_day = len(trades) / trading_days if trading_days > 0 else 0
        else:
            trades_per_day = 0
        
        return {
            "turnover_ratio": turnover_ratio,
            "capital_utilization": {
                "current": capital_efficiency[-1]["value"] if capital_efficiency else 0,
                "average": np.mean([e["value"] for e in capital_efficiency]) if capital_efficiency else 0,
                "history": capital_efficiency
            },
            "trading_frequency": {
                "trades_per_day": trades_per_day,
                "holding_period": self._calculate_avg_holding_period(trades)
            },
            "cost_efficiency": await self._calculate_cost_efficiency(trades, snapshots)
        }
    
    async def _generate_optimization_suggestions(
        self,
        metrics: PortfolioMetrics,
        composition: Dict[str, Any],
        allocation: Dict[str, Any],
        concentration: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate portfolio optimization suggestions"""
        suggestions = []
        
        # Check concentration
        if metrics.herfindahl_index > 0.2:
            suggestions.append({
                "category": "Diversification",
                "priority": "High",
                "suggestion": "Portfolio is highly concentrated. Consider diversifying across more assets.",
                "metric": f"Herfindahl Index: {metrics.herfindahl_index:.3f}"
            })
        
        # Check cash allocation
        current_cash = allocation.get("current", {}).get("cash", 0)
        if current_cash > 0.3:
            suggestions.append({
                "category": "Capital Efficiency",
                "priority": "Medium",
                "suggestion": "High cash allocation detected. Consider deploying more capital.",
                "metric": f"Cash Allocation: {current_cash:.1%}"
            })
        
        # Check number of positions
        if metrics.num_positions < 5:
            suggestions.append({
                "category": "Diversification",
                "priority": "Medium",
                "suggestion": "Low number of positions. Consider adding more positions for better diversification.",
                "metric": f"Number of Positions: {metrics.num_positions}"
            })
        elif metrics.num_positions > 50:
            suggestions.append({
                "category": "Management",
                "priority": "Medium",
                "suggestion": "High number of positions may be difficult to manage effectively.",
                "metric": f"Number of Positions: {metrics.num_positions}"
            })
        
        # Check turnover
        if metrics.turnover_ratio > 10:
            suggestions.append({
                "category": "Trading Costs",
                "priority": "High",
                "suggestion": "Very high turnover ratio. Consider reducing trading frequency to lower costs.",
                "metric": f"Annual Turnover: {metrics.turnover_ratio:.1f}x"
            })
        
        return suggestions
    
    def _get_current_positions(self, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Get current positions from trades"""
        # Simplified position calculation
        positions = defaultdict(lambda: {
            "quantity": 0,
            "value": 0,
            "entry_price": 0,
            "side": None,
            "symbol": None,
            "exchange": None,
            "strategy_id": None
        })
        
        for trade in trades:
            key = f"{trade.symbol}_{trade.strategy_id}"
            pos = positions[key]
            
            if trade.side == "BUY":
                # Update average entry price
                total_value = pos["quantity"] * pos["entry_price"] + trade.quantity * trade.entry_price
                pos["quantity"] += trade.quantity
                pos["entry_price"] = total_value / pos["quantity"] if pos["quantity"] > 0 else 0
            else:  # SELL
                pos["quantity"] -= trade.quantity
            
            pos["value"] = pos["quantity"] * trade.exit_price if trade.exit_price else pos["quantity"] * trade.entry_price
            pos["side"] = "BUY" if pos["quantity"] > 0 else "SELL"
            pos["symbol"] = trade.symbol
            pos["exchange"] = trade.exchange
            pos["strategy_id"] = trade.strategy_id
        
        # Filter out closed positions
        return [pos for pos in positions.values() if abs(pos["quantity"]) > 0.0001]
    
    def _calculate_herfindahl_index(self, positions: List[Dict[str, Any]], total_equity: float) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        if not positions or total_equity == 0:
            return 0
        
        weights_squared = 0
        for position in positions:
            weight = abs(position["value"]) / total_equity
            weights_squared += weight ** 2
        
        return weights_squared
    
    def _calculate_concentration_ratio(self, positions: List[Dict[str, Any]], total_equity: float, top_n: int) -> float:
        """Calculate concentration ratio for top N positions"""
        if not positions or total_equity == 0:
            return 0
        
        # Sort positions by absolute value
        sorted_positions = sorted(positions, key=lambda p: abs(p["value"]), reverse=True)
        
        # Sum top N positions
        top_positions_value = sum(abs(p["value"]) for p in sorted_positions[:top_n])
        
        return top_positions_value / total_equity
    
    def _calculate_effective_diversification(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate effective number of assets (inverse of HHI)"""
        hhi = self._calculate_herfindahl_index(positions, sum(abs(p["value"]) for p in positions))
        
        return 1 / hhi if hhi > 0 else len(positions)
    
    def _calculate_returns(self, snapshots: List[PerformanceSnapshot]) -> List[float]:
        """Calculate returns from snapshots"""
        if len(snapshots) < 2:
            return []
        
        returns = []
        for i in range(1, len(snapshots)):
            if snapshots[i-1].equity > 0:
                ret = (snapshots[i].equity - snapshots[i-1].equity) / snapshots[i-1].equity
                returns.append(ret)
        
        return returns
    
    async def _calculate_turnover_ratio(self, trades: List[Trade], snapshots: List[PerformanceSnapshot]) -> float:
        """Calculate portfolio turnover ratio"""
        if not trades or not snapshots:
            return 0
        
        # Calculate total trading volume
        total_volume = sum(t.quantity * t.entry_price for t in trades)
        
        # Calculate average portfolio value
        avg_portfolio_value = np.mean([s.equity for s in snapshots])
        
        # Annualize if needed
        first_trade = min(trades, key=lambda t: t.timestamp)
        last_trade = max(trades, key=lambda t: t.timestamp)
        period_days = (last_trade.timestamp - first_trade.timestamp).days
        
        if period_days > 0 and avg_portfolio_value > 0:
            annualized_turnover = (total_volume / avg_portfolio_value) * (365 / period_days)
            return annualized_turnover
        
        return 0
    
    def _calculate_avg_holding_period(self, trades: List[Trade]) -> float:
        """Calculate average holding period in days"""
        holding_periods = []
        
        for trade in trades:
            if trade.holding_time:
                holding_periods.append(trade.holding_time.total_seconds() / 86400)  # Convert to days
        
        return np.mean(holding_periods) if holding_periods else 0
    
    async def _calculate_cost_efficiency(self, trades: List[Trade], snapshots: List[PerformanceSnapshot]) -> Dict[str, float]:
        """Calculate cost efficiency metrics"""
        if not trades:
            return {"total_costs": 0, "cost_per_trade": 0, "cost_ratio": 0}
        
        total_costs = sum(t.fees for t in trades)
        total_pnl = sum(t.realized_pnl for t in trades)
        
        return {
            "total_costs": total_costs,
            "cost_per_trade": total_costs / len(trades),
            "cost_ratio": total_costs / abs(total_pnl) if total_pnl != 0 else 0
        }
    
    def _assess_concentration_risk(self, herfindahl_history: List[Dict[str, Any]]) -> str:
        """Assess concentration risk level"""
        if not herfindahl_history:
            return "Unknown"
        
        current_hhi = herfindahl_history[-1]["value"]
        
        if current_hhi < 0.1:
            return "Low"
        elif current_hhi < 0.2:
            return "Medium"
        else:
            return "High"
    
    def _assess_correlation_risk(self, avg_correlation: float) -> str:
        """Assess correlation risk level"""
        if avg_correlation < 0.3:
            return "Low"
        elif avg_correlation < 0.6:
            return "Medium"
        else:
            return "High"