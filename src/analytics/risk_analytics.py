"""
Risk Analytics Module

Provides comprehensive risk analysis including VaR calculations, stress testing,
exposure analysis, and risk attribution.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

from ..core.logger import get_logger
from ..models.trading import Trade
from ..models.risk import RiskMetrics
from .performance_analyzer import PerformanceSnapshot

logger = get_logger(__name__)


class RiskMeasure(Enum):
    """Types of risk measures"""
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    VAR_MONTE_CARLO = "var_monte_carlo"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"


class StressScenario(Enum):
    """Predefined stress test scenarios"""
    MARKET_CRASH = "market_crash"           # -20% market move
    FLASH_CRASH = "flash_crash"             # -10% in minutes
    VOLATILITY_SPIKE = "volatility_spike"   # 3x volatility
    LIQUIDITY_CRISIS = "liquidity_crisis"   # Wide spreads
    CORRELATION_BREAK = "correlation_break"  # Correlation to 1
    BLACK_SWAN = "black_swan"               # 6-sigma event


@dataclass
class RiskAnalysis:
    """Comprehensive risk analysis results"""
    # Value at Risk
    var_1d_95: float = 0.0
    var_1d_99: float = 0.0
    var_10d_95: float = 0.0
    var_10d_99: float = 0.0
    
    # Conditional VaR (Expected Shortfall)
    cvar_1d_95: float = 0.0
    cvar_1d_99: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration: int = 0
    recovery_time: int = 0
    
    # Greeks-like metrics
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0
    
    # Stress test results
    stress_scenarios: Dict[str, float] = None
    worst_case_loss: float = 0.0
    
    # Risk decomposition
    risk_attribution: Dict[str, float] = None
    concentration_risk: float = 0.0
    correlation_risk: float = 0.0
    
    def __post_init__(self):
        if self.stress_scenarios is None:
            self.stress_scenarios = {}
        if self.risk_attribution is None:
            self.risk_attribution = {}


class RiskAnalytics:
    """Analyzes portfolio and trading risks"""
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self._cache: Dict[str, Any] = {}
    
    async def analyze_risk(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot],
        market_data: Optional[Dict[str, Any]] = None,
        benchmark_returns: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis
        
        Args:
            trades: Historical trades
            snapshots: Performance snapshots
            market_data: Market data for correlation
            benchmark_returns: Benchmark returns for relative metrics
            
        Returns:
            Comprehensive risk analysis
        """
        # Calculate returns
        returns = self._calculate_returns(snapshots)
        
        # Calculate risk metrics
        risk_analysis = RiskAnalysis()
        
        # Value at Risk calculations
        await self._calculate_var_metrics(returns, risk_analysis)
        
        # Drawdown analysis
        self._analyze_drawdowns(snapshots, risk_analysis)
        
        # Greeks-like sensitivities
        await self._calculate_sensitivities(trades, snapshots, risk_analysis)
        
        # Stress testing
        stress_results = await self._run_stress_tests(trades, snapshots, returns)
        risk_analysis.stress_scenarios = stress_results
        risk_analysis.worst_case_loss = min(stress_results.values()) if stress_results else 0
        
        # Risk attribution
        risk_attribution = await self._calculate_risk_attribution(trades, returns)
        risk_analysis.risk_attribution = risk_attribution
        
        # Concentration and correlation risk
        risk_analysis.concentration_risk = await self._calculate_concentration_risk(trades)
        risk_analysis.correlation_risk = await self._calculate_correlation_risk(trades, market_data)
        
        # Additional analysis
        additional_metrics = {
            "risk_adjusted_returns": await self._calculate_risk_adjusted_returns(returns, risk_analysis),
            "tail_risk_analysis": self._analyze_tail_risk(returns),
            "risk_budgeting": await self._calculate_risk_budget(trades, returns),
            "scenario_analysis": await self._scenario_analysis(trades, snapshots),
            "liquidity_risk": await self._assess_liquidity_risk(trades),
            "operational_risk": self._assess_operational_risk(trades)
        }
        
        return {
            "risk_metrics": risk_analysis.__dict__,
            "returns_analysis": self._analyze_returns_distribution(returns),
            "risk_factors": await self._identify_risk_factors(trades, returns),
            **additional_metrics
        }
    
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
    
    async def _calculate_var_metrics(self, returns: List[float], risk_analysis: RiskAnalysis):
        """Calculate Value at Risk metrics"""
        if not returns:
            return
        
        # 1-day VaR
        risk_analysis.var_1d_95 = self._calculate_var(returns, 0.95, 1)
        risk_analysis.var_1d_99 = self._calculate_var(returns, 0.99, 1)
        
        # 10-day VaR (scaled)
        risk_analysis.var_10d_95 = self._calculate_var(returns, 0.95, 10)
        risk_analysis.var_10d_99 = self._calculate_var(returns, 0.99, 10)
        
        # Conditional VaR (Expected Shortfall)
        risk_analysis.cvar_1d_95 = self._calculate_cvar(returns, 0.95, 1)
        risk_analysis.cvar_1d_99 = self._calculate_cvar(returns, 0.99, 1)
    
    def _calculate_var(
        self,
        returns: List[float],
        confidence: float,
        days: int,
        method: str = "historical"
    ) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0
        
        if method == "historical":
            # Historical VaR
            var_1d = np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            # Parametric VaR (assuming normal distribution)
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence)
            var_1d = mean + z_score * std
        else:
            # Monte Carlo VaR
            var_1d = self._monte_carlo_var(returns, confidence)
        
        # Scale to N days using square root of time
        var_nd = var_1d * np.sqrt(days)
        
        return abs(var_nd)  # Return as positive value
    
    def _calculate_cvar(
        self,
        returns: List[float],
        confidence: float,
        days: int
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if not returns:
            return 0
        
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return 0
        
        cvar_1d = np.mean(tail_returns)
        cvar_nd = cvar_1d * np.sqrt(days)
        
        return abs(cvar_nd)
    
    def _monte_carlo_var(
        self,
        returns: List[float],
        confidence: float,
        simulations: int = 10000
    ) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        if not returns:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, simulations)
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        
        return var
    
    def _analyze_drawdowns(self, snapshots: List[PerformanceSnapshot], risk_analysis: RiskAnalysis):
        """Analyze drawdown characteristics"""
        if not snapshots:
            return
        
        equity_curve = [s.equity for s in snapshots]
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdowns
        drawdowns = (equity_curve - running_max) / running_max
        
        # Current drawdown
        risk_analysis.current_drawdown = abs(drawdowns[-1]) if drawdowns[-1] < 0 else 0
        
        # Maximum drawdown
        risk_analysis.max_drawdown = abs(np.min(drawdowns))
        
        # Average drawdown
        negative_drawdowns = [d for d in drawdowns if d < 0]
        risk_analysis.avg_drawdown = abs(np.mean(negative_drawdowns)) if negative_drawdowns else 0
        
        # Drawdown duration and recovery
        self._calculate_drawdown_duration(drawdowns, risk_analysis)
    
    def _calculate_drawdown_duration(self, drawdowns: List[float], risk_analysis: RiskAnalysis):
        """Calculate drawdown duration and recovery time"""
        if not drawdowns:
            return
        
        # Find drawdown periods
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        recovery_times = []
        
        for i, dd in enumerate(drawdowns):
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                if in_drawdown:
                    recovery_times.append(current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        risk_analysis.drawdown_duration = max_duration
        risk_analysis.recovery_time = int(np.mean(recovery_times)) if recovery_times else 0
    
    async def _calculate_sensitivities(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot],
        risk_analysis: RiskAnalysis
    ):
        """Calculate portfolio sensitivities (Greeks-like metrics)"""
        if not snapshots or len(snapshots) < 2:
            return
        
        # Delta: Portfolio sensitivity to market moves
        returns = self._calculate_returns(snapshots)
        if returns:
            # Simple beta as proxy for delta
            risk_analysis.portfolio_delta = np.std(returns) / 0.01  # Sensitivity per 1% move
        
        # Gamma: Rate of change of delta
        if len(returns) > 2:
            return_diffs = np.diff(returns)
            risk_analysis.portfolio_gamma = np.std(return_diffs) / 0.01
        
        # Vega: Sensitivity to volatility
        rolling_vols = self._calculate_rolling_volatility(returns, window=20)
        if rolling_vols:
            risk_analysis.portfolio_vega = np.mean(np.abs(np.diff(rolling_vols)))
        
        # Theta: Time decay (using average daily P&L)
        daily_pnls = [s.daily_return * s.equity for s in snapshots[1:]]
        risk_analysis.portfolio_theta = np.mean(daily_pnls) if daily_pnls else 0
    
    def _calculate_rolling_volatility(
        self,
        returns: List[float],
        window: int = 20
    ) -> List[float]:
        """Calculate rolling volatility"""
        if len(returns) < window:
            return []
        
        rolling_vols = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            vol = np.std(window_returns) * np.sqrt(252)  # Annualized
            rolling_vols.append(vol)
        
        return rolling_vols
    
    async def _run_stress_tests(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot],
        returns: List[float]
    ) -> Dict[str, float]:
        """Run stress test scenarios"""
        stress_results = {}
        
        if not returns or not snapshots:
            return stress_results
        
        current_equity = snapshots[-1].equity if snapshots else 0
        
        # Market crash scenario
        stress_results[StressScenario.MARKET_CRASH.value] = -0.20 * current_equity
        
        # Flash crash scenario
        stress_results[StressScenario.FLASH_CRASH.value] = -0.10 * current_equity
        
        # Volatility spike scenario
        current_vol = np.std(returns) if returns else 0
        vol_spike_loss = current_vol * 3 * current_equity
        stress_results[StressScenario.VOLATILITY_SPIKE.value] = -vol_spike_loss
        
        # Liquidity crisis scenario
        avg_trade_size = np.mean([t.quantity * t.entry_price for t in trades]) if trades else 0
        liquidity_loss = avg_trade_size * 0.05  # 5% slippage on average position
        stress_results[StressScenario.LIQUIDITY_CRISIS.value] = -liquidity_loss * len(trades)
        
        # Correlation break scenario
        diversification_benefit = current_equity * 0.1  # Assume 10% diversification benefit
        stress_results[StressScenario.CORRELATION_BREAK.value] = -diversification_benefit
        
        # Black swan event (6-sigma)
        if returns:
            black_swan_loss = 6 * np.std(returns) * current_equity
            stress_results[StressScenario.BLACK_SWAN.value] = -black_swan_loss
        
        return stress_results
    
    async def _calculate_risk_attribution(
        self,
        trades: List[Trade],
        returns: List[float]
    ) -> Dict[str, float]:
        """Calculate risk attribution by factors"""
        attribution = {
            "market_risk": 0.0,
            "specific_risk": 0.0,
            "timing_risk": 0.0,
            "sizing_risk": 0.0,
            "execution_risk": 0.0
        }
        
        if not returns:
            return attribution
        
        total_risk = np.std(returns)
        
        # Simplified attribution (would use factor models in production)
        attribution["market_risk"] = total_risk * 0.6  # 60% from market
        attribution["specific_risk"] = total_risk * 0.2  # 20% from selection
        attribution["timing_risk"] = total_risk * 0.1   # 10% from timing
        attribution["sizing_risk"] = total_risk * 0.05  # 5% from sizing
        attribution["execution_risk"] = total_risk * 0.05  # 5% from execution
        
        return attribution
    
    async def _calculate_concentration_risk(self, trades: List[Trade]) -> float:
        """Calculate concentration risk score"""
        if not trades:
            return 0
        
        # Calculate position concentration
        position_values = defaultdict(float)
        
        for trade in trades:
            position_values[trade.symbol] += abs(trade.quantity * trade.entry_price)
        
        total_value = sum(position_values.values())
        
        if total_value == 0:
            return 0
        
        # Calculate Herfindahl index
        herfindahl = sum((value / total_value) ** 2 for value in position_values.values())
        
        # Convert to risk score (0-1)
        # HHI of 1 = maximum concentration, 1/n = perfect diversification
        n_positions = len(position_values)
        min_hhi = 1 / n_positions if n_positions > 0 else 1
        
        concentration_risk = (herfindahl - min_hhi) / (1 - min_hhi) if min_hhi < 1 else 1
        
        return concentration_risk
    
    async def _calculate_correlation_risk(
        self,
        trades: List[Trade],
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate correlation risk"""
        # Simplified calculation without real correlation data
        # In production, would calculate actual correlation matrix
        
        symbols = list(set(t.symbol for t in trades))
        n_symbols = len(symbols)
        
        if n_symbols < 2:
            return 0
        
        # Assume average correlation increases risk
        avg_correlation = 0.3  # Placeholder
        
        # Higher correlation = higher risk
        correlation_risk = avg_correlation
        
        return correlation_risk
    
    async def _calculate_risk_adjusted_returns(
        self,
        returns: List[float],
        risk_analysis: RiskAnalysis
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        if not returns:
            return {}
        
        annualized_return = np.mean(returns) * 252
        annualized_vol = np.std(returns) * np.sqrt(252)
        
        risk_free_rate = 0.02  # 2% risk-free rate
        
        return {
            "sharpe_ratio": (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0,
            "sortino_ratio": self._calculate_sortino_ratio(returns, risk_free_rate),
            "calmar_ratio": annualized_return / risk_analysis.max_drawdown if risk_analysis.max_drawdown > 0 else 0,
            "return_over_var": annualized_return / risk_analysis.var_1d_95 if risk_analysis.var_1d_95 > 0 else 0
        }
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0
        
        excess_returns = [r - risk_free_rate / 252 for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns:
            return float('inf')
        
        avg_excess_return = np.mean(excess_returns) * 252
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        
        return avg_excess_return / downside_deviation if downside_deviation > 0 else 0
    
    def _analyze_tail_risk(self, returns: List[float]) -> Dict[str, Any]:
        """Analyze tail risk characteristics"""
        if not returns:
            return {}
        
        # Calculate tail statistics
        left_tail = np.percentile(returns, 5)
        right_tail = np.percentile(returns, 95)
        
        # Tail ratios
        left_tail_weight = len([r for r in returns if r <= left_tail]) / len(returns)
        right_tail_weight = len([r for r in returns if r >= right_tail]) / len(returns)
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            "left_tail_5pct": left_tail,
            "right_tail_95pct": right_tail,
            "tail_ratio": abs(left_tail / right_tail) if right_tail != 0 else float('inf'),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "fat_tail_risk": kurtosis > 3  # Leptokurtic distribution
        }
    
    async def _calculate_risk_budget(
        self,
        trades: List[Trade],
        returns: List[float]
    ) -> Dict[str, Any]:
        """Calculate risk budget allocation"""
        # Risk budget by strategy
        strategy_risk = defaultdict(lambda: {"trades": 0, "risk": 0.0})
        
        for trade in trades:
            strategy_risk[trade.strategy_id]["trades"] += 1
            # Simplified risk calculation
            position_risk = abs(trade.quantity * trade.entry_price) * 0.02  # 2% risk assumption
            strategy_risk[trade.strategy_id]["risk"] += position_risk
        
        total_risk = sum(s["risk"] for s in strategy_risk.values())
        
        risk_budget = {}
        for strategy_id, data in strategy_risk.items():
            risk_budget[strategy_id] = {
                "risk_allocation": data["risk"] / total_risk if total_risk > 0 else 0,
                "trade_count": data["trades"],
                "risk_per_trade": data["risk"] / data["trades"] if data["trades"] > 0 else 0
            }
        
        return {
            "by_strategy": risk_budget,
            "total_risk_budget": total_risk,
            "risk_utilization": min(total_risk / 100000, 1.0)  # Assume 100k risk budget
        }
    
    async def _scenario_analysis(
        self,
        trades: List[Trade],
        snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Perform scenario analysis"""
        scenarios = {
            "bull_market": {"market_move": 0.20, "vol_change": -0.2},
            "bear_market": {"market_move": -0.20, "vol_change": 0.5},
            "high_volatility": {"market_move": 0.0, "vol_change": 1.0},
            "low_volatility": {"market_move": 0.05, "vol_change": -0.5}
        }
        
        current_equity = snapshots[-1].equity if snapshots else 0
        
        results = {}
        for scenario_name, params in scenarios.items():
            # Simple linear impact model
            market_impact = current_equity * params["market_move"]
            vol_impact = current_equity * params["vol_change"] * 0.1  # 10% impact from vol
            
            results[scenario_name] = {
                "expected_pnl": market_impact + vol_impact,
                "probability": 0.25,  # Equal probability for simplicity
                "severity": abs(market_impact + vol_impact) / current_equity if current_equity > 0 else 0
            }
        
        return results
    
    async def _assess_liquidity_risk(self, trades: List[Trade]) -> Dict[str, Any]:
        """Assess liquidity risk"""
        if not trades:
            return {}
        
        # Calculate average trade sizes
        trade_sizes = [t.quantity * t.entry_price for t in trades]
        avg_trade_size = np.mean(trade_sizes)
        max_trade_size = max(trade_sizes)
        
        # Estimate market impact (simplified)
        estimated_impact = {
            "linear_impact": avg_trade_size * 0.001,  # 0.1% impact
            "square_root_impact": np.sqrt(avg_trade_size) * 0.01,
            "max_impact": max_trade_size * 0.005  # 0.5% for largest trade
        }
        
        # Liquidity score (0-1, higher = more risk)
        liquidity_score = min(max_trade_size / 1000000, 1.0)  # Normalize to 1M
        
        return {
            "avg_trade_size": avg_trade_size,
            "max_trade_size": max_trade_size,
            "estimated_impact": estimated_impact,
            "liquidity_score": liquidity_score,
            "liquidity_risk_level": "High" if liquidity_score > 0.7 else "Medium" if liquidity_score > 0.3 else "Low"
        }
    
    def _assess_operational_risk(self, trades: List[Trade]) -> Dict[str, Any]:
        """Assess operational risk"""
        if not trades:
            return {}
        
        # Count potential operational issues
        issues = {
            "execution_delays": 0,
            "price_gaps": 0,
            "size_mismatches": 0
        }
        
        for trade in trades:
            # Check for execution delays (simplified)
            if trade.holding_time and trade.holding_time < timedelta(seconds=60):
                issues["execution_delays"] += 1
            
            # Check for price gaps
            if trade.exit_price and abs(trade.exit_price - trade.entry_price) / trade.entry_price > 0.05:
                issues["price_gaps"] += 1
        
        total_issues = sum(issues.values())
        operational_score = min(total_issues / len(trades), 1.0)
        
        return {
            "issues_found": issues,
            "operational_score": operational_score,
            "risk_level": "High" if operational_score > 0.1 else "Medium" if operational_score > 0.05 else "Low"
        }
    
    def _analyze_returns_distribution(self, returns: List[float]) -> Dict[str, Any]:
        """Analyze returns distribution characteristics"""
        if not returns:
            return {}
        
        return {
            "mean": np.mean(returns),
            "std": np.std(returns),
            "min": min(returns),
            "max": max(returns),
            "percentiles": {
                "1%": np.percentile(returns, 1),
                "5%": np.percentile(returns, 5),
                "25%": np.percentile(returns, 25),
                "50%": np.percentile(returns, 50),
                "75%": np.percentile(returns, 75),
                "95%": np.percentile(returns, 95),
                "99%": np.percentile(returns, 99)
            },
            "normality_test": self._test_normality(returns)
        }
    
    def _test_normality(self, returns: List[float]) -> Dict[str, Any]:
        """Test if returns follow normal distribution"""
        if len(returns) < 20:
            return {"test_performed": False, "reason": "Insufficient data"}
        
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Shapiro-Wilk test
        sw_stat, sw_pvalue = stats.shapiro(returns)
        
        return {
            "jarque_bera": {
                "statistic": jb_stat,
                "p_value": jb_pvalue,
                "is_normal": jb_pvalue > 0.05
            },
            "shapiro_wilk": {
                "statistic": sw_stat,
                "p_value": sw_pvalue,
                "is_normal": sw_pvalue > 0.05
            }
        }
    
    async def _identify_risk_factors(
        self,
        trades: List[Trade],
        returns: List[float]
    ) -> Dict[str, Any]:
        """Identify key risk factors"""
        risk_factors = {
            "concentration": await self._calculate_concentration_risk(trades),
            "leverage": self._estimate_leverage(trades),
            "correlation": 0.3,  # Placeholder
            "volatility": np.std(returns) * np.sqrt(252) if returns else 0,
            "tail_risk": len([r for r in returns if r < np.percentile(returns, 5)]) / len(returns) if returns else 0
        }
        
        # Identify top risk factors
        sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "risk_factors": risk_factors,
            "top_risks": sorted_factors[:3],
            "risk_score": sum(risk_factors.values()) / len(risk_factors)
        }
    
    def _estimate_leverage(self, trades: List[Trade]) -> float:
        """Estimate portfolio leverage"""
        if not trades:
            return 0
        
        # Simple leverage estimation based on position sizes
        total_exposure = sum(t.quantity * t.entry_price for t in trades)
        
        # Assume some base capital (would get from portfolio in production)
        estimated_capital = 100000
        
        leverage = total_exposure / estimated_capital
        
        return min(leverage, 10.0)  # Cap at 10x for sanity