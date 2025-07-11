"""
Portfolio Manager for multi-asset portfolio management.

Features:
- Portfolio construction and optimization
- Asset allocation and rebalancing
- Risk-adjusted performance metrics
- Correlation and diversification analysis
- Portfolio constraints and objectives
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ..core.observability.logger import get_logger
from ..core.observability.metrics import get_metrics_collector
from ..core.cache.cache_manager import CacheManager
from .position_tracker import PositionTracker, Position


logger = get_logger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"


class RebalanceStrategy(Enum):
    """Portfolio rebalancing strategies."""
    THRESHOLD = "threshold"  # Rebalance when deviation exceeds threshold
    CALENDAR = "calendar"    # Rebalance on schedule
    DYNAMIC = "dynamic"      # Dynamic rebalancing based on market conditions


@dataclass
class AssetData:
    """Asset data for portfolio analysis."""
    symbol: str
    returns: np.ndarray
    volatility: float
    expected_return: float
    current_price: Decimal
    current_position: Decimal = Decimal("0")
    market_cap: Optional[Decimal] = None
    sector: Optional[str] = None
    
    @property
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if self.volatility == 0:
            return 0
        return (self.expected_return - risk_free_rate) / self.volatility


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_position_size: Optional[Decimal] = None
    sector_limits: Dict[str, float] = field(default_factory=dict)
    asset_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    total_notional: Optional[Decimal] = None
    max_assets: Optional[int] = None
    min_assets: Optional[int] = None
    long_only: bool = True


@dataclass
class OptimizationResult:
    """Portfolio optimization results."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_assets: float
    positions: Dict[str, Decimal]
    rebalance_trades: Dict[str, Decimal]
    transaction_cost: Decimal


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: Decimal
    returns: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float


class PortfolioManager:
    """
    Portfolio manager for multi-asset portfolio optimization and management.
    """
    
    def __init__(self, position_tracker: PositionTracker, 
                 cache_manager: Optional[CacheManager] = None):
        self.position_tracker = position_tracker
        self._logger = get_logger(self.__class__.__name__)
        self._metrics = get_metrics_collector().get_collector("trading")
        self._cache = cache_manager
        
        # Portfolio data
        self._assets: Dict[str, AssetData] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._covariance_matrix: Optional[np.ndarray] = None
        
        # Historical data
        self._returns_history: Dict[str, List[float]] = {}
        self._portfolio_value_history: List[Tuple[datetime, Decimal]] = []
        
        # Optimization settings
        self._risk_free_rate = 0.02  # 2% annual
        self._transaction_cost_bps = 10  # 10 basis points
        
        # Rebalancing
        self._last_rebalance: Optional[datetime] = None
        self._rebalance_threshold = 0.05  # 5% deviation threshold
        
    async def add_asset(self, symbol: str, historical_prices: List[Tuple[datetime, float]],
                       sector: Optional[str] = None, market_cap: Optional[Decimal] = None):
        """Add asset to the portfolio universe."""
        if len(historical_prices) < 2:
            raise ValueError(f"Insufficient price history for {symbol}")
        
        # Calculate returns
        prices = np.array([p[1] for p in historical_prices])
        returns = np.diff(np.log(prices))
        
        # Calculate statistics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        expected_return = np.mean(returns) * 252  # Annualized
        
        # Get current position
        position = await self.position_tracker.get_position(symbol)
        current_position = position.quantity if position else Decimal("0")
        
        self._assets[symbol] = AssetData(
            symbol=symbol,
            returns=returns,
            volatility=volatility,
            expected_return=expected_return,
            current_price=Decimal(str(prices[-1])),
            current_position=current_position,
            market_cap=market_cap,
            sector=sector
        )
        
        # Store returns history
        self._returns_history[symbol] = returns.tolist()
        
        # Invalidate correlation/covariance matrices
        self._correlation_matrix = None
        self._covariance_matrix = None
        
        self._logger.info("Asset added to portfolio",
                         symbol=symbol,
                         expected_return=expected_return,
                         volatility=volatility)
    
    async def update_market_data(self, symbol: str, current_price: Decimal):
        """Update current market data for an asset."""
        if symbol in self._assets:
            self._assets[symbol].current_price = current_price
            
            # Update position
            position = await self.position_tracker.get_position(symbol)
            self._assets[symbol].current_position = position.quantity if position else Decimal("0")
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets."""
        if self._correlation_matrix is not None:
            return self._correlation_matrix
        
        if len(self._assets) < 2:
            return pd.DataFrame()
        
        # Align returns data
        symbols = list(self._assets.keys())
        min_length = min(len(self._returns_history[s]) for s in symbols)
        
        returns_data = {}
        for symbol in symbols:
            returns_data[symbol] = self._returns_history[symbol][-min_length:]
        
        returns_df = pd.DataFrame(returns_data)
        self._correlation_matrix = returns_df.corr()
        
        return self._correlation_matrix
    
    def calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate covariance matrix for portfolio assets."""
        if self._covariance_matrix is not None:
            return self._covariance_matrix
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        if corr_matrix.empty:
            return np.array([])
        
        # Convert to covariance
        symbols = list(self._assets.keys())
        volatilities = np.array([self._assets[s].volatility for s in symbols])
        
        # Cov(i,j) = Corr(i,j) * Vol(i) * Vol(j)
        vol_matrix = np.outer(volatilities, volatilities)
        self._covariance_matrix = corr_matrix.values * vol_matrix
        
        return self._covariance_matrix
    
    async def optimize_portfolio(self, objective: OptimizationObjective,
                               constraints: Optional[PortfolioConstraints] = None,
                               target_value: Optional[Decimal] = None) -> OptimizationResult:
        """
        Optimize portfolio allocation.
        
        Args:
            objective: Optimization objective
            constraints: Portfolio constraints
            target_value: Target portfolio value
            
        Returns:
            Optimization results with weights and metrics
        """
        if not self._assets:
            raise ValueError("No assets in portfolio")
        
        constraints = constraints or PortfolioConstraints()
        symbols = list(self._assets.keys())
        n_assets = len(symbols)
        
        # Get optimization inputs
        expected_returns = np.array([self._assets[s].expected_return for s in symbols])
        cov_matrix = self.calculate_covariance_matrix()
        
        if cov_matrix.size == 0:
            raise ValueError("Unable to calculate covariance matrix")
        
        # Initial weights (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Define optimization constraints
        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Asset bounds
        bounds = []
        for i, symbol in enumerate(symbols):
            if symbol in constraints.asset_limits:
                min_w, max_w = constraints.asset_limits[symbol]
            else:
                min_w, max_w = constraints.min_weight, constraints.max_weight
            
            if constraints.long_only:
                min_w = max(0, min_w)
            
            bounds.append((min_w, max_w))
        
        # Sector constraints
        if constraints.sector_limits:
            for sector, limit in constraints.sector_limits.items():
                sector_indices = [i for i, s in enumerate(symbols) 
                                if self._assets[s].sector == sector]
                if sector_indices:
                    opt_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=sector_indices, lim=limit: lim - np.sum(w[idx])
                    })
        
        # Define objective function
        if objective == OptimizationObjective.MAX_SHARPE:
            def objective_func(w):
                portfolio_return = np.dot(w, expected_returns)
                portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                sharpe = -(portfolio_return - self._risk_free_rate) / portfolio_vol
                return sharpe
        
        elif objective == OptimizationObjective.MIN_VARIANCE:
            def objective_func(w):
                return np.dot(w, np.dot(cov_matrix, w))
        
        elif objective == OptimizationObjective.MAX_RETURN:
            def objective_func(w):
                return -np.dot(w, expected_returns)
        
        elif objective == OptimizationObjective.RISK_PARITY:
            def objective_func(w):
                portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol
                contrib = w * marginal_contrib
                # Minimize deviation from equal risk contribution
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
        
        else:  # MAX_DIVERSIFICATION
            def objective_func(w):
                weighted_vols = np.dot(w, np.array([self._assets[s].volatility for s in symbols]))
                portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                diversification_ratio = weighted_vols / portfolio_vol
                return -diversification_ratio
        
        # Optimize
        result = minimize(
            objective_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            self._logger.warning("Optimization failed", reason=result.message)
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - self._risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate diversification ratio
        weighted_vols = np.dot(optimal_weights, np.array([self._assets[s].volatility for s in symbols]))
        div_ratio = weighted_vols / portfolio_vol if portfolio_vol > 0 else 1
        
        # Calculate effective number of assets (using entropy)
        weights_positive = optimal_weights[optimal_weights > 1e-6]
        if len(weights_positive) > 0:
            entropy = -np.sum(weights_positive * np.log(weights_positive))
            effective_assets = np.exp(entropy)
        else:
            effective_assets = 0
        
        # Calculate target positions
        if target_value is None:
            # Use current portfolio value
            current_value = sum(
                asset.current_position * asset.current_price 
                for asset in self._assets.values()
            )
            target_value = current_value if current_value > 0 else Decimal("100000")
        
        positions = {}
        rebalance_trades = {}
        transaction_cost = Decimal("0")
        
        for i, symbol in enumerate(symbols):
            weight = optimal_weights[i]
            asset = self._assets[symbol]
            
            # Calculate target position
            target_value_asset = target_value * Decimal(str(weight))
            target_position = target_value_asset / asset.current_price
            
            # Round to reasonable precision
            if target_position < Decimal("0.001"):
                target_position = Decimal("0")
            
            positions[symbol] = target_position
            
            # Calculate rebalance trade
            trade_quantity = target_position - asset.current_position
            rebalance_trades[symbol] = trade_quantity
            
            # Estimate transaction cost
            if trade_quantity != 0:
                trade_value = abs(trade_quantity) * asset.current_price
                transaction_cost += trade_value * Decimal(str(self._transaction_cost_bps)) / 10000
        
        result = OptimizationResult(
            weights={s: optimal_weights[i] for i, s in enumerate(symbols)},
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=div_ratio,
            effective_assets=effective_assets,
            positions=positions,
            rebalance_trades=rebalance_trades,
            transaction_cost=transaction_cost
        )
        
        # Cache result
        if self._cache:
            cache_key = f"portfolio_optimization:{objective.value}"
            await self._cache.set_async(cache_key, result.__dict__, ttl=300)
        
        self._logger.info("Portfolio optimized",
                         objective=objective.value,
                         expected_return=portfolio_return,
                         volatility=portfolio_vol,
                         sharpe_ratio=sharpe_ratio)
        
        return result
    
    async def calculate_portfolio_metrics(self, lookback_days: int = 252) -> PortfolioMetrics:
        """Calculate comprehensive portfolio performance metrics."""
        # Get portfolio value history
        if not self._portfolio_value_history:
            await self._update_portfolio_value_history()
        
        if len(self._portfolio_value_history) < 2:
            raise ValueError("Insufficient portfolio history for metrics calculation")
        
        # Calculate returns
        values = np.array([float(v[1]) for v in self._portfolio_value_history])
        returns = np.diff(values) / values[:-1]
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self._risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self._risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * np.sqrt(252)
        
        # Conditional VaR (Expected Shortfall)
        returns_below_var = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = np.mean(returns_below_var) * np.sqrt(252) if len(returns_below_var) > 0 else var_95
        
        # Beta and Alpha (would need market returns)
        beta = 1.0  # Placeholder
        alpha = annualized_return - (self._risk_free_rate + beta * 0.08)  # Assuming 8% market return
        
        # Information ratio and tracking error (would need benchmark)
        information_ratio = 0.0  # Placeholder
        tracking_error = 0.0  # Placeholder
        
        # Current portfolio value
        total_value = sum(
            asset.current_position * asset.current_price 
            for asset in self._assets.values()
        )
        
        return PortfolioMetrics(
            total_value=total_value,
            returns=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    async def check_rebalance_needed(self, strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD,
                                    threshold: float = 0.05) -> Tuple[bool, Dict[str, float]]:
        """
        Check if portfolio rebalancing is needed.
        
        Returns:
            Tuple of (needs_rebalance, deviations)
        """
        if not self._assets:
            return False, {}
        
        # Calculate current weights
        current_weights = {}
        total_value = sum(
            asset.current_position * asset.current_price 
            for asset in self._assets.values()
        )
        
        if total_value == 0:
            return True, {}  # Empty portfolio needs rebalancing
        
        for symbol, asset in self._assets.items():
            current_value = asset.current_position * asset.current_price
            current_weights[symbol] = float(current_value / total_value)
        
        # Get target weights (from last optimization or equal weight)
        if hasattr(self, '_last_optimization_weights'):
            target_weights = self._last_optimization_weights
        else:
            # Equal weight as default
            n_assets = len(self._assets)
            target_weights = {s: 1.0 / n_assets for s in self._assets}
        
        # Calculate deviations
        deviations = {}
        max_deviation = 0
        
        for symbol in self._assets:
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            deviation = abs(current_w - target_w)
            deviations[symbol] = deviation
            max_deviation = max(max_deviation, deviation)
        
        # Check based on strategy
        if strategy == RebalanceStrategy.THRESHOLD:
            needs_rebalance = max_deviation > threshold
        
        elif strategy == RebalanceStrategy.CALENDAR:
            # Check if enough time has passed
            if self._last_rebalance is None:
                needs_rebalance = True
            else:
                days_since_rebalance = (datetime.utcnow() - self._last_rebalance).days
                needs_rebalance = days_since_rebalance >= 30  # Monthly rebalance
        
        else:  # DYNAMIC
            # Dynamic rebalancing based on market conditions
            # Could consider volatility, correlation changes, etc.
            volatility_threshold = 0.25  # 25% annualized vol
            avg_volatility = np.mean([a.volatility for a in self._assets.values()])
            
            if avg_volatility > volatility_threshold:
                threshold = 0.03  # Tighter threshold in high volatility
            else:
                threshold = 0.07  # Looser threshold in low volatility
            
            needs_rebalance = max_deviation > threshold
        
        return needs_rebalance, deviations
    
    async def _update_portfolio_value_history(self):
        """Update portfolio value history."""
        # Get historical snapshots from position tracker
        snapshots = await self.position_tracker.get_snapshots()
        
        self._portfolio_value_history = [
            (s.timestamp, s.total_market_value) 
            for s in snapshots
        ]
        
        # Add current value
        current_value = sum(
            asset.current_position * asset.current_price 
            for asset in self._assets.values()
        )
        
        self._portfolio_value_history.append(
            (datetime.utcnow(), current_value)
        )
    
    def calculate_efficient_frontier(self, n_portfolios: int = 100) -> List[Tuple[float, float]]:
        """
        Calculate efficient frontier portfolios.
        
        Returns:
            List of (return, volatility) tuples
        """
        if len(self._assets) < 2:
            return []
        
        symbols = list(self._assets.keys())
        expected_returns = np.array([self._assets[s].expected_return for s in symbols])
        cov_matrix = self.calculate_covariance_matrix()
        
        if cov_matrix.size == 0:
            return []
        
        # Target returns from minimum to maximum
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier = []
        
        for target_return in target_returns:
            # Minimize variance for target return
            n_assets = len(symbols)
            initial_weights = np.ones(n_assets) / n_assets
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = [(0, 1) for _ in range(n_assets)]
            
            result = minimize(
                lambda w: np.dot(w, np.dot(cov_matrix, w)),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                portfolio_vol = np.sqrt(result.fun)
                frontier.append((target_return, portfolio_vol))
        
        return frontier