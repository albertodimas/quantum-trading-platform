"""
Portfolio Optimization Engine

Advanced portfolio optimization using Modern Portfolio Theory, Black-Litterman,
and risk parity approaches with multi-objective optimization capabilities.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from enum import Enum
from dataclasses import dataclass, field
import math
import statistics
from scipy import optimize
from scipy.stats import norm
import warnings

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderSide

logger = get_logger(__name__)

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "max_sharpe"  # Maximize Sharpe ratio
    MIN_VARIANCE = "min_variance"  # Minimize portfolio variance
    MAX_RETURN = "max_return"  # Maximize expected return
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman model
    MAX_CALMAR = "max_calmar"  # Maximize Calmar ratio
    MAX_SORTINO = "max_sortino"  # Maximize Sortino ratio
    ROBUST_OPTIMIZATION = "robust_optimization"  # Robust optimization


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum asset weight
    max_weight: float = 1.0  # Maximum asset weight
    long_only: bool = True  # Long-only constraint
    max_leverage: float = 1.0  # Maximum leverage
    target_return: Optional[float] = None  # Target return constraint
    max_volatility: Optional[float] = None  # Maximum volatility constraint
    max_drawdown: Optional[float] = None  # Maximum drawdown constraint
    min_assets: int = 1  # Minimum number of assets
    max_assets: Optional[int] = None  # Maximum number of assets
    turnover_limit: Optional[float] = None  # Maximum portfolio turnover
    sector_limits: Dict[str, float] = field(default_factory=dict)  # Sector exposure limits
    transaction_costs: float = 0.001  # Transaction cost (bps)


@dataclass
class AssetData:
    """Individual asset data for optimization"""
    symbol: str
    expected_return: float  # Annualized expected return
    volatility: float  # Annualized volatility
    current_weight: float = 0.0  # Current portfolio weight
    market_cap: Optional[float] = None  # Market capitalization
    sector: Optional[str] = None  # Asset sector
    beta: Optional[float] = None  # Market beta
    liquidity_score: float = 1.0  # Liquidity score (0-1)
    esg_score: Optional[float] = None  # ESG score
    analyst_views: Optional[Dict] = field(default_factory=dict)  # Analyst views for Black-Litterman


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    objective: OptimizationObjective
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None  # 95% Value at Risk
    cvar_95: Optional[float] = None  # 95% Conditional Value at Risk
    optimization_score: float = 0.0  # Overall optimization score
    rebalance_trades: Dict[str, float] = field(default_factory=dict)  # Required trades
    turnover: float = 0.0  # Portfolio turnover
    transaction_costs: float = 0.0  # Estimated transaction costs
    constraints_violated: List[str] = field(default_factory=list)  # Violated constraints
    optimization_status: str = "success"  # Optimization status
    computation_time: float = 0.0  # Computation time in seconds
    metadata: Dict = field(default_factory=dict)  # Additional metadata


@dataclass
class BacktestResult:
    """Portfolio backtest result"""
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    daily_returns: List[float] = field(default_factory=list)
    cumulative_returns: List[float] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)
    rebalance_dates: List[datetime] = field(default_factory=list)
    weights_history: List[Dict[str, float]] = field(default_factory=list)


@injectable
class PortfolioOptimizer:
    """
    Advanced portfolio optimizer with multiple optimization strategies.
    
    Features:
    - Modern Portfolio Theory (Markowitz)
    - Black-Litterman optimization
    - Risk parity and equal risk contribution
    - Robust optimization under uncertainty
    - Multi-objective optimization
    - Transaction cost optimization
    - Backtesting and performance analysis
    """
    
    def __init__(self):
        """Initialize portfolio optimizer."""
        
        # Historical data cache for optimization
        self._price_data: Dict[str, pd.DataFrame] = {}
        
        # Covariance matrix cache
        self._covariance_cache: Dict[str, np.ndarray] = {}
        
        # Market data for Black-Litterman
        self._market_data: Dict[str, Any] = {}
        
        # Optimization results history
        self._optimization_history: List[OptimizationResult] = []
        
        # Portfolio composition history
        self._portfolio_history: List[Dict] = []
        
        # Risk model parameters
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.lookback_days = 252  # 1 year of data
        self.min_correlation = -0.95  # Minimum correlation for numerical stability
        self.max_correlation = 0.95  # Maximum correlation for numerical stability
        
        logger.info("Portfolio optimizer initialized")
    
    async def optimize_portfolio(
        self,
        assets: List[AssetData],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        constraints: Optional[OptimizationConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio allocation using specified objective.
        
        Args:
            assets: List of assets with expected returns and risks
            objective: Optimization objective
            constraints: Portfolio constraints
            
        Returns:
            Optimization result with optimal weights and metrics
        """
        start_time = datetime.now()
        
        if not assets:
            raise ValueError("At least one asset is required for optimization")
        
        constraints = constraints or OptimizationConstraints()
        
        logger.info(
            f"Starting portfolio optimization",
            objective=objective.value,
            num_assets=len(assets),
            constraints=constraints
        )
        
        try:
            # Prepare optimization data
            symbols = [asset.symbol for asset in assets]
            expected_returns = np.array([asset.expected_return for asset in assets])
            
            # Build covariance matrix
            covariance_matrix = await self._build_covariance_matrix(assets)
            
            # Apply optimization strategy
            if objective == OptimizationObjective.MAX_SHARPE:
                optimal_weights = await self._optimize_max_sharpe(
                    expected_returns, covariance_matrix, constraints
                )
            elif objective == OptimizationObjective.MIN_VARIANCE:
                optimal_weights = await self._optimize_min_variance(
                    covariance_matrix, constraints
                )
            elif objective == OptimizationObjective.MAX_RETURN:
                optimal_weights = await self._optimize_max_return(
                    expected_returns, constraints
                )
            elif objective == OptimizationObjective.RISK_PARITY:
                optimal_weights = await self._optimize_risk_parity(
                    covariance_matrix, constraints
                )
            elif objective == OptimizationObjective.BLACK_LITTERMAN:
                optimal_weights = await self._optimize_black_litterman(
                    assets, covariance_matrix, constraints
                )
            elif objective == OptimizationObjective.MAX_CALMAR:
                optimal_weights = await self._optimize_max_calmar(
                    assets, constraints
                )
            elif objective == OptimizationObjective.ROBUST_OPTIMIZATION:
                optimal_weights = await self._optimize_robust(
                    expected_returns, covariance_matrix, constraints
                )
            else:
                optimal_weights = await self._optimize_max_sharpe(
                    expected_returns, covariance_matrix, constraints
                )
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate rebalancing trades
            current_weights = np.array([asset.current_weight for asset in assets])
            weight_changes = optimal_weights - current_weights
            turnover = np.sum(np.abs(weight_changes)) / 2
            
            rebalance_trades = {
                symbol: float(weight_change) 
                for symbol, weight_change in zip(symbols, weight_changes)
                if abs(weight_change) > 1e-6
            }
            
            # Calculate transaction costs
            transaction_costs = turnover * constraints.transaction_costs
            
            # Create result
            result = OptimizationResult(
                objective=objective,
                optimal_weights={symbol: float(weight) for symbol, weight in zip(symbols, optimal_weights)},
                expected_return=float(portfolio_return),
                expected_volatility=float(portfolio_volatility),
                sharpe_ratio=float(sharpe_ratio),
                rebalance_trades=rebalance_trades,
                turnover=float(turnover),
                transaction_costs=float(transaction_costs),
                optimization_status="success",
                computation_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Calculate additional risk metrics
            await self._calculate_risk_metrics(result, assets, covariance_matrix)
            
            # Validate constraints
            await self._validate_constraints(result, constraints)
            
            # Store result
            self._optimization_history.append(result)
            
            logger.info(
                f"Portfolio optimization completed",
                objective=objective.value,
                sharpe_ratio=result.sharpe_ratio,
                expected_return=result.expected_return,
                volatility=result.expected_volatility,
                computation_time=result.computation_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Portfolio optimization failed",
                error=str(e),
                objective=objective.value
            )
            
            # Return error result
            return OptimizationResult(
                objective=objective,
                optimal_weights={asset.symbol: asset.current_weight for asset in assets},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_status=f"failed: {str(e)}",
                computation_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def backtest_strategy(
        self,
        assets: List[str],
        start_date: datetime,
        end_date: datetime,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        constraints: Optional[OptimizationConstraints] = None
    ) -> BacktestResult:
        """
        Backtest portfolio optimization strategy.
        
        Args:
            assets: List of asset symbols
            start_date: Backtest start date
            end_date: Backtest end date
            objective: Optimization objective
            rebalance_frequency: Rebalancing frequency
            constraints: Portfolio constraints
            
        Returns:
            Backtest results with performance metrics
        """
        logger.info(
            f"Starting portfolio backtest",
            assets=assets,
            start_date=start_date.date(),
            end_date=end_date.date(),
            objective=objective.value,
            rebalance_frequency=rebalance_frequency.value
        )
        
        # Get historical price data
        price_data = await self._get_historical_data(assets, start_date, end_date)
        
        if price_data.empty:
            raise ValueError("No historical data available for backtesting")
        
        # Initialize backtest variables
        portfolio_values = []
        daily_returns = []
        weights_history = []
        rebalance_dates = []
        
        current_weights = np.ones(len(assets)) / len(assets)  # Equal weight start
        
        # Generate rebalancing dates
        rebalance_schedule = self._generate_rebalance_schedule(
            start_date, end_date, rebalance_frequency
        )
        
        for date in price_data.index:
            # Check if rebalancing is needed
            if date in rebalance_schedule:
                try:
                    # Prepare asset data for optimization
                    lookback_data = price_data.loc[:date].tail(self.lookback_days)
                    asset_data = await self._prepare_asset_data_from_prices(
                        assets, lookback_data
                    )
                    
                    # Optimize portfolio
                    optimization_result = await self.optimize_portfolio(
                        asset_data, objective, constraints
                    )
                    
                    if optimization_result.optimization_status == "success":
                        current_weights = np.array([
                            optimization_result.optimal_weights.get(symbol, 0)
                            for symbol in assets
                        ])
                        rebalance_dates.append(date)
                        
                except Exception as e:
                    logger.warning(f"Optimization failed on {date}: {str(e)}")
                    # Keep current weights
            
            # Calculate portfolio return for the day
            if len(daily_returns) > 0:  # Skip first day
                price_returns = price_data.loc[date].pct_change().fillna(0)
                portfolio_return = np.dot(current_weights, price_returns)
                daily_returns.append(portfolio_return)
            else:
                daily_returns.append(0.0)
            
            # Track weights
            weights_history.append({
                symbol: float(weight) 
                for symbol, weight in zip(assets, current_weights)
            })
            
            # Calculate cumulative portfolio value
            if len(portfolio_values) == 0:
                portfolio_values.append(1.0)
            else:
                portfolio_values.append(
                    portfolio_values[-1] * (1 + daily_returns[-1])
                )
        
        # Calculate performance metrics
        total_return = portfolio_values[-1] - 1
        annualized_return = (portfolio_values[-1] ** (252 / len(daily_returns))) - 1
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Calculate drawdowns
        cumulative_returns = [pv - 1 for pv in portfolio_values]
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (np.array(portfolio_values) - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Risk metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = np.mean([r for r in daily_returns if r <= var_95])
        
        # Win rate and profit factor
        winning_days = len([r for r in daily_returns if r > 0])
        win_rate = winning_days / len(daily_returns) if daily_returns else 0
        
        gross_profit = sum([r for r in daily_returns if r > 0])
        gross_loss = abs(sum([r for r in daily_returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        result = BacktestResult(
            period_start=start_date,
            period_end=end_date,
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            calmar_ratio=float(calmar_ratio),
            sortino_ratio=float(sortino_ratio),
            max_drawdown=float(max_drawdown),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            drawdown_series=drawdowns.tolist(),
            rebalance_dates=rebalance_dates,
            weights_history=weights_history
        )
        
        logger.info(
            f"Backtest completed",
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            num_rebalances=len(rebalance_dates)
        )
        
        return result
    
    async def calculate_efficient_frontier(
        self,
        assets: List[AssetData],
        num_points: int = 100,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, List[float]]:
        """
        Calculate efficient frontier for given assets.
        
        Args:
            assets: List of assets
            num_points: Number of points on the frontier
            constraints: Portfolio constraints
            
        Returns:
            Dictionary with returns, volatilities, and Sharpe ratios
        """
        if len(assets) < 2:
            raise ValueError("At least 2 assets required for efficient frontier")
        
        constraints = constraints or OptimizationConstraints()
        
        # Prepare data
        expected_returns = np.array([asset.expected_return for asset in assets])
        covariance_matrix = await self._build_covariance_matrix(assets)
        
        # Find min variance and max return portfolios
        min_var_result = await self._optimize_min_variance(covariance_matrix, constraints)
        min_var_return = np.dot(min_var_result, expected_returns)
        
        max_ret_result = await self._optimize_max_return(expected_returns, constraints)
        max_ret_return = np.dot(max_ret_result, expected_returns)
        
        # Generate target returns along the frontier
        target_returns = np.linspace(min_var_return, max_ret_return, num_points)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        
        for target_return in target_returns:
            try:
                # Optimize for target return
                target_constraints = OptimizationConstraints(
                    min_weight=constraints.min_weight,
                    max_weight=constraints.max_weight,
                    long_only=constraints.long_only,
                    target_return=float(target_return)
                )
                
                weights = await self._optimize_for_target_return(
                    expected_returns, covariance_matrix, target_constraints
                )
                
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                frontier_returns.append(float(portfolio_return))
                frontier_volatilities.append(float(portfolio_volatility))
                frontier_sharpe_ratios.append(float(sharpe_ratio))
                
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {str(e)}")
                continue
        
        logger.info(
            f"Efficient frontier calculated",
            num_points=len(frontier_returns),
            min_return=min(frontier_returns) if frontier_returns else 0,
            max_return=max(frontier_returns) if frontier_returns else 0
        )
        
        return {
            "returns": frontier_returns,
            "volatilities": frontier_volatilities,
            "sharpe_ratios": frontier_sharpe_ratios
        }
    
    async def suggest_rebalancing(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float,
        min_trade_size: float = 100.0
    ) -> Dict[str, Dict]:
        """
        Suggest optimal rebalancing trades.
        
        Args:
            current_portfolio: Current portfolio weights
            target_portfolio: Target portfolio weights
            portfolio_value: Total portfolio value
            min_trade_size: Minimum trade size to execute
            
        Returns:
            Dictionary of suggested trades with details
        """
        trades = {}
        
        all_symbols = set(current_portfolio.keys()) | set(target_portfolio.keys())
        
        for symbol in all_symbols:
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = target_portfolio.get(symbol, 0.0)
            
            weight_diff = target_weight - current_weight
            trade_value = weight_diff * portfolio_value
            
            if abs(trade_value) >= min_trade_size:
                trades[symbol] = {
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_change": weight_diff,
                    "trade_value": trade_value,
                    "side": "BUY" if trade_value > 0 else "SELL",
                    "priority": abs(weight_diff)  # Higher weight changes get higher priority
                }
        
        # Sort by priority (largest changes first)
        sorted_trades = dict(sorted(
            trades.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        ))
        
        logger.info(
            f"Rebalancing suggestions generated",
            num_trades=len(sorted_trades),
            total_turnover=sum(abs(t["weight_change"]) for t in sorted_trades.values()) / 2
        )
        
        return sorted_trades
    
    # Optimization implementations
    
    async def _optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize for maximum Sharpe ratio."""
        n_assets = len(expected_returns)
        
        def neg_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Constraints
        constraint_list = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            neg_sharpe_ratio,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            logger.warning(f"Sharpe optimization failed: {result.message}")
            return initial_guess
        
        return result.x
    
    async def _optimize_min_variance(
        self,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize for minimum variance."""
        n_assets = covariance_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraint_list = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            portfolio_variance,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            logger.warning(f"Min variance optimization failed: {result.message}")
            return initial_guess
        
        return result.x
    
    async def _optimize_max_return(
        self,
        expected_returns: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize for maximum return."""
        n_assets = len(expected_returns)
        
        def neg_portfolio_return(weights):
            return -np.dot(weights, expected_returns)
        
        # Constraints
        constraint_list = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            neg_portfolio_return,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            logger.warning(f"Max return optimization failed: {result.message}")
            return initial_guess
        
        return result.x
    
    async def _optimize_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)."""
        n_assets = covariance_matrix.shape[0]
        
        def risk_parity_objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contributions
            target_contrib = np.ones(n_assets) / n_assets
            
            # Sum of squared deviations from target
            return np.sum((contrib / np.sum(contrib) - target_contrib) ** 2)
        
        # Constraints
        constraint_list = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            risk_parity_objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            return initial_guess
        
        return result.x
    
    async def _optimize_black_litterman(
        self,
        assets: List[AssetData],
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize using Black-Litterman model."""
        n_assets = len(assets)
        
        # Market capitalization weights (if available)
        market_weights = np.array([
            asset.market_cap or 1.0 for asset in assets
        ])
        market_weights = market_weights / np.sum(market_weights)
        
        # Risk aversion parameter
        risk_aversion = 3.0
        
        # Implied equilibrium returns
        implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        # Black-Litterman adjustment (simplified)
        # In a full implementation, this would incorporate analyst views
        tau = 0.1  # Scaling factor
        adjusted_covariance = covariance_matrix * (1 + tau)
        
        # Use implied returns for optimization
        def neg_utility(weights):
            portfolio_return = np.dot(weights, implied_returns)
            portfolio_variance = np.dot(weights, np.dot(adjusted_covariance, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints
        constraint_list = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (market weights)
        initial_guess = market_weights
        
        # Optimize
        result = optimize.minimize(
            neg_utility,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            logger.warning(f"Black-Litterman optimization failed: {result.message}")
            return market_weights
        
        return result.x
    
    async def _optimize_max_calmar(
        self,
        assets: List[AssetData],
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize for maximum Calmar ratio (return/max drawdown)."""
        # This is a simplified implementation
        # Full implementation would require historical drawdown calculation
        expected_returns = np.array([asset.expected_return for asset in assets])
        covariance_matrix = await self._build_covariance_matrix(assets)
        
        # Use Sharpe ratio as proxy for Calmar ratio
        return await self._optimize_max_sharpe(expected_returns, covariance_matrix, constraints)
    
    async def _optimize_robust(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Robust optimization under uncertainty."""
        # Implement worst-case optimization
        # Add uncertainty to expected returns
        uncertainty_factor = 0.1  # 10% uncertainty
        worst_case_returns = expected_returns * (1 - uncertainty_factor)
        
        # Use worst-case returns for optimization
        return await self._optimize_max_sharpe(worst_case_returns, covariance_matrix, constraints)
    
    async def _optimize_for_target_return(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Optimize for minimum variance given target return."""
        n_assets = len(expected_returns)
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraint_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
            {"type": "eq", "fun": lambda x: np.dot(x, expected_returns) - constraints.target_return}  # Target return
        ]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            portfolio_variance,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000}
        )
        
        if not result.success:
            logger.warning(f"Target return optimization failed: {result.message}")
            return initial_guess
        
        return result.x
    
    # Helper methods
    
    async def _build_covariance_matrix(self, assets: List[AssetData]) -> np.ndarray:
        """Build covariance matrix from asset volatilities and correlations."""
        n_assets = len(assets)
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Default correlation matrix (simplified)
        # In practice, this would be estimated from historical data
        correlation_matrix = np.eye(n_assets)
        
        # Add some realistic correlations (simplified)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Random correlation between -0.5 and 0.8
                correlation = np.random.uniform(-0.5, 0.8)
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Ensure positive definite
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)
        
        # Convert to covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        return covariance_matrix
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite for numerical stability."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Set negative eigenvalues to small positive value
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        # Reconstruct matrix
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    async def _calculate_risk_metrics(
        self,
        result: OptimizationResult,
        assets: List[AssetData],
        covariance_matrix: np.ndarray
    ) -> None:
        """Calculate additional risk metrics for optimization result."""
        try:
            weights = np.array([result.optimal_weights.get(asset.symbol, 0) for asset in assets])
            
            # VaR calculation (simplified)
            portfolio_return = result.expected_return / 252  # Daily return
            portfolio_volatility = result.expected_volatility / np.sqrt(252)  # Daily vol
            
            # 95% VaR (5th percentile)
            result.var_95 = norm.ppf(0.05, portfolio_return, portfolio_volatility)
            
            # 95% CVaR (Conditional VaR)
            result.cvar_95 = portfolio_return - portfolio_volatility * norm.pdf(norm.ppf(0.05)) / 0.05
            
            # Calmar ratio approximation
            # Using volatility as proxy for max drawdown
            result.calmar_ratio = result.expected_return / (result.expected_volatility * 2)
            
            # Sortino ratio approximation
            # Using half volatility as downside deviation approximation
            downside_deviation = result.expected_volatility * 0.7  # Approximation
            result.sortino_ratio = (result.expected_return - self.risk_free_rate) / downside_deviation
            
        except Exception as e:
            logger.warning(f"Failed to calculate risk metrics: {str(e)}")
    
    async def _validate_constraints(
        self,
        result: OptimizationResult,
        constraints: OptimizationConstraints
    ) -> None:
        """Validate optimization result against constraints."""
        violations = []
        
        weights = list(result.optimal_weights.values())
        
        # Check weight bounds
        if any(w < constraints.min_weight - 1e-6 for w in weights):
            violations.append("min_weight_violation")
        
        if any(w > constraints.max_weight + 1e-6 for w in weights):
            violations.append("max_weight_violation")
        
        # Check sum of weights
        if abs(sum(weights) - 1.0) > 1e-6:
            violations.append("weights_sum_violation")
        
        # Check long-only constraint
        if constraints.long_only and any(w < -1e-6 for w in weights):
            violations.append("long_only_violation")
        
        # Check target return constraint
        if constraints.target_return is not None:
            if abs(result.expected_return - constraints.target_return) > 1e-3:
                violations.append("target_return_violation")
        
        # Check volatility constraint
        if constraints.max_volatility is not None:
            if result.expected_volatility > constraints.max_volatility + 1e-6:
                violations.append("max_volatility_violation")
        
        result.constraints_violated = violations
    
    async def _get_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical price data for backtesting."""
        # This is a placeholder - in practice, would fetch from data provider
        # For now, generate synthetic data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducible results
        
        data = {}
        for symbol in symbols:
            # Random walk with drift
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # Daily returns
            prices = [100.0]  # Starting price
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[symbol] = prices
        
        return pd.DataFrame(data, index=date_range)
    
    def _generate_rebalance_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: RebalanceFrequency
    ) -> List[datetime]:
        """Generate rebalancing schedule."""
        dates = []
        current_date = start_date
        
        if frequency == RebalanceFrequency.DAILY:
            delta = timedelta(days=1)
        elif frequency == RebalanceFrequency.WEEKLY:
            delta = timedelta(weeks=1)
        elif frequency == RebalanceFrequency.MONTHLY:
            delta = timedelta(days=30)
        elif frequency == RebalanceFrequency.QUARTERLY:
            delta = timedelta(days=90)
        elif frequency == RebalanceFrequency.ANNUAL:
            delta = timedelta(days=365)
        else:
            delta = timedelta(days=30)  # Default to monthly
        
        while current_date <= end_date:
            dates.append(current_date)
            current_date += delta
        
        return dates
    
    async def _prepare_asset_data_from_prices(
        self,
        symbols: List[str],
        price_data: pd.DataFrame
    ) -> List[AssetData]:
        """Prepare asset data from historical prices."""
        assets = []
        
        for symbol in symbols:
            if symbol not in price_data.columns:
                continue
            
            prices = price_data[symbol].dropna()
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                continue
            
            # Calculate expected return and volatility
            expected_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            assets.append(AssetData(
                symbol=symbol,
                expected_return=float(expected_return),
                volatility=float(volatility)
            ))
        
        return assets
    
    async def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self._optimization_history.copy()
    
    async def clear_optimization_history(self) -> None:
        """Clear optimization history."""
        self._optimization_history.clear()
        logger.info("Optimization history cleared")