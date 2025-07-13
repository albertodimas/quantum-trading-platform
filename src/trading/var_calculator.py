"""
Value at Risk (VaR) Calculation System

Comprehensive VaR implementation supporting multiple methodologies:
- Historical Simulation VaR
- Parametric VaR (Normal Distribution)
- Monte Carlo VaR
- Cornish-Fisher VaR (with skewness and kurtosis)
- Expected Shortfall (Conditional VaR)
- Component VaR and Marginal VaR
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math
from scipy import stats
from scipy.stats import norm, skew, kurtosis
import warnings

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from .models import Position

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric" 
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    HYBRID = "hybrid"


class VaRHorizon(Enum):
    """VaR time horizons"""
    DAILY = "1d"
    WEEKLY = "7d"
    MONTHLY = "30d"
    QUARTERLY = "90d"
    ANNUAL = "365d"


@dataclass
class VaRConfiguration:
    """VaR calculation configuration"""
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    horizons: List[VaRHorizon] = field(default_factory=lambda: [VaRHorizon.DAILY, VaRHorizon.WEEKLY])
    methods: List[VaRMethod] = field(default_factory=lambda: [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC])
    historical_window: int = 252  # Trading days (1 year)
    monte_carlo_simulations: int = 10000
    scaling_method: str = "sqrt_time"  # "sqrt_time" or "linear"
    lambda_decay: float = 0.94  # EWMA lambda for volatility estimation
    include_expected_shortfall: bool = True
    include_component_var: bool = True


@dataclass
class VaRResult:
    """VaR calculation result"""
    method: VaRMethod
    confidence_level: float
    horizon: VaRHorizon
    var_amount: Decimal  # VaR in base currency
    var_percentage: float  # VaR as percentage of portfolio
    expected_shortfall: Optional[Decimal] = None  # Expected Shortfall (CVaR)
    volatility_estimate: Optional[float] = None
    return_mean: Optional[float] = None
    return_std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    calculation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComponentVaRResult:
    """Component VaR result for individual positions"""
    symbol: str
    component_var: Decimal
    marginal_var: Decimal
    contribution_percentage: float
    position_value: Decimal


@dataclass
class PortfolioVaRReport:
    """Comprehensive portfolio VaR report"""
    portfolio_value: Decimal
    calculation_timestamp: datetime
    var_results: List[VaRResult]
    component_vars: List[ComponentVaRResult]
    risk_metrics: Dict[str, float]
    model_validation: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Optional[Dict] = None


class VaRDataProcessor:
    """Utility class for VaR data processing"""
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate logarithmic returns from price series"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                returns.append(math.log(prices[i] / prices[i-1]))
        
        return returns
    
    @staticmethod
    def scale_var_to_horizon(var_1d: float, horizon: VaRHorizon, method: str = "sqrt_time") -> float:
        """Scale 1-day VaR to different time horizons"""
        horizon_days = {
            VaRHorizon.DAILY: 1,
            VaRHorizon.WEEKLY: 7,
            VaRHorizon.MONTHLY: 30,
            VaRHorizon.QUARTERLY: 90,
            VaRHorizon.ANNUAL: 365
        }
        
        days = horizon_days.get(horizon, 1)
        
        if method == "sqrt_time":
            return var_1d * math.sqrt(days)
        elif method == "linear":
            return var_1d * days
        else:
            return var_1d * math.sqrt(days)  # Default to sqrt scaling
    
    @staticmethod
    def ewma_volatility(returns: List[float], lambda_decay: float = 0.94) -> float:
        """Calculate EWMA (Exponentially Weighted Moving Average) volatility"""
        if len(returns) < 2:
            return 0.0
        
        # Convert to numpy for easier calculation
        returns_array = np.array(returns)
        
        # Initialize with sample variance
        variance = np.var(returns_array)
        
        # EWMA calculation
        for i, ret in enumerate(returns_array):
            if i == 0:
                variance = ret**2
            else:
                variance = lambda_decay * variance + (1 - lambda_decay) * ret**2
        
        return math.sqrt(variance)


@injectable
class VaRCalculator:
    """
    Advanced Value at Risk calculator with multiple methodologies.
    
    Supports:
    - Multiple VaR calculation methods
    - Component and marginal VaR
    - Expected shortfall calculation
    - Model validation and backtesting
    - Stress testing integration
    """
    
    def __init__(self, config: Optional[VaRConfiguration] = None):
        """Initialize VaR calculator"""
        self.config = config or VaRConfiguration()
        self.data_processor = VaRDataProcessor()
        
        # Historical data storage (would be replaced with proper data service)
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._portfolio_history: List[Tuple[datetime, float]] = []
        
        # Calculation cache
        self._calculation_cache: Dict[str, VaRResult] = {}
        
        logger.info(
            "VaR calculator initialized",
            methods=[m.value for m in self.config.methods],
            confidence_levels=self.config.confidence_levels,
            historical_window=self.config.historical_window
        )
    
    async def calculate_portfolio_var(
        self,
        positions: Dict[str, Position],
        portfolio_value: Decimal,
        method: Optional[VaRMethod] = None,
        confidence_level: float = 0.95,
        horizon: VaRHorizon = VaRHorizon.DAILY
    ) -> VaRResult:
        """
        Calculate portfolio VaR using specified method.
        
        Args:
            positions: Current portfolio positions
            portfolio_value: Total portfolio value
            method: VaR calculation method
            confidence_level: Confidence level (0.95 = 95%)
            horizon: Time horizon for VaR
            
        Returns:
            VaR calculation result
        """
        
        if method is None:
            method = self.config.methods[0] if self.config.methods else VaRMethod.HISTORICAL
        
        logger.debug(
            f"Calculating {method.value} VaR",
            confidence_level=confidence_level,
            horizon=horizon.value,
            portfolio_value=float(portfolio_value),
            position_count=len(positions)
        )
        
        # Generate or retrieve historical returns
        portfolio_returns = await self._get_portfolio_returns(positions, portfolio_value)
        
        if not portfolio_returns:
            logger.warning("No historical data available for VaR calculation")
            return self._create_fallback_var_result(method, confidence_level, horizon, portfolio_value)
        
        # Calculate VaR based on method
        if method == VaRMethod.HISTORICAL:
            return await self._calculate_historical_var(
                portfolio_returns, portfolio_value, confidence_level, horizon
            )
        elif method == VaRMethod.PARAMETRIC:
            return await self._calculate_parametric_var(
                portfolio_returns, portfolio_value, confidence_level, horizon
            )
        elif method == VaRMethod.MONTE_CARLO:
            return await self._calculate_monte_carlo_var(
                portfolio_returns, portfolio_value, confidence_level, horizon
            )
        elif method == VaRMethod.CORNISH_FISHER:
            return await self._calculate_cornish_fisher_var(
                portfolio_returns, portfolio_value, confidence_level, horizon
            )
        elif method == VaRMethod.HYBRID:
            return await self._calculate_hybrid_var(
                portfolio_returns, portfolio_value, confidence_level, horizon
            )
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    async def calculate_comprehensive_var_report(
        self,
        positions: Dict[str, Position],
        portfolio_value: Decimal
    ) -> PortfolioVaRReport:
        """
        Calculate comprehensive VaR report with multiple methods and metrics.
        
        Args:
            positions: Current portfolio positions
            portfolio_value: Total portfolio value
            
        Returns:
            Comprehensive VaR report
        """
        
        var_results = []
        
        # Calculate VaR for all configured methods, confidence levels, and horizons
        for method in self.config.methods:
            for confidence_level in self.config.confidence_levels:
                for horizon in self.config.horizons:
                    try:
                        result = await self.calculate_portfolio_var(
                            positions, portfolio_value, method, confidence_level, horizon
                        )
                        var_results.append(result)
                    except Exception as e:
                        logger.error(
                            f"Failed to calculate VaR: {method.value}",
                            confidence_level=confidence_level,
                            horizon=horizon.value,
                            error=str(e)
                        )
        
        # Calculate component VaR if enabled
        component_vars = []
        if self.config.include_component_var and positions:
            component_vars = await self._calculate_component_var(positions, portfolio_value)
        
        # Calculate risk metrics
        portfolio_returns = await self._get_portfolio_returns(positions, portfolio_value)
        risk_metrics = self._calculate_risk_metrics(portfolio_returns)
        
        # Model validation (simplified)
        model_validation = await self._validate_var_models(var_results, portfolio_returns)
        
        report = PortfolioVaRReport(
            portfolio_value=portfolio_value,
            calculation_timestamp=datetime.now(timezone.utc),
            var_results=var_results,
            component_vars=component_vars,
            risk_metrics=risk_metrics,
            model_validation=model_validation
        )
        
        logger.info(
            "Comprehensive VaR report generated",
            portfolio_value=float(portfolio_value),
            var_results_count=len(var_results),
            component_vars_count=len(component_vars)
        )
        
        return report
    
    async def _calculate_historical_var(
        self,
        returns: List[float],
        portfolio_value: Decimal,
        confidence_level: float,
        horizon: VaRHorizon
    ) -> VaRResult:
        """Calculate Historical Simulation VaR"""
        
        if len(returns) < 10:
            return self._create_fallback_var_result(
                VaRMethod.HISTORICAL, confidence_level, horizon, portfolio_value
            )
        
        # Sort returns for percentile calculation
        sorted_returns = sorted(returns)
        
        # Calculate VaR percentile
        percentile = 1 - confidence_level
        var_index = max(0, int(percentile * len(sorted_returns)) - 1)
        var_return = sorted_returns[var_index]
        
        # Scale to horizon
        var_return_scaled = self.data_processor.scale_var_to_horizon(
            var_return, horizon, self.config.scaling_method
        )
        
        # Convert to monetary amount (negative return = loss)
        var_amount = Decimal(str(abs(var_return_scaled * float(portfolio_value))))
        var_percentage = abs(var_return_scaled) * 100
        
        # Calculate Expected Shortfall if enabled
        expected_shortfall = None
        if self.config.include_expected_shortfall:
            tail_returns = sorted_returns[:var_index+1]
            if tail_returns:
                es_return = sum(tail_returns) / len(tail_returns)
                es_return_scaled = self.data_processor.scale_var_to_horizon(
                    es_return, horizon, self.config.scaling_method
                )
                expected_shortfall = Decimal(str(abs(es_return_scaled * float(portfolio_value))))
        
        return VaRResult(
            method=VaRMethod.HISTORICAL,
            confidence_level=confidence_level,
            horizon=horizon,
            var_amount=var_amount,
            var_percentage=var_percentage,
            expected_shortfall=expected_shortfall,
            return_mean=statistics.mean(returns),
            return_std=statistics.stdev(returns) if len(returns) > 1 else 0.0,
            metadata={
                "sample_size": len(returns),
                "var_percentile": percentile,
                "var_return": var_return
            }
        )
    
    async def _calculate_parametric_var(
        self,
        returns: List[float],
        portfolio_value: Decimal,
        confidence_level: float,
        horizon: VaRHorizon
    ) -> VaRResult:
        """Calculate Parametric VaR (Normal Distribution)"""
        
        if len(returns) < 2:
            return self._create_fallback_var_result(
                VaRMethod.PARAMETRIC, confidence_level, horizon, portfolio_value
            )
        
        # Calculate return statistics
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        # Calculate z-score for confidence level
        z_score = norm.ppf(1 - confidence_level)
        
        # Calculate 1-day VaR
        var_return_1d = mean_return + z_score * std_return  # Note: z_score is negative
        
        # Scale to horizon
        var_return_scaled = self.data_processor.scale_var_to_horizon(
            var_return_1d, horizon, self.config.scaling_method
        )
        
        # Convert to monetary amount
        var_amount = Decimal(str(abs(var_return_scaled * float(portfolio_value))))
        var_percentage = abs(var_return_scaled) * 100
        
        # Calculate Expected Shortfall
        expected_shortfall = None
        if self.config.include_expected_shortfall:
            # For normal distribution: ES = μ + σ * φ(Φ^(-1)(α)) / α
            # where φ is PDF and Φ is CDF
            alpha = 1 - confidence_level
            es_return_1d = mean_return - std_return * norm.pdf(z_score) / alpha
            es_return_scaled = self.data_processor.scale_var_to_horizon(
                es_return_1d, horizon, self.config.scaling_method
            )
            expected_shortfall = Decimal(str(abs(es_return_scaled * float(portfolio_value))))
        
        return VaRResult(
            method=VaRMethod.PARAMETRIC,
            confidence_level=confidence_level,
            horizon=horizon,
            var_amount=var_amount,
            var_percentage=var_percentage,
            expected_shortfall=expected_shortfall,
            volatility_estimate=std_return,
            return_mean=mean_return,
            return_std=std_return,
            metadata={
                "z_score": z_score,
                "var_return_1d": var_return_1d,
                "distribution": "normal"
            }
        )
    
    async def _calculate_monte_carlo_var(
        self,
        returns: List[float],
        portfolio_value: Decimal,
        confidence_level: float,
        horizon: VaRHorizon
    ) -> VaRResult:
        """Calculate Monte Carlo VaR"""
        
        if len(returns) < 10:
            return self._create_fallback_var_result(
                VaRMethod.MONTE_CARLO, confidence_level, horizon, portfolio_value
            )
        
        # Estimate parameters from historical data
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducible results
        simulated_returns = np.random.normal(
            mean_return, std_return, self.config.monte_carlo_simulations
        )
        
        # Scale to horizon
        if horizon != VaRHorizon.DAILY:
            horizon_multiplier = self._get_horizon_days(horizon)
            simulated_returns = simulated_returns * math.sqrt(horizon_multiplier)
        
        # Calculate VaR from simulations
        sorted_simulations = np.sort(simulated_returns)
        percentile = 1 - confidence_level
        var_index = max(0, int(percentile * len(sorted_simulations)) - 1)
        var_return = sorted_simulations[var_index]
        
        # Convert to monetary amount
        var_amount = Decimal(str(abs(var_return * float(portfolio_value))))
        var_percentage = abs(var_return) * 100
        
        # Calculate Expected Shortfall
        expected_shortfall = None
        if self.config.include_expected_shortfall:
            tail_returns = sorted_simulations[:var_index+1]
            if len(tail_returns) > 0:
                es_return = np.mean(tail_returns)
                expected_shortfall = Decimal(str(abs(es_return * float(portfolio_value))))
        
        return VaRResult(
            method=VaRMethod.MONTE_CARLO,
            confidence_level=confidence_level,
            horizon=horizon,
            var_amount=var_amount,
            var_percentage=var_percentage,
            expected_shortfall=expected_shortfall,
            return_mean=mean_return,
            return_std=std_return,
            metadata={
                "simulations": self.config.monte_carlo_simulations,
                "var_return": var_return,
                "simulation_mean": float(np.mean(simulated_returns)),
                "simulation_std": float(np.std(simulated_returns))
            }
        )
    
    async def _calculate_cornish_fisher_var(
        self,
        returns: List[float],
        portfolio_value: Decimal,
        confidence_level: float,
        horizon: VaRHorizon
    ) -> VaRResult:
        """Calculate Cornish-Fisher VaR (accounts for skewness and kurtosis)"""
        
        if len(returns) < 10:
            return self._create_fallback_var_result(
                VaRMethod.CORNISH_FISHER, confidence_level, horizon, portfolio_value
            )
        
        # Calculate return statistics
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        returns_array = np.array(returns)
        skewness = float(skew(returns_array))
        kurt = float(kurtosis(returns_array))  # Excess kurtosis
        
        # Calculate standard normal quantile
        z = norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher adjustment
        cf_adjustment = (
            z + 
            (z**2 - 1) * skewness / 6 + 
            (z**3 - 3*z) * kurt / 24 - 
            (2*z**3 - 5*z) * (skewness**2) / 36
        )
        
        # Calculate VaR
        var_return_1d = mean_return + cf_adjustment * std_return
        
        # Scale to horizon
        var_return_scaled = self.data_processor.scale_var_to_horizon(
            var_return_1d, horizon, self.config.scaling_method
        )
        
        # Convert to monetary amount
        var_amount = Decimal(str(abs(var_return_scaled * float(portfolio_value))))
        var_percentage = abs(var_return_scaled) * 100
        
        return VaRResult(
            method=VaRMethod.CORNISH_FISHER,
            confidence_level=confidence_level,
            horizon=horizon,
            var_amount=var_amount,
            var_percentage=var_percentage,
            volatility_estimate=std_return,
            return_mean=mean_return,
            return_std=std_return,
            skewness=skewness,
            kurtosis=kurt,
            metadata={
                "z_score": z,
                "cf_adjustment": cf_adjustment,
                "var_return_1d": var_return_1d
            }
        )
    
    async def _calculate_hybrid_var(
        self,
        returns: List[float],
        portfolio_value: Decimal,
        confidence_level: float,
        horizon: VaRHorizon
    ) -> VaRResult:
        """Calculate Hybrid VaR (combines multiple methods)"""
        
        # Calculate VaR using multiple methods
        historical_var = await self._calculate_historical_var(
            returns, portfolio_value, confidence_level, horizon
        )
        parametric_var = await self._calculate_parametric_var(
            returns, portfolio_value, confidence_level, horizon
        )
        
        # Simple average (could be weighted based on historical performance)
        hybrid_amount = (historical_var.var_amount + parametric_var.var_amount) / 2
        hybrid_percentage = (historical_var.var_percentage + parametric_var.var_percentage) / 2
        
        return VaRResult(
            method=VaRMethod.HYBRID,
            confidence_level=confidence_level,
            horizon=horizon,
            var_amount=hybrid_amount,
            var_percentage=hybrid_percentage,
            return_mean=statistics.mean(returns),
            return_std=statistics.stdev(returns) if len(returns) > 1 else 0.0,
            metadata={
                "historical_var": float(historical_var.var_amount),
                "parametric_var": float(parametric_var.var_amount),
                "method": "simple_average"
            }
        )
    
    async def _calculate_component_var(
        self,
        positions: Dict[str, Position],
        portfolio_value: Decimal
    ) -> List[ComponentVaRResult]:
        """Calculate Component VaR for individual positions"""
        
        component_vars = []
        
        for symbol, position in positions.items():
            position_value = abs(position.quantity) * position.current_price
            weight = position_value / portfolio_value if portfolio_value > 0 else Decimal("0")
            
            # Simplified component VaR calculation
            # In practice, this would require covariance matrix and marginal VaR calculation
            position_returns = await self._get_position_returns(symbol)
            
            if position_returns and len(position_returns) > 1:
                position_volatility = statistics.stdev(position_returns)
                # Simplified: assume correlation of 0.5 with portfolio
                component_var_amount = position_value * Decimal(str(position_volatility * 1.65 * 0.5))  # 95% VaR approximation
                marginal_var_amount = component_var_amount / weight if weight > 0 else Decimal("0")
            else:
                component_var_amount = position_value * Decimal("0.02")  # 2% default
                marginal_var_amount = component_var_amount
            
            contribution_percentage = float(component_var_amount / portfolio_value * 100) if portfolio_value > 0 else 0
            
            component_vars.append(ComponentVaRResult(
                symbol=symbol,
                component_var=component_var_amount,
                marginal_var=marginal_var_amount,
                contribution_percentage=contribution_percentage,
                position_value=position_value
            ))
        
        return component_vars
    
    def _calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        
        if not returns or len(returns) < 2:
            return {}
        
        returns_array = np.array(returns)
        
        metrics = {
            "mean_return": float(np.mean(returns_array)),
            "volatility": float(np.std(returns_array)),
            "skewness": float(skew(returns_array)),
            "kurtosis": float(kurtosis(returns_array)),
            "min_return": float(np.min(returns_array)),
            "max_return": float(np.max(returns_array)),
            "sharpe_ratio": 0.0,  # Would need risk-free rate
            "max_drawdown": 0.0   # Would need cumulative returns
        }
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = metrics["mean_return"] / metrics["volatility"]
        
        return metrics
    
    async def _validate_var_models(
        self,
        var_results: List[VaRResult],
        returns: List[float]
    ) -> Dict[str, float]:
        """Validate VaR models using backtesting"""
        
        # Simplified validation - would implement proper backtesting in production
        validation = {}
        
        for result in var_results:
            method_key = f"{result.method.value}_{result.confidence_level}"
            
            # Count violations (simplified)
            expected_violations = (1 - result.confidence_level) * len(returns)
            actual_violations = len([r for r in returns if abs(r) > result.var_percentage / 100])
            
            validation[method_key] = {
                "expected_violations": expected_violations,
                "actual_violations": actual_violations,
                "violation_ratio": actual_violations / expected_violations if expected_violations > 0 else 0
            }
        
        return validation
    
    async def _get_portfolio_returns(
        self,
        positions: Dict[str, Position],
        portfolio_value: Decimal
    ) -> List[float]:
        """Get or simulate portfolio returns"""
        
        # In production, this would retrieve actual historical portfolio returns
        # For now, simulate based on position data
        
        if not positions:
            return []
        
        # Simulate portfolio returns (simplified)
        returns = []
        for i in range(self.config.historical_window):
            # Generate synthetic portfolio return
            portfolio_return = 0.0
            
            for symbol, position in positions.items():
                position_weight = float(abs(position.quantity) * position.current_price / portfolio_value)
                # Simulate asset return (would be actual historical data)
                asset_return = np.random.normal(0.0005, 0.02)  # 0.05% mean, 2% volatility
                portfolio_return += position_weight * asset_return
            
            returns.append(portfolio_return)
        
        return returns
    
    async def _get_position_returns(self, symbol: str) -> List[float]:
        """Get historical returns for a specific position"""
        
        # Simulate returns for the symbol (would be actual data in production)
        return [np.random.normal(0.0003, 0.025) for _ in range(self.config.historical_window)]
    
    def _create_fallback_var_result(
        self,
        method: VaRMethod,
        confidence_level: float,
        horizon: VaRHorizon,
        portfolio_value: Decimal
    ) -> VaRResult:
        """Create fallback VaR result when insufficient data"""
        
        # Use conservative 5% portfolio VaR as fallback
        fallback_percentage = 5.0
        var_amount = portfolio_value * Decimal("0.05")
        
        return VaRResult(
            method=method,
            confidence_level=confidence_level,
            horizon=horizon,
            var_amount=var_amount,
            var_percentage=fallback_percentage,
            metadata={"fallback": True, "reason": "insufficient_data"}
        )
    
    def _get_horizon_days(self, horizon: VaRHorizon) -> int:
        """Get number of days for horizon"""
        horizon_map = {
            VaRHorizon.DAILY: 1,
            VaRHorizon.WEEKLY: 7,
            VaRHorizon.MONTHLY: 30,
            VaRHorizon.QUARTERLY: 90,
            VaRHorizon.ANNUAL: 365
        }
        return horizon_map.get(horizon, 1)
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of VaR calculations"""
        return {
            "configuration": {
                "methods": [m.value for m in self.config.methods],
                "confidence_levels": self.config.confidence_levels,
                "horizons": [h.value for h in self.config.horizons],
                "historical_window": self.config.historical_window,
                "monte_carlo_simulations": self.config.monte_carlo_simulations
            },
            "cache_size": len(self._calculation_cache),
            "price_history_symbols": len(self._price_history),
            "portfolio_history_points": len(self._portfolio_history)
        }