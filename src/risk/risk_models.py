"""
Advanced Risk Models for Portfolio Risk Assessment.

Features:
- Value at Risk (VaR) calculation
- Conditional VaR (CVaR/Expected Shortfall)
- Monte Carlo simulation
- Stress testing and scenario analysis
- Factor risk models
- Copula-based dependency modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import pandas as pd

from ..core.observability.logger import get_logger


logger = get_logger(__name__)


class VaRMethod(Enum):
    """Value at Risk calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


class StressScenario(Enum):
    """Predefined stress test scenarios."""
    MARKET_CRASH = "market_crash"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    BLACK_SWAN = "black_swan"


@dataclass
class VaRResult:
    """Value at Risk calculation results."""
    var_value: Decimal
    confidence_level: float
    time_horizon: int  # days
    method: VaRMethod
    portfolio_value: Decimal
    var_percentage: Decimal
    
    # Additional metrics
    expected_shortfall: Optional[Decimal] = None
    marginal_var: Optional[Dict[str, Decimal]] = None
    component_var: Optional[Dict[str, Decimal]] = None
    
    # Backtesting
    violations: Optional[int] = None
    expected_violations: Optional[int] = None
    kupiec_test: Optional[float] = None  # p-value


@dataclass
class StressTestResult:
    """Stress test results."""
    scenario: StressScenario
    portfolio_loss: Decimal
    loss_percentage: Decimal
    worst_performers: List[Tuple[str, Decimal]]
    risk_factors: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FactorExposure:
    """Factor risk exposures."""
    market_beta: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    volatility_factor: float
    liquidity_factor: float
    custom_factors: Dict[str, float] = field(default_factory=dict)


class RiskModels:
    """
    Advanced risk modeling for portfolio risk assessment.
    """
    
    def __init__(self):
        self._logger = get_logger(self.__class__.__name__)
        
        # Model parameters
        self._lookback_days = 252  # 1 year
        self._confidence_levels = [0.95, 0.99]
        self._time_horizons = [1, 5, 10, 20]  # days
        
        # Monte Carlo settings
        self._mc_simulations = 10000
        self._mc_seed = 42
        
        # Factor model settings
        self._factor_returns: Optional[pd.DataFrame] = None
        self._factor_loadings: Optional[pd.DataFrame] = None
    
    def calculate_var(self, returns: np.ndarray, portfolio_value: Decimal,
                     confidence_level: float = 0.95, time_horizon: int = 1,
                     method: VaRMethod = VaRMethod.HISTORICAL) -> VaRResult:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns array
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: VaR calculation method
            
        Returns:
            VaR calculation results
        """
        if method == VaRMethod.HISTORICAL:
            result = self._historical_var(returns, portfolio_value, confidence_level, time_horizon)
        elif method == VaRMethod.PARAMETRIC:
            result = self._parametric_var(returns, portfolio_value, confidence_level, time_horizon)
        elif method == VaRMethod.MONTE_CARLO:
            result = self._monte_carlo_var(returns, portfolio_value, confidence_level, time_horizon)
        else:  # CORNISH_FISHER
            result = self._cornish_fisher_var(returns, portfolio_value, confidence_level, time_horizon)
        
        # Calculate Expected Shortfall (CVaR)
        result.expected_shortfall = self._calculate_expected_shortfall(
            returns, confidence_level, time_horizon, portfolio_value
        )
        
        self._logger.info("VaR calculated",
                         method=method.value,
                         var=str(result.var_value),
                         confidence=confidence_level,
                         horizon=time_horizon)
        
        return result
    
    def _historical_var(self, returns: np.ndarray, portfolio_value: Decimal,
                       confidence_level: float, time_horizon: int) -> VaRResult:
        """Calculate historical VaR."""
        # Scale returns to time horizon
        scaled_returns = returns * np.sqrt(time_horizon)
        
        # Calculate VaR percentile
        var_percentile = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        var_value = abs(var_percentile) * float(portfolio_value)
        
        return VaRResult(
            var_value=Decimal(str(var_value)),
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_value,
            var_percentage=Decimal(str(abs(var_percentile)))
        )
    
    def _parametric_var(self, returns: np.ndarray, portfolio_value: Decimal,
                       confidence_level: float, time_horizon: int) -> VaRResult:
        """Calculate parametric (variance-covariance) VaR."""
        # Calculate return statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Scale to time horizon
        scaled_mean = mean_return * time_horizon
        scaled_std = std_return * np.sqrt(time_horizon)
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var_return = scaled_mean + z_score * scaled_std
        var_value = abs(var_return) * float(portfolio_value)
        
        return VaRResult(
            var_value=Decimal(str(var_value)),
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=portfolio_value,
            var_percentage=Decimal(str(abs(var_return)))
        )
    
    def _monte_carlo_var(self, returns: np.ndarray, portfolio_value: Decimal,
                        confidence_level: float, time_horizon: int) -> VaRResult:
        """Calculate Monte Carlo VaR."""
        np.random.seed(self._mc_seed)
        
        # Fit distribution to returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate Monte Carlo simulations
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            self._mc_simulations
        )
        
        # Calculate VaR from simulations
        var_percentile = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        var_value = abs(var_percentile) * float(portfolio_value)
        
        return VaRResult(
            var_value=Decimal(str(var_value)),
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.MONTE_CARLO,
            portfolio_value=portfolio_value,
            var_percentage=Decimal(str(abs(var_percentile)))
        )
    
    def _cornish_fisher_var(self, returns: np.ndarray, portfolio_value: Decimal,
                           confidence_level: float, time_horizon: int) -> VaRResult:
        """Calculate Cornish-Fisher VaR (adjusts for skewness and kurtosis)."""
        # Calculate moments
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Normal quantile
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        cf_quantile = z + (z**2 - 1) * skewness / 6 + \
                      (z**3 - 3*z) * kurtosis / 24 - \
                      (2*z**3 - 5*z) * skewness**2 / 36
        
        # Calculate VaR
        scaled_std = std_return * np.sqrt(time_horizon)
        var_return = mean_return * time_horizon + cf_quantile * scaled_std
        var_value = abs(var_return) * float(portfolio_value)
        
        return VaRResult(
            var_value=Decimal(str(var_value)),
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.CORNISH_FISHER,
            portfolio_value=portfolio_value,
            var_percentage=Decimal(str(abs(var_return)))
        )
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float,
                                    time_horizon: int, portfolio_value: Decimal) -> Decimal:
        """Calculate Expected Shortfall (Conditional VaR)."""
        # Scale returns
        scaled_returns = returns * np.sqrt(time_horizon)
        
        # Get returns below VaR threshold
        var_threshold = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        tail_returns = scaled_returns[scaled_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return Decimal("0")
        
        # Expected shortfall is average of tail returns
        es_return = np.mean(tail_returns)
        es_value = abs(es_return) * float(portfolio_value)
        
        return Decimal(str(es_value))
    
    def calculate_marginal_var(self, returns_matrix: np.ndarray, weights: np.ndarray,
                             portfolio_value: Decimal, confidence_level: float = 0.95,
                             time_horizon: int = 1) -> Dict[str, Decimal]:
        """
        Calculate Marginal VaR for each asset.
        
        Marginal VaR = ∂VaR/∂w_i (change in VaR from small change in weight)
        """
        n_assets = returns_matrix.shape[1]
        marginal_vars = {}
        
        # Calculate portfolio VaR
        portfolio_returns = np.dot(returns_matrix, weights)
        base_var = self.calculate_var(
            portfolio_returns, portfolio_value, confidence_level, time_horizon
        ).var_value
        
        # Calculate marginal VaR for each asset
        delta = 0.01  # 1% change in weight
        
        for i in range(n_assets):
            # Increase weight of asset i
            weights_up = weights.copy()
            weights_up[i] += delta
            weights_up /= weights_up.sum()  # Renormalize
            
            # Calculate new VaR
            portfolio_returns_up = np.dot(returns_matrix, weights_up)
            var_up = self.calculate_var(
                portfolio_returns_up, portfolio_value, confidence_level, time_horizon
            ).var_value
            
            # Marginal VaR
            marginal_var = (var_up - base_var) / (delta * float(portfolio_value))
            marginal_vars[f"asset_{i}"] = Decimal(str(marginal_var))
        
        return marginal_vars
    
    def calculate_component_var(self, returns_matrix: np.ndarray, weights: np.ndarray,
                              portfolio_value: Decimal, confidence_level: float = 0.95,
                              time_horizon: int = 1) -> Dict[str, Decimal]:
        """
        Calculate Component VaR for each asset.
        
        Component VaR = weight_i * Marginal VaR_i
        """
        marginal_vars = self.calculate_marginal_var(
            returns_matrix, weights, portfolio_value, confidence_level, time_horizon
        )
        
        component_vars = {}
        for i, (asset, marginal_var) in enumerate(marginal_vars.items()):
            component_var = Decimal(str(weights[i])) * marginal_var * portfolio_value
            component_vars[asset] = component_var
        
        return component_vars
    
    def stress_test(self, returns_matrix: np.ndarray, weights: np.ndarray,
                   portfolio_value: Decimal, scenario: StressScenario) -> StressTestResult:
        """
        Perform stress testing on portfolio.
        
        Args:
            returns_matrix: Historical returns for each asset
            weights: Portfolio weights
            portfolio_value: Current portfolio value
            scenario: Stress test scenario
            
        Returns:
            Stress test results
        """
        # Define scenario parameters
        scenario_params = self._get_scenario_parameters(scenario)
        
        # Apply stress to returns
        stressed_returns = self._apply_stress_scenario(returns_matrix, scenario_params)
        
        # Calculate portfolio impact
        portfolio_returns = np.dot(returns_matrix, weights)
        stressed_portfolio_returns = np.dot(stressed_returns, weights)
        
        # Calculate loss
        normal_value = portfolio_value * (1 + Decimal(str(np.mean(portfolio_returns))))
        stressed_value = portfolio_value * (1 + Decimal(str(np.mean(stressed_portfolio_returns))))
        loss = normal_value - stressed_value
        loss_pct = loss / portfolio_value
        
        # Find worst performers
        asset_losses = []
        for i in range(returns_matrix.shape[1]):
            asset_return = np.mean(returns_matrix[:, i])
            stressed_return = np.mean(stressed_returns[:, i])
            asset_loss = (asset_return - stressed_return) * weights[i]
            asset_losses.append((f"asset_{i}", Decimal(str(asset_loss))))
        
        worst_performers = sorted(asset_losses, key=lambda x: x[1], reverse=True)[:5]
        
        return StressTestResult(
            scenario=scenario,
            portfolio_loss=loss,
            loss_percentage=loss_pct,
            worst_performers=worst_performers,
            risk_factors=scenario_params
        )
    
    def _get_scenario_parameters(self, scenario: StressScenario) -> Dict[str, float]:
        """Get stress scenario parameters."""
        scenarios = {
            StressScenario.MARKET_CRASH: {
                "market_shock": -0.20,  # -20% market drop
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.3
            },
            StressScenario.FLASH_CRASH: {
                "market_shock": -0.10,  # -10% sudden drop
                "volatility_multiplier": 5.0,
                "liquidity_penalty": 0.05
            },
            StressScenario.LIQUIDITY_CRISIS: {
                "bid_ask_spread": 0.05,  # 5% spread widening
                "market_impact": 0.03,
                "funding_cost": 0.02
            },
            StressScenario.VOLATILITY_SPIKE: {
                "volatility_multiplier": 3.0,
                "mean_reversion": -0.5,
                "tail_risk_increase": 0.2
            },
            StressScenario.CORRELATION_BREAKDOWN: {
                "correlation_flip": -0.5,  # Correlations flip sign
                "dispersion_increase": 2.0,
                "sector_rotation": 0.3
            },
            StressScenario.BLACK_SWAN: {
                "market_shock": -0.40,  # -40% extreme event
                "volatility_multiplier": 10.0,
                "correlation_to_one": 0.9,
                "liquidity_freeze": 0.10
            }
        }
        
        return scenarios.get(scenario, scenarios[StressScenario.MARKET_CRASH])
    
    def _apply_stress_scenario(self, returns: np.ndarray, 
                              params: Dict[str, float]) -> np.ndarray:
        """Apply stress scenario to returns."""
        stressed_returns = returns.copy()
        
        # Apply market shock
        if "market_shock" in params:
            stressed_returns += params["market_shock"]
        
        # Apply volatility shock
        if "volatility_multiplier" in params:
            mean_returns = np.mean(stressed_returns, axis=0)
            stressed_returns = mean_returns + (stressed_returns - mean_returns) * params["volatility_multiplier"]
        
        # Apply correlation shock
        if "correlation_increase" in params:
            # Increase correlation by blending with market returns
            market_returns = np.mean(stressed_returns, axis=1, keepdims=True)
            blend_factor = params["correlation_increase"]
            stressed_returns = (1 - blend_factor) * stressed_returns + blend_factor * market_returns
        
        return stressed_returns
    
    def calculate_factor_exposures(self, returns: np.ndarray, 
                                 factor_returns: pd.DataFrame) -> FactorExposure:
        """
        Calculate factor exposures using regression.
        
        Args:
            returns: Asset returns
            factor_returns: Returns of risk factors
            
        Returns:
            Factor exposures
        """
        # Ensure alignment
        if len(returns) != len(factor_returns):
            min_len = min(len(returns), len(factor_returns))
            returns = returns[-min_len:]
            factor_returns = factor_returns.iloc[-min_len:]
        
        # Prepare factor matrix
        factors = factor_returns[['market', 'size', 'value', 'momentum', 'volatility', 'liquidity']].values
        
        # Run regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(factors, returns)
        
        exposures = model.coef_
        
        return FactorExposure(
            market_beta=float(exposures[0]),
            size_factor=float(exposures[1]),
            value_factor=float(exposures[2]),
            momentum_factor=float(exposures[3]),
            volatility_factor=float(exposures[4]),
            liquidity_factor=float(exposures[5])
        )
    
    def calculate_risk_attribution(self, returns_matrix: np.ndarray, weights: np.ndarray,
                                 factor_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose portfolio risk into factor contributions.
        
        Returns:
            Dictionary of factor -> risk contribution percentage
        """
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Get factor exposures
        exposures = self.calculate_factor_exposures(portfolio_returns, factor_returns)
        
        # Calculate factor contributions to variance
        factor_dict = {
            'market': exposures.market_beta,
            'size': exposures.size_factor,
            'value': exposures.value_factor,
            'momentum': exposures.momentum_factor,
            'volatility': exposures.volatility_factor,
            'liquidity': exposures.liquidity_factor
        }
        
        # Factor covariance matrix
        factor_cov = factor_returns[list(factor_dict.keys())].cov()
        
        # Calculate risk contributions
        total_var = 0
        contributions = {}
        
        for factor1, exposure1 in factor_dict.items():
            contribution = 0
            for factor2, exposure2 in factor_dict.items():
                contribution += exposure1 * exposure2 * factor_cov.loc[factor1, factor2]
            contributions[factor1] = contribution
            total_var += contribution
        
        # Convert to percentages
        if total_var > 0:
            for factor in contributions:
                contributions[factor] = contributions[factor] / total_var
        
        return contributions
    
    def backtest_var(self, returns: np.ndarray, var_results: List[VaRResult],
                    actual_returns: np.ndarray) -> Dict[str, Any]:
        """
        Backtest VaR model accuracy.
        
        Args:
            returns: Historical returns used for VaR calculation
            var_results: VaR results to backtest
            actual_returns: Actual returns that occurred
            
        Returns:
            Backtesting results including violation counts and statistical tests
        """
        results = {}
        
        for var_result in var_results:
            # Count violations
            var_threshold = -float(var_result.var_percentage)
            violations = np.sum(actual_returns < var_threshold)
            expected_violations = len(actual_returns) * (1 - var_result.confidence_level)
            
            # Kupiec test (unconditional coverage)
            kupiec_stat, kupiec_pvalue = self._kupiec_test(
                violations, len(actual_returns), var_result.confidence_level
            )
            
            # Christoffersen test (conditional coverage)
            christ_stat, christ_pvalue = self._christoffersen_test(
                actual_returns, var_threshold, var_result.confidence_level
            )
            
            results[f"{var_result.method.value}_{var_result.confidence_level}"] = {
                "violations": violations,
                "expected_violations": expected_violations,
                "violation_rate": violations / len(actual_returns),
                "kupiec_test": {"statistic": kupiec_stat, "p_value": kupiec_pvalue},
                "christoffersen_test": {"statistic": christ_stat, "p_value": christ_pvalue}
            }
        
        return results
    
    def _kupiec_test(self, violations: int, n_obs: int, confidence_level: float) -> Tuple[float, float]:
        """Kupiec unconditional coverage test."""
        p = 1 - confidence_level
        if violations == 0:
            likelihood_ratio = -2 * n_obs * np.log(1 - p)
        else:
            likelihood_ratio = -2 * (
                violations * np.log(violations / (n_obs * p)) +
                (n_obs - violations) * np.log((n_obs - violations) / (n_obs * (1 - p)))
            )
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(likelihood_ratio, 1)
        
        return likelihood_ratio, p_value
    
    def _christoffersen_test(self, returns: np.ndarray, threshold: float, 
                           confidence_level: float) -> Tuple[float, float]:
        """Christoffersen conditional coverage test."""
        # Create violation indicator
        violations = (returns < threshold).astype(int)
        
        # Count transitions
        n_00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n_01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n_10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n_11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        # Calculate probabilities
        p_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
        p_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
        p = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)
        
        # Likelihood ratio test
        if p_01 > 0 and p_11 > 0 and p > 0 and p < 1:
            lr_ind = -2 * (
                n_00 * np.log(1 - p) + n_01 * np.log(p) +
                n_10 * np.log(1 - p) + n_11 * np.log(p) -
                n_00 * np.log(1 - p_01) - n_01 * np.log(p_01) -
                n_10 * np.log(1 - p_11) - n_11 * np.log(p_11)
            )
        else:
            lr_ind = 0
        
        # Add Kupiec component
        kupiec_stat, _ = self._kupiec_test(np.sum(violations), len(returns), confidence_level)
        lr_cc = kupiec_stat + lr_ind
        
        # Chi-square test with 2 degrees of freedom
        p_value = 1 - stats.chi2.cdf(lr_cc, 2)
        
        return lr_cc, p_value