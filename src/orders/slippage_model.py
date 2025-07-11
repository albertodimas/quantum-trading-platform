"""
Slippage and Market Impact Models for Order Execution.

Features:
- Linear and non-linear market impact models
- Temporary and permanent impact components
- Spread cost modeling
- Slippage prediction and monitoring
- Adaptive model calibration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from abc import ABC, abstractmethod

from ..core.observability.logger import get_logger


logger = get_logger(__name__)


class ImpactModel(Enum):
    """Market impact model types."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    POWER_LAW = "power_law"
    ALMGREN_CHRISS = "almgren_chriss"


@dataclass
class MarketConditions:
    """Current market conditions for slippage calculation."""
    bid: Decimal
    ask: Decimal
    mid_price: Decimal
    spread: Decimal
    volume_24h: Decimal
    volatility: Decimal  # Daily volatility
    avg_trade_size: Decimal
    order_book_depth: Dict[str, List[Tuple[Decimal, Decimal]]]  # {"bids": [(price, size)], "asks": [...]}


@dataclass
class SlippageEstimate:
    """Slippage estimation results."""
    expected_price: Decimal
    expected_slippage: Decimal  # In basis points
    spread_cost: Decimal
    temporary_impact: Decimal
    permanent_impact: Decimal
    total_cost: Decimal
    confidence_interval: Tuple[Decimal, Decimal]  # 95% CI
    
    @property
    def slippage_percentage(self) -> Decimal:
        """Get slippage as percentage."""
        return self.expected_slippage / Decimal("100")


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    def __init__(self, model_params: Optional[Dict[str, float]] = None):
        self._logger = get_logger(self.__class__.__name__)
        self.params = model_params or self.get_default_params()
        self.calibration_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def estimate_slippage(self, order_size: Decimal, side: str,
                         market_conditions: MarketConditions) -> SlippageEstimate:
        """Estimate slippage for an order."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, float]:
        """Get default model parameters."""
        pass
    
    def calibrate(self, historical_executions: List[Dict[str, Any]]):
        """Calibrate model parameters using historical data."""
        if not historical_executions:
            return
        
        # Calculate realized slippage
        realized_slippages = []
        for execution in historical_executions:
            if all(k in execution for k in ["expected_price", "executed_price", "side"]):
                expected = float(execution["expected_price"])
                executed = float(execution["executed_price"])
                side = execution["side"]
                
                if side == "buy":
                    slippage = (executed - expected) / expected
                else:
                    slippage = (expected - executed) / expected
                
                realized_slippages.append(slippage * 10000)  # Convert to bps
        
        if realized_slippages:
            # Simple calibration - adjust impact coefficient
            avg_realized = np.mean(realized_slippages)
            avg_estimated = np.mean([e.get("estimated_slippage", 0) for e in historical_executions])
            
            if avg_estimated > 0:
                adjustment_factor = avg_realized / avg_estimated
                
                # Update parameters
                for key in self.params:
                    if "impact" in key:
                        self.params[key] *= adjustment_factor
                
                self.calibration_history.append({
                    "timestamp": datetime.utcnow(),
                    "samples": len(realized_slippages),
                    "adjustment_factor": adjustment_factor,
                    "avg_realized": avg_realized,
                    "avg_estimated": avg_estimated
                })
                
                self._logger.info("Model calibrated",
                                adjustment_factor=adjustment_factor,
                                samples=len(realized_slippages))


class LinearImpactModel(SlippageModel):
    """
    Linear market impact model.
    
    Impact = β * (Order Size / ADV)
    """
    
    def get_default_params(self) -> Dict[str, float]:
        return {
            "temporary_impact_coef": 10.0,  # basis points
            "permanent_impact_coef": 5.0,    # basis points
            "spread_multiplier": 0.5         # fraction of spread to cross
        }
    
    def estimate_slippage(self, order_size: Decimal, side: str,
                         market_conditions: MarketConditions) -> SlippageEstimate:
        """Estimate slippage using linear impact model."""
        # Calculate participation rate
        participation_rate = order_size / market_conditions.volume_24h
        
        # Spread cost (half-spread for crossing)
        spread_cost = market_conditions.spread * Decimal(str(self.params["spread_multiplier"]))
        
        # Temporary impact (linear in size)
        temp_impact = Decimal(str(self.params["temporary_impact_coef"])) * participation_rate
        
        # Permanent impact (linear in size)
        perm_impact = Decimal(str(self.params["permanent_impact_coef"])) * participation_rate
        
        # Total expected slippage in basis points
        total_slippage = spread_cost * 10000 / market_conditions.mid_price + temp_impact + perm_impact
        
        # Expected execution price
        if side == "buy":
            expected_price = market_conditions.ask * (1 + total_slippage / 10000)
        else:
            expected_price = market_conditions.bid * (1 - total_slippage / 10000)
        
        # Calculate confidence interval (simplified)
        volatility_adjustment = market_conditions.volatility * Decimal("1.96")  # 95% CI
        ci_lower = total_slippage - volatility_adjustment * 10000
        ci_upper = total_slippage + volatility_adjustment * 10000
        
        # Total cost in currency units
        total_cost = order_size * market_conditions.mid_price * total_slippage / 10000
        
        return SlippageEstimate(
            expected_price=expected_price,
            expected_slippage=total_slippage,
            spread_cost=spread_cost,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_cost=total_cost,
            confidence_interval=(ci_lower, ci_upper)
        )


class SquareRootImpactModel(SlippageModel):
    """
    Square-root market impact model.
    
    Impact = β * sqrt(Order Size / ADV) * σ
    """
    
    def get_default_params(self) -> Dict[str, float]:
        return {
            "impact_coefficient": 1.0,
            "temporary_impact_ratio": 0.7,  # 70% temporary, 30% permanent
            "spread_multiplier": 0.5,
            "urgency_multiplier": 1.0
        }
    
    def estimate_slippage(self, order_size: Decimal, side: str,
                         market_conditions: MarketConditions) -> SlippageEstimate:
        """Estimate slippage using square-root impact model."""
        # Calculate participation rate
        participation_rate = float(order_size / market_conditions.volume_24h)
        
        # Spread cost
        spread_cost = market_conditions.spread * Decimal(str(self.params["spread_multiplier"]))
        
        # Market impact (square-root law)
        impact_bps = (self.params["impact_coefficient"] * 
                     np.sqrt(participation_rate) * 
                     float(market_conditions.volatility) * 
                     10000 *  # Convert to bps
                     self.params["urgency_multiplier"])
        
        impact_bps = Decimal(str(impact_bps))
        
        # Split into temporary and permanent
        temp_impact = impact_bps * Decimal(str(self.params["temporary_impact_ratio"]))
        perm_impact = impact_bps * (1 - Decimal(str(self.params["temporary_impact_ratio"])))
        
        # Total slippage
        total_slippage = spread_cost * 10000 / market_conditions.mid_price + impact_bps
        
        # Expected execution price
        if side == "buy":
            expected_price = market_conditions.ask * (1 + total_slippage / 10000)
        else:
            expected_price = market_conditions.bid * (1 - total_slippage / 10000)
        
        # Confidence interval
        volatility_adjustment = market_conditions.volatility * Decimal("1.96") * Decimal(str(np.sqrt(participation_rate)))
        ci_lower = total_slippage - volatility_adjustment * 10000
        ci_upper = total_slippage + volatility_adjustment * 10000
        
        # Total cost
        total_cost = order_size * market_conditions.mid_price * total_slippage / 10000
        
        return SlippageEstimate(
            expected_price=expected_price,
            expected_slippage=total_slippage,
            spread_cost=spread_cost,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_cost=total_cost,
            confidence_interval=(ci_lower, ci_upper)
        )


class AlmgrenChrissModel(SlippageModel):
    """
    Almgren-Chriss model for optimal execution.
    
    Includes both temporary and permanent impact with risk aversion.
    """
    
    def get_default_params(self) -> Dict[str, float]:
        return {
            "eta": 2.5e-6,          # Permanent impact coefficient
            "gamma": 2.5e-7,        # Temporary impact coefficient
            "alpha": 0.0625,        # Power law exponent
            "risk_aversion": 1e-6,  # Risk aversion parameter
            "spread_multiplier": 0.5
        }
    
    def estimate_slippage(self, order_size: Decimal, side: str,
                         market_conditions: MarketConditions) -> SlippageEstimate:
        """Estimate slippage using Almgren-Chriss model."""
        # Model parameters
        eta = self.params["eta"]
        gamma = self.params["gamma"]
        alpha = self.params["alpha"]
        
        # Convert to float for calculations
        X = float(order_size)
        V = float(market_conditions.volume_24h)
        sigma = float(market_conditions.volatility)
        S = float(market_conditions.mid_price)
        
        # Permanent impact: g(v) = eta * v^alpha
        # where v = X/V (participation rate)
        v = X / V
        permanent_impact_frac = eta * (v ** alpha)
        
        # Temporary impact: h(v) = gamma * v
        temporary_impact_frac = gamma * v
        
        # Convert to basis points
        perm_impact = Decimal(str(permanent_impact_frac * 10000))
        temp_impact = Decimal(str(temporary_impact_frac * 10000))
        
        # Spread cost
        spread_cost = market_conditions.spread * Decimal(str(self.params["spread_multiplier"]))
        
        # Total slippage
        total_slippage = spread_cost * 10000 / market_conditions.mid_price + perm_impact + temp_impact
        
        # Expected execution price
        if side == "buy":
            expected_price = market_conditions.ask * (1 + total_slippage / 10000)
        else:
            expected_price = market_conditions.bid * (1 - total_slippage / 10000)
        
        # Risk-adjusted confidence interval
        risk_factor = np.sqrt(self.params["risk_aversion"] * sigma * np.sqrt(X/V))
        volatility_adjustment = Decimal(str(risk_factor * 1.96 * 10000))
        ci_lower = total_slippage - volatility_adjustment
        ci_upper = total_slippage + volatility_adjustment
        
        # Total cost
        total_cost = order_size * market_conditions.mid_price * total_slippage / 10000
        
        return SlippageEstimate(
            expected_price=expected_price,
            expected_slippage=total_slippage,
            spread_cost=spread_cost,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_cost=total_cost,
            confidence_interval=(ci_lower, ci_upper)
        )


class AdaptiveSlippageModel:
    """
    Adaptive slippage model that switches between models based on market conditions.
    """
    
    def __init__(self):
        self._logger = get_logger(self.__class__.__name__)
        self.models = {
            ImpactModel.LINEAR: LinearImpactModel(),
            ImpactModel.SQUARE_ROOT: SquareRootImpactModel(),
            ImpactModel.ALMGREN_CHRISS: AlmgrenChrissModel()
        }
        self.model_performance: Dict[ImpactModel, List[float]] = {
            model: [] for model in ImpactModel
        }
    
    def estimate_slippage(self, order_size: Decimal, side: str,
                         market_conditions: MarketConditions,
                         preferred_model: Optional[ImpactModel] = None) -> SlippageEstimate:
        """
        Estimate slippage using the most appropriate model.
        
        Args:
            order_size: Size of the order
            side: "buy" or "sell"
            market_conditions: Current market conditions
            preferred_model: Optional preferred model to use
            
        Returns:
            Slippage estimate from the selected model
        """
        # Select model based on conditions or preference
        if preferred_model:
            model = self.models[preferred_model]
        else:
            model = self._select_best_model(order_size, market_conditions)
        
        # Get estimate
        estimate = model.estimate_slippage(order_size, side, market_conditions)
        
        self._logger.debug("Slippage estimated",
                         model=model.__class__.__name__,
                         order_size=str(order_size),
                         expected_slippage=str(estimate.expected_slippage))
        
        return estimate
    
    def _select_best_model(self, order_size: Decimal, 
                          market_conditions: MarketConditions) -> SlippageModel:
        """Select best model based on order characteristics and market conditions."""
        participation_rate = order_size / market_conditions.volume_24h
        
        # Simple heuristic for model selection
        if participation_rate < Decimal("0.01"):  # Small orders
            return self.models[ImpactModel.LINEAR]
        elif participation_rate < Decimal("0.05"):  # Medium orders
            return self.models[ImpactModel.SQUARE_ROOT]
        else:  # Large orders
            return self.models[ImpactModel.ALMGREN_CHRISS]
    
    def update_performance(self, model_type: ImpactModel, 
                          estimated_slippage: float, 
                          realized_slippage: float):
        """Update model performance metrics."""
        error = abs(estimated_slippage - realized_slippage)
        self.model_performance[model_type].append(error)
        
        # Keep only recent performance
        if len(self.model_performance[model_type]) > 100:
            self.model_performance[model_type] = self.model_performance[model_type][-100:]
    
    def calibrate_all_models(self, historical_executions: List[Dict[str, Any]]):
        """Calibrate all models using historical data."""
        for model in self.models.values():
            model.calibrate(historical_executions)