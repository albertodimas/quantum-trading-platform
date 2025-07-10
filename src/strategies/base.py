"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from src.core.logging import get_logger
from src.trading.models import Signal, OrderSide

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize base strategy."""
        self.name = name
        self.config = config or {}
        self._active = False
        self._performance_metrics = {
            "total_signals": 0,
            "winning_signals": 0,
            "losing_signals": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }
        
        # Strategy parameters
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.position_size = self.config.get("position_size", 0.02)  # 2% per trade
        self.max_positions = self.config.get("max_positions", 5)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.02)  # 2%
        self.take_profit_pct = self.config.get("take_profit_pct", 0.05)  # 5%
        
        # Initialize strategy-specific parameters
        self._initialize_parameters()
    
    @abstractmethod
    def _initialize_parameters(self):
        """Initialize strategy-specific parameters."""
        pass
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate strategy insights.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Strategy analysis results
        """
        pass
    
    @abstractmethod
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on strategy analysis.
        
        Args:
            analysis: Strategy analysis results
            
        Returns:
            List of trading signals
        """
        pass
    
    def start(self):
        """Activate the strategy."""
        self._active = True
        logger.info(f"Strategy {self.name} started")
    
    def stop(self):
        """Deactivate the strategy."""
        self._active = False
        logger.info(f"Strategy {self.name} stopped")
    
    def is_active(self) -> bool:
        """Check if strategy is active."""
        return self._active
    
    def update_performance(self, signal_result: Dict[str, Any]):
        """Update strategy performance metrics."""
        self._performance_metrics["total_signals"] += 1
        
        pnl = signal_result.get("pnl", 0)
        self._performance_metrics["total_pnl"] += pnl
        
        if pnl > 0:
            self._performance_metrics["winning_signals"] += 1
        else:
            self._performance_metrics["losing_signals"] += 1
        
        # Update max drawdown
        self._update_drawdown(pnl)
        
        # Update Sharpe ratio
        self._update_sharpe_ratio()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        metrics = self._performance_metrics.copy()
        
        # Calculate win rate
        total = metrics["total_signals"]
        if total > 0:
            metrics["win_rate"] = metrics["winning_signals"] / total
            metrics["average_pnl"] = metrics["total_pnl"] / total
        else:
            metrics["win_rate"] = 0
            metrics["average_pnl"] = 0
        
        return metrics
    
    def _update_drawdown(self, pnl: float):
        """Update maximum drawdown."""
        # Simplified drawdown calculation
        if pnl < 0:
            current_drawdown = abs(pnl)
            if current_drawdown > self._performance_metrics["max_drawdown"]:
                self._performance_metrics["max_drawdown"] = current_drawdown
    
    def _update_sharpe_ratio(self):
        """Update Sharpe ratio calculation."""
        # Simplified Sharpe ratio - would need returns history for accurate calculation
        if self._performance_metrics["total_signals"] > 0:
            avg_return = self._performance_metrics["total_pnl"] / self._performance_metrics["total_signals"]
            # Assume 20% annualized volatility
            volatility = 0.20
            risk_free_rate = 0.02
            
            if volatility > 0:
                self._performance_metrics["sharpe_ratio"] = (avg_return - risk_free_rate) / volatility
    
    def calculate_position_size(self, capital: float, risk_per_trade: Optional[float] = None) -> float:
        """
        Calculate position size based on Kelly Criterion or fixed percentage.
        
        Args:
            capital: Available capital
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Position size in base currency
        """
        if risk_per_trade is None:
            risk_per_trade = self.position_size
        
        # Use simplified Kelly Criterion
        win_rate = self.get_performance_metrics().get("win_rate", 0.5)
        
        if win_rate > 0 and win_rate < 1:
            # Kelly fraction = (p * b - q) / b
            # where p = win rate, q = 1 - p, b = win/loss ratio
            avg_win = 2.0  # Assume 2:1 win/loss ratio
            kelly_fraction = (win_rate * avg_win - (1 - win_rate)) / avg_win
            
            # Use fractional Kelly (25%) for safety
            position_size = capital * min(kelly_fraction * 0.25, risk_per_trade)
        else:
            position_size = capital * risk_per_trade
        
        return max(position_size, 0)
    
    def calculate_stop_loss(self, entry_price: float, side: OrderSide) -> float:
        """Calculate stop loss price."""
        if side == OrderSide.BUY:
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, side: OrderSide) -> float:
        """Calculate take profit price."""
        if side == OrderSide.BUY:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal before execution.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid
        """
        # Check confidence
        if signal.confidence < self.min_confidence:
            logger.debug(f"Signal rejected: low confidence {signal.confidence}")
            return False
        
        # Check required fields
        if not signal.symbol or not signal.side:
            logger.debug("Signal rejected: missing required fields")
            return False
        
        # Strategy-specific validation
        return self._validate_signal_custom(signal)
    
    def _validate_signal_custom(self, signal: Signal) -> bool:
        """
        Strategy-specific signal validation.
        
        Override in subclasses for custom validation.
        """
        return True
    
    def adjust_for_market_conditions(self, market_conditions: Dict[str, Any]):
        """
        Adjust strategy parameters based on market conditions.
        
        Args:
            market_conditions: Current market conditions
        """
        volatility = market_conditions.get("volatility", 0.2)
        trend_strength = market_conditions.get("trend_strength", 0.5)
        
        # Adjust stop loss based on volatility
        if volatility > 0.3:  # High volatility
            self.stop_loss_pct = min(self.stop_loss_pct * 1.5, 0.05)
        elif volatility < 0.1:  # Low volatility
            self.stop_loss_pct = max(self.stop_loss_pct * 0.75, 0.01)
        
        # Adjust position size based on market regime
        market_regime = market_conditions.get("regime", "normal")
        if market_regime == "high_volatility":
            self.position_size *= 0.5  # Reduce position size
        elif market_regime == "trending":
            self.position_size *= 1.2  # Increase position size
        
        logger.info(f"Strategy {self.name} adjusted for market conditions: "
                   f"volatility={volatility:.2f}, regime={market_regime}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self.name,
            "active": self._active,
            "min_confidence": self.min_confidence,
            "position_size": self.position_size,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            **self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update strategy configuration."""
        # Update base parameters
        if "min_confidence" in new_config:
            self.min_confidence = new_config["min_confidence"]
        if "position_size" in new_config:
            self.position_size = new_config["position_size"]
        if "max_positions" in new_config:
            self.max_positions = new_config["max_positions"]
        if "stop_loss_pct" in new_config:
            self.stop_loss_pct = new_config["stop_loss_pct"]
        if "take_profit_pct" in new_config:
            self.take_profit_pct = new_config["take_profit_pct"]
        
        # Update strategy-specific config
        self.config.update(new_config)
        
        logger.info(f"Strategy {self.name} configuration updated")