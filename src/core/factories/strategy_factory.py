"""
Advanced Strategy Factory with parameter optimization and dynamic loading.

Features:
- Strategy creation with parameter optimization
- Dynamic strategy loading and hot-swap capabilities
- Performance monitoring and backtesting integration
- Risk-aware strategy selection
- Multi-strategy portfolio management
"""

import asyncio
import time
import json
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Union, Callable
from enum import Enum
import hashlib
import threading
from pathlib import Path

from ..observability.logger import get_logger
from ..observability.metrics import get_metrics_collector
from ..observability.tracing import trace_sync
from ..configuration.models import TradingConfig, RiskConfig
from ..cache.decorators import cache_strategy_result


logger = get_logger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    SCALPING = "scalping"
    SWING = "swing"
    AI_ML = "ai_ml"
    QUANTITATIVE = "quantitative"
    FUNDAMENTAL = "fundamental"


class StrategyStatus(Enum):
    """Strategy execution status."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    BACKTESTING = "backtesting"
    OPTIMIZING = "optimizing"


@dataclass
class StrategyParameters:
    """Container for strategy parameters with validation."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_targets: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate parameters against constraints."""
        for param_name, value in self.parameters.items():
            if param_name in self.constraints:
                constraint = self.constraints[param_name]
                
                # Check type
                if 'type' in constraint:
                    expected_type = constraint['type']
                    if not isinstance(value, expected_type):
                        return False
                
                # Check range
                if 'min' in constraint and value < constraint['min']:
                    return False
                if 'max' in constraint and value > constraint['max']:
                    return False
                
                # Check allowed values
                if 'allowed' in constraint and value not in constraint['allowed']:
                    return False
        
        return True
    
    def get_parameter_hash(self) -> str:
        """Get hash of parameters for caching."""
        param_str = json.dumps(self.parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio
        }


@dataclass
class StrategyInstance:
    """Container for strategy instance with metadata."""
    strategy: Any
    strategy_type: StrategyType
    parameters: StrategyParameters
    status: StrategyStatus = StrategyStatus.INACTIVE
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)
    risk_score: float = 0.0
    allocation: float = 0.0  # Portfolio allocation percentage
    active_positions: int = 0
    error_count: int = 0
    
    def touch(self):
        """Update last updated timestamp."""
        self.last_updated_at = time.time()
    
    @property
    def age_seconds(self) -> float:
        """Get instance age in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_healthy(self) -> bool:
        """Check if strategy is healthy."""
        return self.status in [StrategyStatus.ACTIVE, StrategyStatus.PAUSED] and self.error_count < 5


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, parameters: StrategyParameters):
        self.parameters = parameters
        self.status = StrategyStatus.INACTIVE
        self._logger = get_logger(f"strategy.{self.__class__.__name__}")
        self._metrics = get_metrics_collector().get_collector("trading")
        self._performance = StrategyPerformance()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize strategy."""
        pass
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals."""
        pass
    
    @abstractmethod
    async def execute_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trading signals."""
        pass
    
    @abstractmethod
    async def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance metrics."""
        pass
    
    @abstractmethod
    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for optimization."""
        pass
    
    async def start(self) -> bool:
        """Start strategy execution."""
        try:
            if await self.initialize():
                self.status = StrategyStatus.ACTIVE
                self._logger.info("Strategy started successfully")
                return True
            else:
                self.status = StrategyStatus.ERROR
                self._logger.error("Strategy initialization failed")
                return False
        except Exception as e:
            self.status = StrategyStatus.ERROR
            self._logger.error("Strategy start error", error=str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop strategy execution."""
        try:
            self.status = StrategyStatus.INACTIVE
            self._logger.info("Strategy stopped")
            return True
        except Exception as e:
            self._logger.error("Strategy stop error", error=str(e))
            return False
    
    async def pause(self) -> bool:
        """Pause strategy execution."""
        try:
            if self.status == StrategyStatus.ACTIVE:
                self.status = StrategyStatus.PAUSED
                self._logger.info("Strategy paused")
                return True
            return False
        except Exception as e:
            self._logger.error("Strategy pause error", error=str(e))
            return False
    
    async def resume(self) -> bool:
        """Resume strategy execution."""
        try:
            if self.status == StrategyStatus.PAUSED:
                self.status = StrategyStatus.ACTIVE
                self._logger.info("Strategy resumed")
                return True
            return False
        except Exception as e:
            self._logger.error("Strategy resume error", error=str(e))
            return False
    
    def get_performance(self) -> StrategyPerformance:
        """Get current performance metrics."""
        return self._performance


class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy implementation."""
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.lookback_period = parameters.parameters.get('lookback_period', 20)
        self.momentum_threshold = parameters.parameters.get('momentum_threshold', 0.02)
        self.stop_loss = parameters.parameters.get('stop_loss', 0.05)
        self.take_profit = parameters.parameters.get('take_profit', 0.10)
    
    async def initialize(self) -> bool:
        """Initialize momentum strategy."""
        self._logger.info("Initializing momentum strategy", 
                         lookback_period=self.lookback_period,
                         momentum_threshold=self.momentum_threshold)
        
        # Validate parameters
        if not self.parameters.validate():
            self._logger.error("Parameter validation failed")
            return False
        
        return True
    
    @cache_strategy_result("momentum", ttl=300)
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate momentum-based trading signals."""
        signals = []
        
        try:
            # Simulate momentum calculation
            price = market_data.get('price', 100.0)
            historical_prices = market_data.get('historical_prices', [95, 96, 97, 98, 99, 100])
            
            if len(historical_prices) >= self.lookback_period:
                momentum = (price - historical_prices[-self.lookback_period]) / historical_prices[-self.lookback_period]
                
                if momentum > self.momentum_threshold:
                    signals.append({
                        'action': 'buy',
                        'symbol': market_data.get('symbol', 'BTC/USDT'),
                        'strength': min(momentum / self.momentum_threshold, 1.0),
                        'price': price,
                        'timestamp': time.time(),
                        'strategy': 'momentum',
                        'stop_loss': price * (1 - self.stop_loss),
                        'take_profit': price * (1 + self.take_profit)
                    })
                elif momentum < -self.momentum_threshold:
                    signals.append({
                        'action': 'sell',
                        'symbol': market_data.get('symbol', 'BTC/USDT'),
                        'strength': min(abs(momentum) / self.momentum_threshold, 1.0),
                        'price': price,
                        'timestamp': time.time(),
                        'strategy': 'momentum',
                        'stop_loss': price * (1 + self.stop_loss),
                        'take_profit': price * (1 - self.take_profit)
                    })
            
            self._logger.debug("Generated momentum signals", signal_count=len(signals))
            
        except Exception as e:
            self._logger.error("Error generating momentum signals", error=str(e))
        
        return signals
    
    async def execute_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute momentum trading signals."""
        results = []
        
        for signal in signals:
            try:
                # Simulate order execution
                await asyncio.sleep(0.01)  # Simulate execution time
                
                result = {
                    'signal': signal,
                    'order_id': f"momentum_{int(time.time() * 1000)}",
                    'status': 'executed',
                    'execution_price': signal['price'] * (1 + 0.001),  # Simulate slippage
                    'execution_time': time.time(),
                    'commission': signal['price'] * 0.001  # 0.1% commission
                }
                
                results.append(result)
                
                if self._metrics:
                    self._metrics.record_trade(
                        signal['symbol'],
                        signal['action'],
                        1.0,  # Quantity
                        result['execution_price']
                    )
                
            except Exception as e:
                self._logger.error("Error executing signal", signal=signal, error=str(e))
                results.append({
                    'signal': signal,
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': time.time()
                })
        
        return results
    
    async def update_performance(self, trade_result: Dict[str, Any]):
        """Update momentum strategy performance."""
        try:
            # Update performance metrics based on trade result
            if trade_result.get('status') == 'executed':
                self._performance.total_trades += 1
                
                # Calculate profit/loss (simplified)
                entry_price = trade_result['execution_price']
                exit_price = entry_price * 1.02  # Simulate 2% profit
                pnl = (exit_price - entry_price) / entry_price
                
                self._performance.total_return += pnl
                
                if pnl > 0:
                    win_rate = (self._performance.win_rate * (self._performance.total_trades - 1) + 1) / self._performance.total_trades
                    self._performance.win_rate = win_rate
                
                # Update other metrics (simplified)
                self._performance.sharpe_ratio = self._performance.total_return / max(0.1, abs(self._performance.total_return) * 0.5)
                
        except Exception as e:
            self._logger.error("Error updating performance", error=str(e))
    
    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for optimization."""
        return {
            'lookback_period': {'type': int, 'min': 5, 'max': 100},
            'momentum_threshold': {'type': float, 'min': 0.001, 'max': 0.1},
            'stop_loss': {'type': float, 'min': 0.01, 'max': 0.2},
            'take_profit': {'type': float, 'min': 0.02, 'max': 0.5}
        }


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy implementation."""
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.lookback_period = parameters.parameters.get('lookback_period', 30)
        self.deviation_threshold = parameters.parameters.get('deviation_threshold', 2.0)
        self.stop_loss = parameters.parameters.get('stop_loss', 0.03)
        self.take_profit = parameters.parameters.get('take_profit', 0.06)
    
    async def initialize(self) -> bool:
        """Initialize mean reversion strategy."""
        self._logger.info("Initializing mean reversion strategy",
                         lookback_period=self.lookback_period,
                         deviation_threshold=self.deviation_threshold)
        return self.parameters.validate()
    
    @cache_strategy_result("mean_reversion", ttl=300)
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mean reversion trading signals."""
        signals = []
        
        try:
            price = market_data.get('price', 100.0)
            historical_prices = market_data.get('historical_prices', [95, 96, 97, 98, 99, 100])
            
            if len(historical_prices) >= self.lookback_period:
                mean_price = sum(historical_prices[-self.lookback_period:]) / self.lookback_period
                std_dev = (sum((p - mean_price) ** 2 for p in historical_prices[-self.lookback_period:]) / self.lookback_period) ** 0.5
                
                if std_dev > 0:
                    z_score = (price - mean_price) / std_dev
                    
                    if z_score > self.deviation_threshold:
                        # Price is too high, expect reversion
                        signals.append({
                            'action': 'sell',
                            'symbol': market_data.get('symbol', 'BTC/USDT'),
                            'strength': min(abs(z_score) / self.deviation_threshold, 1.0),
                            'price': price,
                            'timestamp': time.time(),
                            'strategy': 'mean_reversion',
                            'stop_loss': price * (1 + self.stop_loss),
                            'take_profit': price * (1 - self.take_profit)
                        })
                    elif z_score < -self.deviation_threshold:
                        # Price is too low, expect reversion
                        signals.append({
                            'action': 'buy',
                            'symbol': market_data.get('symbol', 'BTC/USDT'),
                            'strength': min(abs(z_score) / self.deviation_threshold, 1.0),
                            'price': price,
                            'timestamp': time.time(),
                            'strategy': 'mean_reversion',
                            'stop_loss': price * (1 - self.stop_loss),
                            'take_profit': price * (1 + self.take_profit)
                        })
        
        except Exception as e:
            self._logger.error("Error generating mean reversion signals", error=str(e))
        
        return signals
    
    async def execute_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute mean reversion trading signals."""
        results = []
        
        for signal in signals:
            try:
                await asyncio.sleep(0.01)
                
                result = {
                    'signal': signal,
                    'order_id': f"mean_reversion_{int(time.time() * 1000)}",
                    'status': 'executed',
                    'execution_price': signal['price'] * (1 + 0.0005),  # Minimal slippage for mean reversion
                    'execution_time': time.time(),
                    'commission': signal['price'] * 0.001
                }
                
                results.append(result)
                
            except Exception as e:
                self._logger.error("Error executing mean reversion signal", error=str(e))
                results.append({
                    'signal': signal,
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': time.time()
                })
        
        return results
    
    async def update_performance(self, trade_result: Dict[str, Any]):
        """Update mean reversion strategy performance."""
        # Similar to momentum strategy but with different metrics
        if trade_result.get('status') == 'executed':
            self._performance.total_trades += 1
            # Update performance metrics...
    
    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter constraints for optimization."""
        return {
            'lookback_period': {'type': int, 'min': 10, 'max': 200},
            'deviation_threshold': {'type': float, 'min': 1.0, 'max': 4.0},
            'stop_loss': {'type': float, 'min': 0.01, 'max': 0.1},
            'take_profit': {'type': float, 'min': 0.02, 'max': 0.2}
        }


class StrategyBuilder:
    """Builder for trading strategies with parameter optimization."""
    
    def __init__(self):
        self._strategy_type: Optional[StrategyType] = None
        self._parameters: Optional[StrategyParameters] = None
        self._custom_class: Optional[Type[BaseStrategy]] = None
    
    def with_type(self, strategy_type: StrategyType) -> 'StrategyBuilder':
        """Set strategy type."""
        self._strategy_type = strategy_type
        return self
    
    def with_parameters(self, parameters: StrategyParameters) -> 'StrategyBuilder':
        """Set strategy parameters."""
        self._parameters = parameters
        return self
    
    def with_custom_class(self, strategy_class: Type[BaseStrategy]) -> 'StrategyBuilder':
        """Set custom strategy class."""
        self._custom_class = strategy_class
        return self
    
    def build(self) -> BaseStrategy:
        """Build strategy instance."""
        if not self._parameters:
            raise ValueError("Strategy parameters are required")
        
        if self._custom_class:
            return self._custom_class(self._parameters)
        
        if not self._strategy_type:
            raise ValueError("Strategy type is required")
        
        strategy_map = {
            StrategyType.MOMENTUM: MomentumStrategy,
            StrategyType.MEAN_REVERSION: MeanReversionStrategy,
            # Add more strategies as needed
        }
        
        strategy_class = strategy_map.get(self._strategy_type)
        if not strategy_class:
            raise ValueError(f"Unsupported strategy type: {self._strategy_type}")
        
        return strategy_class(self._parameters)


class StrategyFactory:
    """
    Advanced factory for trading strategies.
    
    Features:
    - Multiple strategy types with auto-selection
    - Parameter optimization and backtesting
    - Performance monitoring and comparison
    - Dynamic strategy loading and hot-swap
    - Risk-aware strategy management
    """
    
    def __init__(self):
        self._instances: Dict[str, StrategyInstance] = {}
        self._strategy_map = {
            StrategyType.MOMENTUM: MomentumStrategy,
            StrategyType.MEAN_REVERSION: MeanReversionStrategy,
        }
        self._logger = get_logger(__name__)
        self._metrics = get_metrics_collector().get_collector("trading")
        self._performance_history: Dict[str, List[StrategyPerformance]] = {}
    
    def register_strategy(self, strategy_type: StrategyType, strategy_class: Type[BaseStrategy]):
        """Register a custom strategy class."""
        self._strategy_map[strategy_type] = strategy_class
        self._logger.info("Registered strategy class", strategy_type=strategy_type.value)
    
    @trace_sync(name="create_strategy", tags={"component": "factory"})
    async def create_strategy(self, strategy_type: StrategyType, 
                            parameters: StrategyParameters,
                            strategy_id: Optional[str] = None) -> BaseStrategy:
        """
        Create or retrieve strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            parameters: Strategy parameters
            strategy_id: Optional custom strategy ID
            
        Returns:
            Strategy instance
        """
        if not strategy_id:
            strategy_id = f"{strategy_type.value}_{parameters.get_parameter_hash()}"
        
        # Check for existing instance
        if strategy_id in self._instances:
            instance = self._instances[strategy_id]
            if instance.is_healthy:
                instance.touch()
                self._logger.debug("Reusing existing strategy instance", strategy_id=strategy_id)
                return instance.strategy
            else:
                # Remove unhealthy instance
                del self._instances[strategy_id]
        
        # Create new instance
        try:
            strategy = await self._create_new_strategy(strategy_type, parameters)
            
            instance = StrategyInstance(
                strategy=strategy,
                strategy_type=strategy_type,
                parameters=parameters,
                status=StrategyStatus.INACTIVE
            )
            
            self._instances[strategy_id] = instance
            
            self._logger.info("Created new strategy instance",
                            strategy_type=strategy_type.value,
                            strategy_id=strategy_id)
            
            if self._metrics:
                self._metrics.record_metric("strategy.created", 1, 
                                          tags={"strategy_type": strategy_type.value})
            
            return strategy
            
        except Exception as e:
            self._logger.error("Failed to create strategy instance",
                             strategy_type=strategy_type.value,
                             error=str(e))
            raise
    
    async def _create_new_strategy(self, strategy_type: StrategyType, 
                                 parameters: StrategyParameters) -> BaseStrategy:
        """Create new strategy instance."""
        strategy_class = self._strategy_map.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        
        return strategy_class(parameters)
    
    async def optimize_parameters(self, strategy_type: StrategyType,
                                base_parameters: StrategyParameters,
                                optimization_range: Dict[str, List[Any]],
                                optimization_metric: str = "sharpe_ratio") -> StrategyParameters:
        """
        Optimize strategy parameters using backtesting.
        
        Args:
            strategy_type: Type of strategy to optimize
            base_parameters: Base parameters to start from
            optimization_range: Range of values for each parameter
            optimization_metric: Metric to optimize
            
        Returns:
            Optimized parameters
        """
        best_parameters = base_parameters
        best_score = -float('inf')
        
        # Simple grid search (in production, use more sophisticated optimization)
        for param_name, value_range in optimization_range.items():
            for value in value_range:
                try:
                    # Create test parameters
                    test_params = StrategyParameters(
                        parameters={**base_parameters.parameters, param_name: value},
                        constraints=base_parameters.constraints,
                        optimization_targets=base_parameters.optimization_targets
                    )
                    
                    # Create and backtest strategy
                    strategy = await self._create_new_strategy(strategy_type, test_params)
                    performance = await self._backtest_strategy(strategy)
                    
                    # Evaluate performance
                    score = getattr(performance, optimization_metric, 0.0)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = test_params
                        
                        self._logger.info("Found better parameters",
                                        param_name=param_name,
                                        value=value,
                                        score=score)
                
                except Exception as e:
                    self._logger.error("Error in parameter optimization",
                                     param_name=param_name,
                                     value=value,
                                     error=str(e))
        
        return best_parameters
    
    async def _backtest_strategy(self, strategy: BaseStrategy) -> StrategyPerformance:
        """Backtest strategy on historical data."""
        # Simplified backtesting implementation
        # In production, this would use historical market data
        
        await strategy.initialize()
        
        # Simulate trading for backtesting
        for i in range(100):  # 100 time periods
            market_data = {
                'price': 100 + i * 0.1,  # Trending price
                'historical_prices': [100 + j * 0.1 for j in range(max(0, i-30), i+1)],
                'symbol': 'BTC/USDT'
            }
            
            signals = await strategy.generate_signals(market_data)
            
            if signals:
                results = await strategy.execute_signals(signals)
                for result in results:
                    await strategy.update_performance(result)
        
        return strategy.get_performance()
    
    def get_all_strategies(self) -> List[StrategyInstance]:
        """Get all strategy instances."""
        return list(self._instances.values())
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[StrategyInstance]:
        """Get strategy instance by ID."""
        return self._instances.get(strategy_id)
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[StrategyInstance]:
        """Get strategies by type."""
        return [instance for instance in self._instances.values() 
                if instance.strategy_type == strategy_type]
    
    async def performance_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across all strategies."""
        comparison = {}
        
        for strategy_id, instance in self._instances.items():
            comparison[strategy_id] = {
                'strategy_type': instance.strategy_type.value,
                'total_return': instance.performance.total_return,
                'sharpe_ratio': instance.performance.sharpe_ratio,
                'max_drawdown': instance.performance.max_drawdown,
                'win_rate': instance.performance.win_rate,
                'total_trades': instance.performance.total_trades,
                'status': instance.status.value,
                'age_seconds': instance.age_seconds
            }
        
        return comparison
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        total_strategies = len(self._instances)
        active_strategies = sum(1 for i in self._instances.values() 
                              if i.status == StrategyStatus.ACTIVE)
        
        strategy_types = {}
        for instance in self._instances.values():
            strategy_type = instance.strategy_type.value
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
        
        return {
            "total_strategies": total_strategies,
            "active_strategies": active_strategies,
            "supported_types": list(self._strategy_map.keys()),
            "strategy_types_count": strategy_types,
            "instance_details": {
                strategy_id: {
                    "strategy_type": instance.strategy_type.value,
                    "status": instance.status.value,
                    "performance": instance.performance.to_dict(),
                    "age_seconds": instance.age_seconds,
                    "risk_score": instance.risk_score,
                    "allocation": instance.allocation,
                    "active_positions": instance.active_positions
                }
                for strategy_id, instance in self._instances.items()
            }
        }


# Global strategy factory instance
_global_strategy_factory: Optional[StrategyFactory] = None


def get_strategy_factory() -> StrategyFactory:
    """Get the global strategy factory instance."""
    global _global_strategy_factory
    if _global_strategy_factory is None:
        _global_strategy_factory = StrategyFactory()
    return _global_strategy_factory


def create_strategy(strategy_type: StrategyType, parameters: Dict[str, Any]) -> BaseStrategy:
    """Convenience function to create strategy."""
    strategy_params = StrategyParameters(parameters=parameters)
    return StrategyBuilder().with_type(strategy_type).with_parameters(strategy_params).build()