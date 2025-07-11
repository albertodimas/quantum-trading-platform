"""
Execution Algorithms for the Order Management System.

Implements advanced order execution algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg Orders
- Implementation Shortfall
- Participation Rate
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal
import numpy as np
from abc import ABC, abstractmethod

from ..core.observability.logger import get_logger
from ..core.observability.metrics import get_metrics_collector
from ..core.observability.tracing import trace_async
from ..exchange.exchange_interface import OrderType, OrderSide, Order
from .order_manager import OrderManager, OrderRequest


logger = get_logger(__name__)


class AlgorithmType(Enum):
    """Execution algorithm types."""
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    PARTICIPATION_RATE = "participation_rate"
    POV = "percentage_of_volume"
    SMART = "smart"


@dataclass
class AlgorithmConfig:
    """Base configuration for execution algorithms."""
    algorithm_type: AlgorithmType
    symbol: str
    side: OrderSide
    total_quantity: Decimal
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_order_size: Decimal = Decimal("0.001")
    max_order_size: Optional[Decimal] = None
    price_limit: Optional[Decimal] = None
    urgency: float = 0.5  # 0 = patient, 1 = aggressive
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TWAPConfig(AlgorithmConfig):
    """TWAP algorithm configuration."""
    num_slices: int = 10
    randomize_size: bool = True
    randomize_timing: bool = True
    size_variance: float = 0.2  # ±20% size variation
    time_variance: float = 0.1  # ±10% time variation


@dataclass
class VWAPConfig(AlgorithmConfig):
    """VWAP algorithm configuration."""
    historical_volume_profile: Optional[List[float]] = None
    volume_participation: float = 0.1  # 10% of market volume
    adapt_to_real_time: bool = True
    min_participation: float = 0.05
    max_participation: float = 0.25


@dataclass
class IcebergConfig(AlgorithmConfig):
    """Iceberg order configuration."""
    visible_quantity: Decimal = Decimal("0")
    reload_quantity: Decimal = Decimal("0")
    price_offset: Decimal = Decimal("0")  # Price improvement offset
    auto_adjust_price: bool = True


@dataclass
class AlgorithmExecution:
    """Execution tracking for algorithms."""
    algorithm_id: str
    config: AlgorithmConfig
    child_orders: List[str] = field(default_factory=list)
    executed_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "active"
    metrics: Dict[str, Any] = field(default_factory=dict)


class ExecutionAlgorithm(ABC):
    """Abstract base class for execution algorithms."""
    
    def __init__(self, order_manager: OrderManager, config: AlgorithmConfig):
        self.order_manager = order_manager
        self.config = config
        self._logger = get_logger(f"{self.__class__.__name__}.{config.symbol}")
        self._metrics = get_metrics_collector().get_collector("trading")
        self._execution = AlgorithmExecution(
            algorithm_id=f"{config.algorithm_type.value}_{config.symbol}_{int(time.time())}",
            config=config
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    @abstractmethod
    async def execute(self) -> AlgorithmExecution:
        """Execute the algorithm."""
        pass
    
    async def start(self) -> str:
        """Start algorithm execution."""
        if self._running:
            raise RuntimeError("Algorithm already running")
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        
        self._logger.info("Algorithm started",
                         algorithm_id=self._execution.algorithm_id,
                         type=self.config.algorithm_type.value,
                         symbol=self.config.symbol,
                         quantity=str(self.config.total_quantity))
        
        return self._execution.algorithm_id
    
    async def stop(self):
        """Stop algorithm execution."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._execution.end_time = datetime.utcnow()
        self._execution.status = "stopped"
        
        self._logger.info("Algorithm stopped",
                         algorithm_id=self._execution.algorithm_id,
                         executed_quantity=str(self._execution.executed_quantity))
    
    async def _run(self):
        """Run the algorithm execution loop."""
        try:
            await self.execute()
            self._execution.status = "completed"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.error("Algorithm execution failed",
                             algorithm_id=self._execution.algorithm_id,
                             error=str(e))
            self._execution.status = "failed"
            raise
        finally:
            self._execution.end_time = datetime.utcnow()
    
    async def _submit_child_order(self, quantity: Decimal, 
                                 order_type: OrderType = OrderType.MARKET,
                                 price: Optional[Decimal] = None) -> Optional[str]:
        """Submit a child order as part of the algorithm."""
        if quantity <= 0:
            return None
        
        request = OrderRequest(
            symbol=self.config.symbol,
            side=self.config.side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            strategy_id=self._execution.algorithm_id,
            metadata={"algorithm": self.config.algorithm_type.value}
        )
        
        try:
            order_id = await self.order_manager.submit_order(request)
            self._execution.child_orders.append(order_id)
            
            if self._metrics:
                self._metrics.record_metric("algo.child_orders.submitted", 1, tags={
                    "algorithm": self.config.algorithm_type.value,
                    "symbol": self.config.symbol
                })
            
            return order_id
            
        except Exception as e:
            self._logger.error("Failed to submit child order",
                             algorithm_id=self._execution.algorithm_id,
                             error=str(e))
            return None
    
    async def _wait_for_fill(self, order_id: str, timeout: float = 30) -> bool:
        """Wait for order to be filled."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            tracking = await self.order_manager.get_order_status(order_id)
            
            if tracking and tracking.status.value in ["filled", "cancelled", "rejected"]:
                if tracking.status.value == "filled":
                    # Update execution metrics
                    self._execution.executed_quantity += tracking.total_executed_quantity
                    
                    if tracking.average_price:
                        # Update weighted average price
                        if self._execution.average_price == 0:
                            self._execution.average_price = tracking.average_price
                        else:
                            total_value = (self._execution.average_price * 
                                         (self._execution.executed_quantity - tracking.total_executed_quantity))
                            total_value += tracking.average_price * tracking.total_executed_quantity
                            self._execution.average_price = total_value / self._execution.executed_quantity
                
                return tracking.status.value == "filled"
            
            await asyncio.sleep(1)
        
        return False
    
    def get_remaining_quantity(self) -> Decimal:
        """Get remaining quantity to execute."""
        return self.config.total_quantity - self._execution.executed_quantity
    
    def get_execution_progress(self) -> float:
        """Get execution progress percentage."""
        if self.config.total_quantity == 0:
            return 100.0
        return float(self._execution.executed_quantity / self.config.total_quantity * 100)


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price algorithm.
    
    Splits orders evenly across time periods to minimize market impact.
    """
    
    def __init__(self, order_manager: OrderManager, config: TWAPConfig):
        super().__init__(order_manager, config)
        self.twap_config = config
    
    @trace_async(name="twap_execute", tags={"algorithm": "twap"})
    async def execute(self) -> AlgorithmExecution:
        """Execute TWAP algorithm."""
        # Calculate time parameters
        start_time = self.config.start_time or datetime.utcnow()
        end_time = self.config.end_time or (start_time + timedelta(hours=1))
        duration = (end_time - start_time).total_seconds()
        
        if duration <= 0:
            raise ValueError("Invalid time window for TWAP")
        
        # Calculate slice parameters
        slice_duration = duration / self.twap_config.num_slices
        base_slice_quantity = self.config.total_quantity / self.twap_config.num_slices
        
        self._logger.info("Starting TWAP execution",
                         slices=self.twap_config.num_slices,
                         slice_duration=slice_duration,
                         base_quantity=str(base_slice_quantity))
        
        # Execute slices
        for i in range(self.twap_config.num_slices):
            if not self._running:
                break
            
            # Calculate slice quantity with optional randomization
            slice_quantity = base_slice_quantity
            if self.twap_config.randomize_size:
                variance = 1 + (np.random.uniform(-1, 1) * self.twap_config.size_variance)
                slice_quantity = base_slice_quantity * Decimal(str(variance))
            
            # Ensure we don't exceed total quantity
            remaining = self.get_remaining_quantity()
            slice_quantity = min(slice_quantity, remaining)
            
            if slice_quantity < self.config.min_order_size:
                break
            
            # Submit slice order
            order_id = await self._submit_child_order(slice_quantity)
            
            if order_id:
                # Wait for fill or timeout
                filled = await self._wait_for_fill(order_id, timeout=slice_duration * 0.8)
                
                if not filled:
                    self._logger.warning("Slice order not filled",
                                       slice=i,
                                       order_id=order_id)
            
            # Wait for next slice with optional randomization
            if i < self.twap_config.num_slices - 1:
                wait_time = slice_duration
                if self.twap_config.randomize_timing:
                    variance = 1 + (np.random.uniform(-1, 1) * self.twap_config.time_variance)
                    wait_time *= variance
                
                await asyncio.sleep(wait_time)
        
        # Submit any remaining quantity
        remaining = self.get_remaining_quantity()
        if remaining >= self.config.min_order_size:
            await self._submit_child_order(remaining)
        
        self._logger.info("TWAP execution completed",
                         executed_quantity=str(self._execution.executed_quantity),
                         average_price=str(self._execution.average_price))
        
        return self._execution


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price algorithm.
    
    Distributes orders according to historical or real-time volume patterns.
    """
    
    def __init__(self, order_manager: OrderManager, config: VWAPConfig):
        super().__init__(order_manager, config)
        self.vwap_config = config
        self._volume_profile = self._initialize_volume_profile()
    
    def _initialize_volume_profile(self) -> List[float]:
        """Initialize volume profile."""
        if self.vwap_config.historical_volume_profile:
            return self.vwap_config.historical_volume_profile
        
        # Default intraday volume profile (U-shaped)
        hours = 6.5  # Trading hours
        points = int(hours * 4)  # 15-minute intervals
        
        profile = []
        for i in range(points):
            # U-shaped curve
            x = i / points
            volume = 0.3 + 0.7 * (1 - abs(2 * x - 1)) ** 2
            profile.append(volume)
        
        # Normalize
        total = sum(profile)
        return [v / total for v in profile]
    
    @trace_async(name="vwap_execute", tags={"algorithm": "vwap"})
    async def execute(self) -> AlgorithmExecution:
        """Execute VWAP algorithm."""
        start_time = self.config.start_time or datetime.utcnow()
        end_time = self.config.end_time or (start_time + timedelta(hours=6.5))
        
        total_intervals = len(self._volume_profile)
        interval_duration = (end_time - start_time).total_seconds() / total_intervals
        
        self._logger.info("Starting VWAP execution",
                         intervals=total_intervals,
                         interval_duration=interval_duration)
        
        # Track market volume for adaptive execution
        market_volume_tracker = []
        
        for i, volume_weight in enumerate(self._volume_profile):
            if not self._running:
                break
            
            # Calculate target quantity for this interval
            target_quantity = self.config.total_quantity * Decimal(str(volume_weight))
            
            # Adjust based on real-time volume if enabled
            if self.vwap_config.adapt_to_real_time and market_volume_tracker:
                # Simple adaptation based on actual vs expected volume
                actual_volume_ratio = self._calculate_volume_ratio(market_volume_tracker, i)
                target_quantity *= Decimal(str(actual_volume_ratio))
            
            # Ensure within participation limits
            # This would need real market volume data in production
            target_quantity = self._apply_participation_limits(target_quantity)
            
            # Ensure we don't exceed remaining quantity
            remaining = self.get_remaining_quantity()
            target_quantity = min(target_quantity, remaining)
            
            if target_quantity < self.config.min_order_size:
                continue
            
            # Split into smaller orders within interval
            num_orders = max(1, int(target_quantity / (self.config.max_order_size or target_quantity)))
            order_size = target_quantity / num_orders
            
            interval_start = time.time()
            
            for j in range(num_orders):
                if not self._running:
                    break
                
                # Submit order
                order_id = await self._submit_child_order(order_size, OrderType.LIMIT)
                
                if order_id:
                    # Wait for fill with shorter timeout
                    await self._wait_for_fill(order_id, timeout=interval_duration / num_orders * 0.8)
                
                # Pace orders within interval
                if j < num_orders - 1:
                    await asyncio.sleep(interval_duration / num_orders)
            
            # Wait for next interval
            elapsed = time.time() - interval_start
            if elapsed < interval_duration and i < total_intervals - 1:
                await asyncio.sleep(interval_duration - elapsed)
        
        # Submit any remaining quantity
        remaining = self.get_remaining_quantity()
        if remaining >= self.config.min_order_size:
            await self._submit_child_order(remaining)
        
        self._logger.info("VWAP execution completed",
                         executed_quantity=str(self._execution.executed_quantity),
                         average_price=str(self._execution.average_price))
        
        return self._execution
    
    def _calculate_volume_ratio(self, volume_tracker: List[float], current_interval: int) -> float:
        """Calculate volume ratio for adaptation."""
        # Simplified - in production would use real market data
        return 1.0
    
    def _apply_participation_limits(self, quantity: Decimal) -> Decimal:
        """Apply volume participation limits."""
        # In production, would check against real market volume
        # For now, just ensure within configured limits
        return quantity


class IcebergAlgorithm(ExecutionAlgorithm):
    """
    Iceberg order algorithm.
    
    Shows only a small portion of the total order to the market.
    """
    
    def __init__(self, order_manager: OrderManager, config: IcebergConfig):
        super().__init__(order_manager, config)
        self.iceberg_config = config
    
    @trace_async(name="iceberg_execute", tags={"algorithm": "iceberg"})
    async def execute(self) -> AlgorithmExecution:
        """Execute Iceberg algorithm."""
        # Determine visible and reload quantities
        visible_qty = self.iceberg_config.visible_quantity
        if visible_qty == 0:
            # Default to 5% of total or minimum size
            visible_qty = max(
                self.config.min_order_size,
                self.config.total_quantity * Decimal("0.05")
            )
        
        reload_qty = self.iceberg_config.reload_quantity
        if reload_qty == 0:
            reload_qty = visible_qty
        
        self._logger.info("Starting Iceberg execution",
                         visible_quantity=str(visible_qty),
                         reload_quantity=str(reload_qty))
        
        current_price = self.config.price_limit
        price_offset = self.iceberg_config.price_offset
        
        while self._running and self.get_remaining_quantity() > 0:
            # Calculate order size
            remaining = self.get_remaining_quantity()
            order_size = min(visible_qty, remaining)
            
            if order_size < self.config.min_order_size:
                break
            
            # Determine price
            if self.iceberg_config.auto_adjust_price and not current_price:
                # Get market price
                try:
                    current_price = await self._get_best_price()
                except Exception as e:
                    self._logger.error(f"Failed to get market price: {e}")
                    await asyncio.sleep(5)
                    continue
            
            # Apply price offset for passive execution
            if current_price and price_offset:
                if self.config.side == OrderSide.BUY:
                    current_price -= price_offset
                else:
                    current_price += price_offset
            
            # Submit visible order
            order_id = await self._submit_child_order(
                order_size,
                OrderType.LIMIT if current_price else OrderType.MARKET,
                current_price
            )
            
            if order_id:
                # Wait for fill
                filled = await self._wait_for_fill(order_id, timeout=60)
                
                if filled:
                    # Reload with next slice
                    await asyncio.sleep(np.random.uniform(0.5, 2))  # Random delay
                else:
                    # Adjust price if needed
                    if self.iceberg_config.auto_adjust_price:
                        if self.config.side == OrderSide.BUY:
                            current_price = current_price * Decimal("1.001")  # Increase by 0.1%
                        else:
                            current_price = current_price * Decimal("0.999")  # Decrease by 0.1%
            else:
                await asyncio.sleep(5)
        
        self._logger.info("Iceberg execution completed",
                         executed_quantity=str(self._execution.executed_quantity),
                         average_price=str(self._execution.average_price))
        
        return self._execution
    
    async def _get_best_price(self) -> Decimal:
        """Get best price from market."""
        # In production, would get from market data
        # For now, return a placeholder
        return Decimal("50000")  # Placeholder


class AlgorithmFactory:
    """Factory for creating execution algorithms."""
    
    @staticmethod
    def create_twap(order_manager: OrderManager, 
                   symbol: str,
                   side: OrderSide,
                   quantity: Decimal,
                   duration_minutes: int = 60,
                   **kwargs) -> TWAPAlgorithm:
        """Create TWAP algorithm."""
        config = TWAPConfig(
            algorithm_type=AlgorithmType.TWAP,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=duration_minutes),
            **kwargs
        )
        return TWAPAlgorithm(order_manager, config)
    
    @staticmethod
    def create_vwap(order_manager: OrderManager,
                   symbol: str,
                   side: OrderSide,
                   quantity: Decimal,
                   **kwargs) -> VWAPAlgorithm:
        """Create VWAP algorithm."""
        config = VWAPConfig(
            algorithm_type=AlgorithmType.VWAP,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            **kwargs
        )
        return VWAPAlgorithm(order_manager, config)
    
    @staticmethod
    def create_iceberg(order_manager: OrderManager,
                      symbol: str,
                      side: OrderSide,
                      quantity: Decimal,
                      visible_percentage: float = 0.05,
                      **kwargs) -> IcebergAlgorithm:
        """Create Iceberg algorithm."""
        visible_qty = quantity * Decimal(str(visible_percentage))
        
        config = IcebergConfig(
            algorithm_type=AlgorithmType.ICEBERG,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            visible_quantity=visible_qty,
            **kwargs
        )
        return IcebergAlgorithm(order_manager, config)