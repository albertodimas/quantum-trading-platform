"""
Execution Algorithms for Order Management

Implements TWAP, VWAP, and other sophisticated execution algorithms
for optimal order execution with minimal market impact.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import math
import statistics

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderStatus, OrderSide, OrderType, TimeInForce, MarketData
from .models import Order, ExecutionReport
from .order_manager import OrderManager

logger = get_logger(__name__)


class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    POV = "pov"    # Percentage of Volume
    ICEBERG = "iceberg"  # Iceberg orders
    SMART = "smart"  # Smart order routing
    MARKET = "market"  # Immediate market execution


@dataclass
class ExecutionParameters:
    """Parameters for execution algorithms"""
    algorithm: ExecutionAlgorithm
    total_quantity: Decimal
    time_horizon: int  # minutes
    participation_rate: Optional[float] = None  # % of market volume
    price_limit: Optional[Decimal] = None
    slice_size: Optional[Decimal] = None
    intervals: Optional[int] = None
    urgency: float = 0.5  # 0 = passive, 1 = aggressive
    risk_tolerance: float = 0.3  # 0 = risk averse, 1 = risk seeking
    max_spread_bps: Optional[int] = None  # max spread in basis points
    min_fill_size: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionSlice:
    """Individual execution slice"""
    slice_id: str
    quantity: Decimal
    target_time: datetime
    price_guidance: Optional[Decimal] = None
    urgency_factor: float = 0.5
    max_deviation_bps: Optional[int] = None
    status: str = "pending"
    order_id: Optional[str] = None
    actual_quantity: Decimal = Decimal("0")
    actual_price: Optional[Decimal] = None
    executed_at: Optional[datetime] = None


@dataclass
class ExecutionState:
    """Execution state tracking"""
    algorithm_id: str
    symbol: str
    side: OrderSide
    total_quantity: Decimal
    executed_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    slices: List[ExecutionSlice] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"
    market_data_history: List[MarketData] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@injectable
class ExecutionEngine:
    """
    Advanced execution engine with multiple algorithms.
    
    Features:
    - TWAP and VWAP execution strategies
    - Smart order routing and iceberg orders
    - Real-time adaptation to market conditions
    - Performance tracking and optimization
    """
    
    def __init__(
        self,
        order_manager: OrderManager = inject()
    ):
        """Initialize execution engine."""
        self.order_manager = order_manager
        
        # Active executions tracking
        self._active_executions: Dict[str, ExecutionState] = {}
        
        # Algorithm implementations
        self._algorithms = {
            ExecutionAlgorithm.TWAP: self._execute_twap,
            ExecutionAlgorithm.VWAP: self._execute_vwap,
            ExecutionAlgorithm.POV: self._execute_pov,
            ExecutionAlgorithm.ICEBERG: self._execute_iceberg,
            ExecutionAlgorithm.SMART: self._execute_smart,
            ExecutionAlgorithm.MARKET: self._execute_market,
        }
        
        # Market data for adaptive execution
        self._market_data_cache: Dict[str, List[MarketData]] = {}
        
        # Performance tracking
        self._execution_history: List[Dict] = []
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info("Execution engine initialized")
    
    async def start(self) -> None:
        """Start execution engine background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background monitoring
        self._background_tasks = [
            asyncio.create_task(self._execution_monitor()),
            asyncio.create_task(self._market_data_processor()),
            asyncio.create_task(self._performance_calculator()),
        ]
        
        logger.info("Execution engine started")
    
    async def stop(self) -> None:
        """Stop execution engine"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Execution engine stopped")
    
    async def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> str:
        """
        Execute order using specified algorithm.
        
        Args:
            symbol: Trading symbol
            side: Order side
            parameters: Execution parameters
            market_data_callback: Callback for market data updates
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create execution state
        execution_state = ExecutionState(
            algorithm_id=execution_id,
            symbol=symbol,
            side=side,
            total_quantity=parameters.total_quantity,
            remaining_quantity=parameters.total_quantity
        )
        
        self._active_executions[execution_id] = execution_state
        
        logger.info(
            "Starting order execution",
            execution_id=execution_id,
            algorithm=parameters.algorithm.value,
            symbol=symbol,
            side=side.value,
            quantity=float(parameters.total_quantity),
            time_horizon=parameters.time_horizon
        )
        
        # Start algorithm execution
        algorithm_func = self._algorithms[parameters.algorithm]
        asyncio.create_task(algorithm_func(execution_id, parameters, market_data_callback))
        
        return execution_id
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel active execution"""
        execution_state = self._active_executions.get(execution_id)
        if not execution_state:
            return False
        
        # Cancel pending orders
        for slice_info in execution_state.slices:
            if slice_info.status == "pending" and slice_info.order_id:
                await self.order_manager.cancel_order(slice_info.order_id)
        
        execution_state.status = "cancelled"
        
        logger.info(f"Execution cancelled: {execution_id}")
        return True
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get execution status"""
        execution_state = self._active_executions.get(execution_id)
        if not execution_state:
            return None
        
        return {
            "execution_id": execution_id,
            "symbol": execution_state.symbol,
            "side": execution_state.side.value,
            "algorithm": execution_state.algorithm_id,
            "status": execution_state.status,
            "total_quantity": float(execution_state.total_quantity),
            "executed_quantity": float(execution_state.executed_quantity),
            "remaining_quantity": float(execution_state.remaining_quantity),
            "average_price": float(execution_state.average_price) if execution_state.average_price else None,
            "progress_percentage": float(execution_state.executed_quantity / execution_state.total_quantity * 100),
            "slices_count": len(execution_state.slices),
            "active_slices": len([s for s in execution_state.slices if s.status == "pending"]),
            "performance_metrics": execution_state.performance_metrics,
            "start_time": execution_state.start_time.isoformat()
        }
    
    # Algorithm Implementations
    
    async def _execute_twap(
        self,
        execution_id: str,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> None:
        """
        Time Weighted Average Price execution.
        
        Splits large order into smaller time-based slices to minimize market impact.
        """
        execution_state = self._active_executions[execution_id]
        
        # Calculate slicing parameters
        intervals = parameters.intervals or max(3, min(20, parameters.time_horizon // 5))
        slice_duration = timedelta(minutes=parameters.time_horizon / intervals)
        slice_quantity = parameters.total_quantity / intervals
        
        # Generate execution slices
        current_time = datetime.now(timezone.utc)
        
        for i in range(intervals):
            target_time = current_time + (slice_duration * i)
            
            slice_info = ExecutionSlice(
                slice_id=f"{execution_id}_slice_{i+1}",
                quantity=slice_quantity,
                target_time=target_time,
                urgency_factor=parameters.urgency,
                max_deviation_bps=parameters.max_spread_bps
            )
            
            execution_state.slices.append(slice_info)
        
        logger.info(
            f"TWAP execution plan created: {intervals} slices over {parameters.time_horizon} minutes",
            execution_id=execution_id,
            slice_quantity=float(slice_quantity)
        )
        
        # Execute slices with timing
        for slice_info in execution_state.slices:
            # Wait until target time
            now = datetime.now(timezone.utc)
            if slice_info.target_time > now:
                wait_seconds = (slice_info.target_time - now).total_seconds()
                await asyncio.sleep(wait_seconds)
            
            # Execute slice
            await self._execute_slice(execution_state, slice_info, parameters)
            
            # Check if execution is cancelled
            if execution_state.status == "cancelled":
                break
        
        # Mark execution as completed
        if execution_state.status != "cancelled":
            execution_state.status = "completed"
            await self._calculate_final_metrics(execution_state)
    
    async def _execute_vwap(
        self,
        execution_id: str,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> None:
        """
        Volume Weighted Average Price execution.
        
        Adapts slice sizes based on historical volume patterns and current market activity.
        """
        execution_state = self._active_executions[execution_id]
        
        # Get historical volume data for VWAP calculation
        volume_profile = await self._get_volume_profile(execution_state.symbol, parameters.time_horizon)
        
        # Calculate dynamic slice sizes based on volume
        intervals = parameters.intervals or len(volume_profile)
        total_expected_volume = sum(volume_profile)
        
        current_time = datetime.now(timezone.utc)
        slice_duration = timedelta(minutes=parameters.time_horizon / intervals)
        
        for i, volume_fraction in enumerate(volume_profile):
            # Calculate slice quantity based on volume proportion
            volume_weight = volume_fraction / total_expected_volume
            slice_quantity = parameters.total_quantity * Decimal(str(volume_weight))
            
            target_time = current_time + (slice_duration * i)
            
            slice_info = ExecutionSlice(
                slice_id=f"{execution_id}_vwap_{i+1}",
                quantity=slice_quantity,
                target_time=target_time,
                urgency_factor=parameters.urgency * (1 + volume_weight),  # More urgent in high volume
                max_deviation_bps=parameters.max_spread_bps
            )
            
            execution_state.slices.append(slice_info)
        
        logger.info(
            f"VWAP execution plan created based on volume profile",
            execution_id=execution_id,
            intervals=intervals,
            total_volume_weight=total_expected_volume
        )
        
        # Execute slices with volume-based timing
        for slice_info in execution_state.slices:
            # Wait until target time
            now = datetime.now(timezone.utc)
            if slice_info.target_time > now:
                wait_seconds = (slice_info.target_time - now).total_seconds()
                await asyncio.sleep(wait_seconds)
            
            # Execute slice with volume consideration
            await self._execute_slice(execution_state, slice_info, parameters)
            
            if execution_state.status == "cancelled":
                break
        
        if execution_state.status != "cancelled":
            execution_state.status = "completed"
            await self._calculate_final_metrics(execution_state)
    
    async def _execute_pov(
        self,
        execution_id: str,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> None:
        """
        Percentage of Volume execution.
        
        Maintains a target percentage of market volume to minimize impact.
        """
        execution_state = self._active_executions[execution_id]
        participation_rate = parameters.participation_rate or 0.1  # 10% default
        
        logger.info(
            f"POV execution started with {participation_rate:.1%} participation rate",
            execution_id=execution_id
        )
        
        # Monitor volume and execute adaptively
        check_interval = 30  # seconds
        
        while (execution_state.remaining_quantity > 0 and 
               execution_state.status == "active"):
            
            # Get current market volume
            current_volume = await self._get_current_volume(execution_state.symbol)
            
            # Calculate target quantity for this interval
            target_quantity = Decimal(str(current_volume * participation_rate))
            slice_quantity = min(target_quantity, execution_state.remaining_quantity)
            
            if slice_quantity > 0:
                slice_info = ExecutionSlice(
                    slice_id=f"{execution_id}_pov_{len(execution_state.slices)+1}",
                    quantity=slice_quantity,
                    target_time=datetime.now(timezone.utc),
                    urgency_factor=parameters.urgency
                )
                
                execution_state.slices.append(slice_info)
                await self._execute_slice(execution_state, slice_info, parameters)
            
            # Wait for next interval
            await asyncio.sleep(check_interval)
        
        if execution_state.status != "cancelled":
            execution_state.status = "completed"
            await self._calculate_final_metrics(execution_state)
    
    async def _execute_iceberg(
        self,
        execution_id: str,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> None:
        """
        Iceberg order execution.
        
        Shows only small portions of large orders to minimize market impact.
        """
        execution_state = self._active_executions[execution_id]
        
        # Calculate iceberg parameters
        visible_size = parameters.slice_size or (parameters.total_quantity * Decimal("0.05"))  # 5% visible
        visible_size = max(visible_size, parameters.min_fill_size or Decimal("1"))
        
        logger.info(
            f"Iceberg execution started with {float(visible_size)} visible size",
            execution_id=execution_id
        )
        
        slice_count = 0
        
        while (execution_state.remaining_quantity > 0 and 
               execution_state.status == "active"):
            
            slice_count += 1
            slice_quantity = min(visible_size, execution_state.remaining_quantity)
            
            slice_info = ExecutionSlice(
                slice_id=f"{execution_id}_iceberg_{slice_count}",
                quantity=slice_quantity,
                target_time=datetime.now(timezone.utc),
                urgency_factor=parameters.urgency
            )
            
            execution_state.slices.append(slice_info)
            await self._execute_slice(execution_state, slice_info, parameters)
            
            # Small delay between iceberg slices
            if execution_state.remaining_quantity > 0:
                await asyncio.sleep(2)
        
        if execution_state.status != "cancelled":
            execution_state.status = "completed"
            await self._calculate_final_metrics(execution_state)
    
    async def _execute_smart(
        self,
        execution_id: str,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> None:
        """
        Smart order routing execution.
        
        Dynamically chooses best execution strategy based on market conditions.
        """
        execution_state = self._active_executions[execution_id]
        
        # Analyze market conditions
        market_conditions = await self._analyze_market_conditions(execution_state.symbol)
        
        # Choose best algorithm based on conditions
        if market_conditions["volatility"] > 0.02:  # High volatility
            chosen_algorithm = ExecutionAlgorithm.ICEBERG
        elif market_conditions["volume"] > market_conditions["avg_volume"] * 1.5:  # High volume
            chosen_algorithm = ExecutionAlgorithm.VWAP
        else:
            chosen_algorithm = ExecutionAlgorithm.TWAP
        
        logger.info(
            f"Smart routing chose {chosen_algorithm.value} based on market conditions",
            execution_id=execution_id,
            volatility=market_conditions["volatility"],
            volume_ratio=market_conditions["volume"] / market_conditions["avg_volume"]
        )
        
        # Execute using chosen algorithm
        algorithm_func = self._algorithms[chosen_algorithm]
        await algorithm_func(execution_id, parameters, market_data_callback)
    
    async def _execute_market(
        self,
        execution_id: str,
        parameters: ExecutionParameters,
        market_data_callback: Optional[Callable] = None
    ) -> None:
        """
        Immediate market execution.
        
        Executes the entire order immediately at market prices.
        """
        execution_state = self._active_executions[execution_id]
        
        slice_info = ExecutionSlice(
            slice_id=f"{execution_id}_market",
            quantity=parameters.total_quantity,
            target_time=datetime.now(timezone.utc),
            urgency_factor=1.0  # Maximum urgency
        )
        
        execution_state.slices.append(slice_info)
        
        logger.info(f"Market execution for full quantity", execution_id=execution_id)
        
        await self._execute_slice(execution_state, slice_info, parameters, market_order=True)
        
        execution_state.status = "completed"
        await self._calculate_final_metrics(execution_state)
    
    # Helper Methods
    
    async def _execute_slice(
        self,
        execution_state: ExecutionState,
        slice_info: ExecutionSlice,
        parameters: ExecutionParameters,
        market_order: bool = False
    ) -> None:
        """Execute individual slice"""
        try:
            slice_info.status = "executing"
            
            # Determine order type and price
            if market_order:
                order_type = OrderType.MARKET
                price = None
            else:
                order_type = OrderType.LIMIT
                price = await self._calculate_slice_price(
                    execution_state.symbol,
                    execution_state.side,
                    slice_info.urgency_factor,
                    parameters.price_limit
                )
            
            # Create order
            order = await self.order_manager.create_order(
                symbol=execution_state.symbol,
                side=execution_state.side,
                order_type=order_type,
                quantity=slice_info.quantity,
                price=price,
                time_in_force=TimeInForce.GTC,
                metadata={
                    "execution_id": execution_state.algorithm_id,
                    "slice_id": slice_info.slice_id,
                    "algorithm": "execution_engine"
                }
            )
            
            slice_info.order_id = order.id
            
            logger.debug(
                f"Slice order created",
                execution_id=execution_state.algorithm_id,
                slice_id=slice_info.slice_id,
                order_id=order.id,
                quantity=float(slice_info.quantity),
                price=float(price) if price else "market"
            )
            
            # Monitor execution (simplified - would integrate with order manager events)
            await asyncio.sleep(1)  # Simulate execution time
            
            # Update execution state
            slice_info.actual_quantity = slice_info.quantity
            slice_info.actual_price = price or Decimal("100")  # Mock price
            slice_info.executed_at = datetime.now(timezone.utc)
            slice_info.status = "completed"
            
            execution_state.executed_quantity += slice_info.actual_quantity
            execution_state.remaining_quantity -= slice_info.actual_quantity
            
            # Update average price
            if execution_state.average_price is None:
                execution_state.average_price = slice_info.actual_price
            else:
                total_value = (execution_state.average_price * 
                             (execution_state.executed_quantity - slice_info.actual_quantity))
                total_value += slice_info.actual_price * slice_info.actual_quantity
                execution_state.average_price = total_value / execution_state.executed_quantity
            
        except Exception as e:
            logger.error(f"Slice execution failed: {str(e)}", execution_id=execution_state.algorithm_id)
            slice_info.status = "failed"
    
    async def _calculate_slice_price(
        self,
        symbol: str,
        side: OrderSide,
        urgency_factor: float,
        price_limit: Optional[Decimal]
    ) -> Decimal:
        """Calculate optimal price for slice execution"""
        # Mock implementation - would use real market data
        base_price = Decimal("100")  # Mock current price
        spread = Decimal("0.01")  # Mock spread
        
        if side == OrderSide.BUY:
            # For buy orders, use ask price with urgency adjustment
            target_price = base_price + (spread * Decimal(str(urgency_factor)))
        else:
            # For sell orders, use bid price with urgency adjustment
            target_price = base_price - (spread * Decimal(str(urgency_factor)))
        
        # Apply price limit if specified
        if price_limit:
            if side == OrderSide.BUY:
                target_price = min(target_price, price_limit)
            else:
                target_price = max(target_price, price_limit)
        
        return target_price
    
    async def _get_volume_profile(self, symbol: str, time_horizon: int) -> List[float]:
        """Get historical volume profile for VWAP calculation"""
        # Mock implementation - would use real historical data
        # Generate typical intraday volume pattern
        intervals = max(3, min(20, time_horizon // 5))
        
        # U-shaped volume pattern (high at open/close, low at midday)
        volume_profile = []
        for i in range(intervals):
            position = i / (intervals - 1)  # 0 to 1
            # U-shaped curve: high at 0 and 1, low at 0.5
            volume_factor = 1 - 4 * position * (1 - position) * 0.6
            volume_profile.append(volume_factor)
        
        return volume_profile
    
    async def _get_current_volume(self, symbol: str) -> float:
        """Get current market volume"""
        # Mock implementation
        return 1000.0  # Mock volume
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, float]:
        """Analyze current market conditions"""
        # Mock implementation - would use real market data
        return {
            "volatility": 0.015,  # 1.5% volatility
            "volume": 1200.0,
            "avg_volume": 1000.0,
            "spread_bps": 5.0,
            "momentum": 0.2
        }
    
    async def _calculate_final_metrics(self, execution_state: ExecutionState) -> None:
        """Calculate final execution performance metrics"""
        if not execution_state.slices:
            return
        
        # Calculate execution metrics
        total_executed = execution_state.executed_quantity
        total_planned = execution_state.total_quantity
        
        fill_rate = float(total_executed / total_planned) if total_planned > 0 else 0
        
        # Calculate time-based metrics
        execution_time = (datetime.now(timezone.utc) - execution_state.start_time).total_seconds()
        
        # Calculate price performance (simplified)
        if execution_state.average_price:
            # Mock benchmark price for comparison
            benchmark_price = Decimal("100")
            price_impact_bps = float(abs(execution_state.average_price - benchmark_price) / benchmark_price * 10000)
        else:
            price_impact_bps = 0
        
        execution_state.performance_metrics = {
            "fill_rate": fill_rate,
            "execution_time_seconds": execution_time,
            "average_price": float(execution_state.average_price) if execution_state.average_price else 0,
            "price_impact_bps": price_impact_bps,
            "slices_executed": len([s for s in execution_state.slices if s.status == "completed"]),
            "slices_failed": len([s for s in execution_state.slices if s.status == "failed"])
        }
        
        logger.info(
            f"Execution completed with metrics",
            execution_id=execution_state.algorithm_id,
            fill_rate=f"{fill_rate:.2%}",
            execution_time=f"{execution_time:.1f}s",
            price_impact=f"{price_impact_bps:.1f}bps"
        )
    
    # Background monitoring tasks
    
    async def _execution_monitor(self) -> None:
        """Monitor active executions"""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for execution_id, execution_state in list(self._active_executions.items()):
                    # Check for stale executions
                    if execution_state.status == "active":
                        elapsed = (current_time - execution_state.start_time).total_seconds()
                        if elapsed > 3600:  # 1 hour timeout
                            execution_state.status = "timeout"
                            logger.warning(f"Execution timed out: {execution_id}")
                    
                    # Clean up completed executions
                    if execution_state.status in ["completed", "cancelled", "timeout"]:
                        if execution_id in self._active_executions:
                            del self._active_executions[execution_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in execution monitor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _market_data_processor(self) -> None:
        """Process market data for adaptive execution"""
        while self._running:
            try:
                # Update market data cache for active symbols
                active_symbols = set(es.symbol for es in self._active_executions.values())
                
                for symbol in active_symbols:
                    # Mock market data update
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        bid_price=Decimal("99.99"),
                        ask_price=Decimal("100.01"),
                        last_price=Decimal("100.00"),
                        volume=Decimal("1000"),
                        bid_size=Decimal("100"),
                        ask_size=Decimal("100")
                    )
                    
                    if symbol not in self._market_data_cache:
                        self._market_data_cache[symbol] = []
                    
                    self._market_data_cache[symbol].append(market_data)
                    
                    # Keep only recent data
                    self._market_data_cache[symbol] = self._market_data_cache[symbol][-100:]
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in market data processor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _performance_calculator(self) -> None:
        """Calculate ongoing performance metrics"""
        while self._running:
            try:
                # Calculate performance for active executions
                for execution_state in self._active_executions.values():
                    if execution_state.executed_quantity > 0:
                        await self._calculate_final_metrics(execution_state)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance calculator: {str(e)}")
                await asyncio.sleep(5)