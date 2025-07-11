"""
Stop Loss and Take Profit Management System.

Features:
- Multiple stop loss types (fixed, trailing, percentage, ATR-based)
- Take profit management
- Dynamic stop adjustment
- Risk/reward ratio enforcement
- Partial position exits
- Emergency stop mechanisms
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import uuid

from ..core.observability.logger import get_logger
from ..core.observability.metrics import get_metrics_collector
from ..core.messaging.event_bus import get_event_bus, Event, EventPriority
from ..core.cache.cache_manager import CacheManager
from ..positions.position_tracker import Position, PositionSide
from ..orders.order_manager import OrderManager, OrderRequest, OrderSide, OrderType


logger = get_logger(__name__)


class StopType(Enum):
    """Types of stop orders."""
    FIXED = "fixed"
    TRAILING = "trailing"
    PERCENTAGE = "percentage"
    ATR = "atr"
    VOLATILITY = "volatility"
    TIME = "time"
    BRACKET = "bracket"


class StopStatus(Enum):
    """Stop order status."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PARTIALLY_FILLED = "partially_filled"


@dataclass
class StopLossConfig:
    """Stop loss configuration."""
    stop_type: StopType
    trigger_price: Optional[Decimal] = None
    stop_distance: Optional[Decimal] = None
    stop_percentage: Optional[Decimal] = None
    atr_multiplier: Optional[Decimal] = None
    trailing_distance: Optional[Decimal] = None
    trailing_percentage: Optional[Decimal] = None
    
    # Advanced features
    move_to_breakeven: bool = False
    breakeven_trigger: Optional[Decimal] = None
    partial_exits: List[Tuple[Decimal, Decimal]] = field(default_factory=list)  # [(price, percentage)]
    time_stop_hours: Optional[int] = None
    
    # Risk management
    max_loss_amount: Optional[Decimal] = None
    risk_reward_ratio: Optional[Decimal] = None


@dataclass
class TakeProfitConfig:
    """Take profit configuration."""
    target_price: Optional[Decimal] = None
    target_percentage: Optional[Decimal] = None
    target_amount: Optional[Decimal] = None
    
    # Multiple targets
    targets: List[Tuple[Decimal, Decimal]] = field(default_factory=list)  # [(price, percentage)]
    
    # Dynamic targets
    atr_multiplier: Optional[Decimal] = None
    volatility_adjusted: bool = False
    time_decay: bool = False


@dataclass
class StopOrder:
    """Stop order tracking."""
    stop_id: str
    position_id: str
    symbol: str
    side: OrderSide  # Side to execute when triggered
    quantity: Decimal
    stop_config: StopLossConfig
    take_profit_config: Optional[TakeProfitConfig] = None
    
    # State
    status: StopStatus = StopStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None
    
    # Tracking
    initial_price: Optional[Decimal] = None
    current_stop_price: Optional[Decimal] = None
    highest_price: Optional[Decimal] = None  # For trailing stops
    lowest_price: Optional[Decimal] = None   # For trailing stops
    
    # Execution
    triggered_orders: List[str] = field(default_factory=list)
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class StopLossManager:
    """
    Stop loss and take profit management system.
    
    Features:
    - Multiple stop types
    - Dynamic stop adjustment
    - Risk/reward management
    - Partial position exits
    """
    
    def __init__(self, order_manager: OrderManager,
                 cache_manager: Optional[CacheManager] = None):
        self.order_manager = order_manager
        self._logger = get_logger(self.__class__.__name__)
        self._metrics = get_metrics_collector().get_collector("risk")
        self._event_bus = get_event_bus()
        self._cache = cache_manager
        
        # Stop order storage
        self._stop_orders: Dict[str, StopOrder] = {}
        self._position_stops: Dict[str, List[str]] = {}  # position_id -> [stop_ids]
        self._symbol_stops: Dict[str, List[str]] = {}    # symbol -> [stop_ids]
        
        # Market data cache
        self._price_cache: Dict[str, Decimal] = {}
        self._atr_cache: Dict[str, Decimal] = {}
        self._volatility_cache: Dict[str, Decimal] = {}
        
        # Configuration
        self._check_interval = 1  # seconds
        self._emergency_stop_pct = Decimal("0.10")  # 10% emergency stop
        
        # Background task
        self._monitor_task = asyncio.create_task(self._monitor_stops())
    
    async def create_stop_loss(self, position: Position, 
                             stop_config: StopLossConfig,
                             take_profit_config: Optional[TakeProfitConfig] = None) -> str:
        """
        Create a stop loss order for a position.
        
        Args:
            position: Position to protect
            stop_config: Stop loss configuration
            take_profit_config: Optional take profit configuration
            
        Returns:
            Stop order ID
        """
        # Validate position
        if position.quantity == 0:
            raise ValueError("Cannot create stop for flat position")
        
        # Determine stop side (opposite of position)
        stop_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        
        # Create stop order
        stop_order = StopOrder(
            stop_id=str(uuid.uuid4()),
            position_id=f"{position.symbol}_{position.exchange}",
            symbol=position.symbol,
            side=stop_side,
            quantity=abs(position.quantity),
            stop_config=stop_config,
            take_profit_config=take_profit_config,
            initial_price=position.current_price or position.average_entry_price
        )
        
        # Calculate initial stop price
        stop_price = await self._calculate_stop_price(stop_order, position)
        stop_order.current_stop_price = stop_price
        
        # Initialize trailing stop tracking
        if stop_config.stop_type == StopType.TRAILING:
            if position.side == PositionSide.LONG:
                stop_order.highest_price = position.current_price
            else:
                stop_order.lowest_price = position.current_price
        
        # Validate stop price
        self._validate_stop_price(stop_order, position)
        
        # Store stop order
        self._stop_orders[stop_order.stop_id] = stop_order
        
        # Update indices
        position_id = stop_order.position_id
        if position_id not in self._position_stops:
            self._position_stops[position_id] = []
        self._position_stops[position_id].append(stop_order.stop_id)
        
        if stop_order.symbol not in self._symbol_stops:
            self._symbol_stops[stop_order.symbol] = []
        self._symbol_stops[stop_order.symbol].append(stop_order.stop_id)
        
        # Cache stop order
        if self._cache:
            await self._cache.set_async(f"stop:{stop_order.stop_id}", stop_order.__dict__, ttl=3600)
        
        # Publish event
        await self._publish_stop_event("stop.created", stop_order)
        
        # Record metrics
        if self._metrics:
            self._metrics.record_metric("stops.created", 1, tags={
                "symbol": stop_order.symbol,
                "type": stop_config.stop_type.value
            })
        
        self._logger.info("Stop loss created",
                         stop_id=stop_order.stop_id,
                         symbol=stop_order.symbol,
                         stop_price=str(stop_price),
                         type=stop_config.stop_type.value)
        
        return stop_order.stop_id
    
    async def _calculate_stop_price(self, stop_order: StopOrder, 
                                  position: Position) -> Decimal:
        """Calculate stop price based on configuration."""
        config = stop_order.stop_config
        current_price = position.current_price or position.average_entry_price
        
        if config.stop_type == StopType.FIXED:
            if config.trigger_price:
                return config.trigger_price
            elif config.stop_distance:
                if position.side == PositionSide.LONG:
                    return current_price - config.stop_distance
                else:
                    return current_price + config.stop_distance
        
        elif config.stop_type == StopType.PERCENTAGE:
            if config.stop_percentage:
                distance = current_price * config.stop_percentage
                if position.side == PositionSide.LONG:
                    return current_price - distance
                else:
                    return current_price + distance
        
        elif config.stop_type == StopType.TRAILING:
            if config.trailing_distance:
                if position.side == PositionSide.LONG:
                    return current_price - config.trailing_distance
                else:
                    return current_price + config.trailing_distance
            elif config.trailing_percentage:
                distance = current_price * config.trailing_percentage
                if position.side == PositionSide.LONG:
                    return current_price - distance
                else:
                    return current_price + distance
        
        elif config.stop_type == StopType.ATR:
            atr = await self._get_atr(stop_order.symbol)
            if atr and config.atr_multiplier:
                distance = atr * config.atr_multiplier
                if position.side == PositionSide.LONG:
                    return current_price - distance
                else:
                    return current_price + distance
        
        elif config.stop_type == StopType.VOLATILITY:
            volatility = await self._get_volatility(stop_order.symbol)
            if volatility:
                # Use 2 standard deviations by default
                distance = current_price * volatility * Decimal("2")
                if position.side == PositionSide.LONG:
                    return current_price - distance
                else:
                    return current_price + distance
        
        # Default to 2% stop
        distance = current_price * Decimal("0.02")
        if position.side == PositionSide.LONG:
            return current_price - distance
        else:
            return current_price + distance
    
    def _validate_stop_price(self, stop_order: StopOrder, position: Position):
        """Validate stop price is reasonable."""
        current_price = position.current_price or position.average_entry_price
        stop_price = stop_order.current_stop_price
        
        if position.side == PositionSide.LONG:
            # For long positions, stop should be below current price
            if stop_price >= current_price:
                raise ValueError(f"Stop price {stop_price} must be below current price {current_price} for long position")
            
            # Check for reasonable stop distance
            stop_distance_pct = (current_price - stop_price) / current_price
            if stop_distance_pct > Decimal("0.50"):  # 50% stop is too wide
                self._logger.warning("Stop distance is very wide",
                                   stop_distance_pct=str(stop_distance_pct))
        else:
            # For short positions, stop should be above current price
            if stop_price <= current_price:
                raise ValueError(f"Stop price {stop_price} must be above current price {current_price} for short position")
            
            # Check for reasonable stop distance
            stop_distance_pct = (stop_price - current_price) / current_price
            if stop_distance_pct > Decimal("0.50"):  # 50% stop is too wide
                self._logger.warning("Stop distance is very wide",
                                   stop_distance_pct=str(stop_distance_pct))
    
    async def update_market_price(self, symbol: str, price: Decimal):
        """Update market price for stop monitoring."""
        self._price_cache[symbol] = price
        
        # Update trailing stops
        if symbol in self._symbol_stops:
            for stop_id in self._symbol_stops[symbol]:
                stop_order = self._stop_orders.get(stop_id)
                if stop_order and stop_order.status == StopStatus.ACTIVE:
                    if stop_order.stop_config.stop_type == StopType.TRAILING:
                        await self._update_trailing_stop(stop_order, price)
    
    async def _update_trailing_stop(self, stop_order: StopOrder, current_price: Decimal):
        """Update trailing stop based on price movement."""
        config = stop_order.stop_config
        
        # Determine if price moved favorably
        if stop_order.side == OrderSide.SELL:  # Long position
            if current_price > (stop_order.highest_price or current_price):
                stop_order.highest_price = current_price
                
                # Calculate new stop
                if config.trailing_distance:
                    new_stop = current_price - config.trailing_distance
                elif config.trailing_percentage:
                    new_stop = current_price * (1 - config.trailing_percentage)
                else:
                    return
                
                # Only move stop up, never down
                if new_stop > stop_order.current_stop_price:
                    stop_order.current_stop_price = new_stop
                    
                    self._logger.info("Trailing stop updated",
                                     stop_id=stop_order.stop_id,
                                     new_stop=str(new_stop),
                                     high_price=str(current_price))
                    
                    await self._publish_stop_event("stop.updated", stop_order)
        
        else:  # Short position
            if current_price < (stop_order.lowest_price or current_price):
                stop_order.lowest_price = current_price
                
                # Calculate new stop
                if config.trailing_distance:
                    new_stop = current_price + config.trailing_distance
                elif config.trailing_percentage:
                    new_stop = current_price * (1 + config.trailing_percentage)
                else:
                    return
                
                # Only move stop down, never up
                if new_stop < stop_order.current_stop_price:
                    stop_order.current_stop_price = new_stop
                    
                    self._logger.info("Trailing stop updated",
                                     stop_id=stop_order.stop_id,
                                     new_stop=str(new_stop),
                                     low_price=str(current_price))
                    
                    await self._publish_stop_event("stop.updated", stop_order)
    
    async def _monitor_stops(self):
        """Background task to monitor stop conditions."""
        while True:
            try:
                await asyncio.sleep(self._check_interval)
                
                # Check all active stops
                for stop_id, stop_order in list(self._stop_orders.items()):
                    if stop_order.status != StopStatus.ACTIVE:
                        continue
                    
                    # Get current price
                    current_price = self._price_cache.get(stop_order.symbol)
                    if not current_price:
                        continue
                    
                    # Check stop condition
                    should_trigger = await self._check_stop_condition(stop_order, current_price)
                    
                    if should_trigger:
                        await self._trigger_stop(stop_order, current_price)
                    
                    # Check take profit
                    if stop_order.take_profit_config:
                        should_take_profit = await self._check_take_profit_condition(
                            stop_order, current_price
                        )
                        if should_take_profit:
                            await self._trigger_take_profit(stop_order, current_price)
                    
                    # Check time stop
                    if stop_order.stop_config.time_stop_hours:
                        age_hours = (datetime.utcnow() - stop_order.created_at).total_seconds() / 3600
                        if age_hours >= stop_order.stop_config.time_stop_hours:
                            await self._trigger_time_stop(stop_order)
                
            except Exception as e:
                self._logger.error(f"Error in stop monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _check_stop_condition(self, stop_order: StopOrder, 
                                  current_price: Decimal) -> bool:
        """Check if stop condition is met."""
        stop_price = stop_order.current_stop_price
        
        if stop_order.side == OrderSide.SELL:  # Long position stop
            return current_price <= stop_price
        else:  # Short position stop
            return current_price >= stop_price
    
    async def _check_take_profit_condition(self, stop_order: StopOrder,
                                         current_price: Decimal) -> bool:
        """Check if take profit condition is met."""
        config = stop_order.take_profit_config
        if not config:
            return False
        
        # Single target
        if config.target_price:
            if stop_order.side == OrderSide.SELL:  # Long position
                return current_price >= config.target_price
            else:  # Short position
                return current_price <= config.target_price
        
        # Percentage target
        if config.target_percentage and stop_order.initial_price:
            if stop_order.side == OrderSide.SELL:  # Long position
                target = stop_order.initial_price * (1 + config.target_percentage)
                return current_price >= target
            else:  # Short position
                target = stop_order.initial_price * (1 - config.target_percentage)
                return current_price <= target
        
        return False
    
    async def _trigger_stop(self, stop_order: StopOrder, trigger_price: Decimal):
        """Trigger stop loss order."""
        stop_order.status = StopStatus.TRIGGERED
        stop_order.triggered_at = datetime.utcnow()
        
        self._logger.warning("Stop loss triggered",
                           stop_id=stop_order.stop_id,
                           symbol=stop_order.symbol,
                           trigger_price=str(trigger_price),
                           stop_price=str(stop_order.current_stop_price))
        
        # Create market order to exit position
        order_request = OrderRequest(
            symbol=stop_order.symbol,
            side=stop_order.side,
            quantity=stop_order.quantity - stop_order.filled_quantity,
            order_type=OrderType.MARKET,
            metadata={
                "stop_id": stop_order.stop_id,
                "trigger_type": "stop_loss",
                "trigger_price": str(trigger_price)
            }
        )
        
        try:
            # Submit exit order
            order_id = await self.order_manager.submit_order(order_request)
            stop_order.triggered_orders.append(order_id)
            
            # Publish event
            await self._publish_stop_event("stop.triggered", stop_order, {
                "trigger_price": str(trigger_price),
                "order_id": order_id
            })
            
            # Record metrics
            if self._metrics:
                self._metrics.record_metric("stops.triggered", 1, tags={
                    "symbol": stop_order.symbol,
                    "type": "stop_loss"
                })
            
        except Exception as e:
            self._logger.error("Failed to submit stop order",
                             stop_id=stop_order.stop_id,
                             error=str(e))
            stop_order.status = StopStatus.ACTIVE  # Revert status
    
    async def _trigger_take_profit(self, stop_order: StopOrder, trigger_price: Decimal):
        """Trigger take profit order."""
        config = stop_order.take_profit_config
        if not config:
            return
        
        self._logger.info("Take profit triggered",
                         stop_id=stop_order.stop_id,
                         symbol=stop_order.symbol,
                         trigger_price=str(trigger_price))
        
        # Determine quantity for partial exits
        exit_quantity = stop_order.quantity - stop_order.filled_quantity
        
        # Check for partial targets
        if config.targets:
            for target_price, target_pct in config.targets:
                if stop_order.side == OrderSide.SELL and trigger_price >= target_price:
                    exit_quantity = stop_order.quantity * target_pct
                    break
                elif stop_order.side == OrderSide.BUY and trigger_price <= target_price:
                    exit_quantity = stop_order.quantity * target_pct
                    break
        
        # Create order
        order_request = OrderRequest(
            symbol=stop_order.symbol,
            side=stop_order.side,
            quantity=exit_quantity,
            order_type=OrderType.LIMIT,
            price=trigger_price,
            metadata={
                "stop_id": stop_order.stop_id,
                "trigger_type": "take_profit",
                "trigger_price": str(trigger_price)
            }
        )
        
        try:
            order_id = await self.order_manager.submit_order(order_request)
            stop_order.triggered_orders.append(order_id)
            
            # Update filled quantity
            stop_order.filled_quantity += exit_quantity
            
            # Check if fully filled
            if stop_order.filled_quantity >= stop_order.quantity:
                stop_order.status = StopStatus.TRIGGERED
                stop_order.triggered_at = datetime.utcnow()
            else:
                stop_order.status = StopStatus.PARTIALLY_FILLED
            
            await self._publish_stop_event("stop.take_profit", stop_order, {
                "trigger_price": str(trigger_price),
                "order_id": order_id,
                "quantity": str(exit_quantity)
            })
            
            if self._metrics:
                self._metrics.record_metric("stops.triggered", 1, tags={
                    "symbol": stop_order.symbol,
                    "type": "take_profit"
                })
            
        except Exception as e:
            self._logger.error("Failed to submit take profit order",
                             stop_id=stop_order.stop_id,
                             error=str(e))
    
    async def _trigger_time_stop(self, stop_order: StopOrder):
        """Trigger time-based stop."""
        stop_order.status = StopStatus.TRIGGERED
        stop_order.triggered_at = datetime.utcnow()
        
        self._logger.info("Time stop triggered",
                         stop_id=stop_order.stop_id,
                         symbol=stop_order.symbol,
                         age_hours=stop_order.stop_config.time_stop_hours)
        
        # Create market order
        order_request = OrderRequest(
            symbol=stop_order.symbol,
            side=stop_order.side,
            quantity=stop_order.quantity - stop_order.filled_quantity,
            order_type=OrderType.MARKET,
            metadata={
                "stop_id": stop_order.stop_id,
                "trigger_type": "time_stop"
            }
        )
        
        try:
            order_id = await self.order_manager.submit_order(order_request)
            stop_order.triggered_orders.append(order_id)
            
            await self._publish_stop_event("stop.time_triggered", stop_order, {
                "order_id": order_id
            })
            
        except Exception as e:
            self._logger.error("Failed to submit time stop order",
                             stop_id=stop_order.stop_id,
                             error=str(e))
            stop_order.status = StopStatus.ACTIVE
    
    async def cancel_stop(self, stop_id: str):
        """Cancel a stop order."""
        stop_order = self._stop_orders.get(stop_id)
        if not stop_order:
            raise ValueError(f"Stop order {stop_id} not found")
        
        if stop_order.status != StopStatus.ACTIVE:
            raise ValueError(f"Stop order {stop_id} is not active")
        
        stop_order.status = StopStatus.CANCELLED
        
        # Remove from indices
        position_id = stop_order.position_id
        if position_id in self._position_stops:
            self._position_stops[position_id].remove(stop_id)
        
        if stop_order.symbol in self._symbol_stops:
            self._symbol_stops[stop_order.symbol].remove(stop_id)
        
        await self._publish_stop_event("stop.cancelled", stop_order)
        
        self._logger.info("Stop order cancelled", stop_id=stop_id)
    
    async def get_position_stops(self, position_id: str) -> List[StopOrder]:
        """Get all stops for a position."""
        stop_ids = self._position_stops.get(position_id, [])
        return [self._stop_orders[sid] for sid in stop_ids if sid in self._stop_orders]
    
    async def get_symbol_stops(self, symbol: str) -> List[StopOrder]:
        """Get all stops for a symbol."""
        stop_ids = self._symbol_stops.get(symbol, [])
        return [self._stop_orders[sid] for sid in stop_ids if sid in self._stop_orders]
    
    async def _get_atr(self, symbol: str) -> Optional[Decimal]:
        """Get Average True Range for symbol."""
        # Check cache first
        if symbol in self._atr_cache:
            return self._atr_cache[symbol]
        
        # In production, would calculate from price data
        # For now, return placeholder
        return Decimal("100")  # Placeholder
    
    async def _get_volatility(self, symbol: str) -> Optional[Decimal]:
        """Get volatility for symbol."""
        # Check cache first
        if symbol in self._volatility_cache:
            return self._volatility_cache[symbol]
        
        # In production, would calculate from price data
        # For now, return placeholder
        return Decimal("0.02")  # 2% daily volatility
    
    async def _publish_stop_event(self, event_type: str, stop_order: StopOrder,
                                additional_data: Optional[Dict[str, Any]] = None):
        """Publish stop-related event."""
        event_data = {
            "stop_id": stop_order.stop_id,
            "symbol": stop_order.symbol,
            "status": stop_order.status.value,
            "stop_type": stop_order.stop_config.stop_type.value,
            "current_stop_price": str(stop_order.current_stop_price) if stop_order.current_stop_price else None
        }
        
        if additional_data:
            event_data.update(additional_data)
        
        event = Event(
            type=event_type,
            data=event_data,
            source="stop_loss_manager",
            priority=EventPriority.HIGH if "triggered" in event_type else EventPriority.NORMAL
        )
        
        await self._event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get stop loss manager statistics."""
        active_stops = sum(1 for s in self._stop_orders.values() if s.status == StopStatus.ACTIVE)
        triggered_stops = sum(1 for s in self._stop_orders.values() if s.status == StopStatus.TRIGGERED)
        
        return {
            "total_stops": len(self._stop_orders),
            "active_stops": active_stops,
            "triggered_stops": triggered_stops,
            "stops_by_type": self._get_stops_by_type(),
            "stops_by_symbol": {s: len(stops) for s, stops in self._symbol_stops.items()}
        }
    
    def _get_stops_by_type(self) -> Dict[str, int]:
        """Get stop count by type."""
        from collections import defaultdict
        counts = defaultdict(int)
        for stop in self._stop_orders.values():
            counts[stop.stop_config.stop_type.value] += 1
        return dict(counts)
    
    async def shutdown(self):
        """Shutdown the stop loss manager."""
        self._monitor_task.cancel()
        await self._monitor_task
        
        self._logger.info("Stop loss manager shutdown complete")