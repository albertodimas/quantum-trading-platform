"""
Order Management System (OMS) for the Quantum Trading Platform.

Features:
- Comprehensive order lifecycle management
- Order routing and smart order routing (SOR)
- Order validation and risk checks
- Position and exposure tracking
- Execution reporting and analytics
- Order persistence and recovery
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from decimal import Decimal

from ..core.observability.logger import get_logger
from ..core.observability.metrics import get_metrics_collector
from ..core.observability.tracing import trace_async
from ..core.messaging.event_bus import get_event_bus, Event, EventPriority
from ..core.architecture.circuit_breaker import CircuitBreaker
from ..core.architecture.rate_limiter import RateLimiter, TokenBucketStrategy
from ..core.architecture.factory_registry import get_factory_registry
from ..exchange.exchange_interface import (
    Order, OrderType, OrderSide, OrderStatus, ExchangeInterface
)
from ..exchange.exchange_manager import ExchangeManager


logger = get_logger(__name__)


class OrderValidationError(Exception):
    """Order validation error."""
    pass


class InsufficientBalanceError(Exception):
    """Insufficient balance error."""
    pass


class RiskLimitExceededError(Exception):
    """Risk limit exceeded error."""
    pass


@dataclass
class OrderRequest:
    """Order request from client/strategy."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderExecution:
    """Order execution details."""
    order_id: str
    executed_quantity: Decimal
    executed_price: Decimal
    commission: Decimal
    timestamp: datetime
    exchange: str
    trade_id: Optional[str] = None


@dataclass
class OrderTracking:
    """Internal order tracking information."""
    internal_id: str
    request: OrderRequest
    exchange_orders: Dict[str, Order] = field(default_factory=dict)  # exchange -> Order
    executions: List[OrderExecution] = field(default_factory=list)
    status: OrderStatus = OrderStatus.NEW
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def total_executed_quantity(self) -> Decimal:
        """Get total executed quantity."""
        return sum(execution.executed_quantity for execution in self.executions)
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to execute."""
        return self.request.quantity - self.total_executed_quantity
    
    @property
    def average_price(self) -> Optional[Decimal]:
        """Calculate average execution price."""
        if not self.executions:
            return None
        
        total_value = sum(
            execution.executed_quantity * execution.executed_price 
            for execution in self.executions
        )
        total_quantity = self.total_executed_quantity
        
        return total_value / total_quantity if total_quantity > 0 else Decimal("0")
    
    @property
    def total_commission(self) -> Decimal:
        """Get total commission paid."""
        return sum(execution.commission for execution in self.executions)


class OrderManager:
    """
    Order Management System for handling order lifecycle.
    
    Features:
    - Order validation and risk checks
    - Smart order routing
    - Position tracking
    - Execution reporting
    - Order persistence and recovery
    """
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self._logger = get_logger(self.__class__.__name__)
        self._metrics = get_metrics_collector().get_collector("trading")
        self._event_bus = get_event_bus()
        
        # Order tracking
        self._active_orders: Dict[str, OrderTracking] = {}
        self._order_history: List[OrderTracking] = []
        self._max_history_size = 10000
        
        # Position tracking
        self._positions: Dict[str, Dict[str, Decimal]] = {}  # symbol -> {exchange: quantity}
        self._balances: Dict[str, Dict[str, Decimal]] = {}   # exchange -> {currency: amount}
        
        # Risk limits
        self._risk_limits = {
            "max_order_size": Decimal("100000"),  # USD value
            "max_position_size": Decimal("500000"),  # USD value
            "max_daily_loss": Decimal("50000"),  # USD
            "max_open_orders": 100,
            "min_order_size": Decimal("10")  # USD value
        }
        
        # Circuit breakers for each exchange
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Rate limiters for order submission
        self._order_rate_limiter = RateLimiter(
            name="order_submission",
            strategy=TokenBucketStrategy(rate=10, capacity=20),  # 10 orders/sec, burst 20
            metrics_enabled=True
        )
        
        # Statistics
        self._stats = {
            "orders_submitted": 0,
            "orders_executed": 0,
            "orders_rejected": 0,
            "orders_failed": 0,
            "total_volume": Decimal("0"),
            "total_commission": Decimal("0"),
            "daily_pnl": Decimal("0")
        }
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._monitor_orders()),
            asyncio.create_task(self._update_positions()),
            asyncio.create_task(self._check_risk_limits())
        ]
    
    @trace_async(name="submit_order", tags={"component": "oms"})
    async def submit_order(self, request: OrderRequest) -> str:
        """
        Submit an order for execution.
        
        Args:
            request: Order request details
            
        Returns:
            Internal order ID
            
        Raises:
            OrderValidationError: If order validation fails
            RiskLimitExceededError: If risk limits exceeded
        """
        # Generate internal order ID
        internal_id = request.client_order_id or str(uuid.uuid4())
        
        try:
            # Validate order
            await self._validate_order(request)
            
            # Check rate limits
            if not await self._order_rate_limiter.check_async("global"):
                raise OrderValidationError("Order rate limit exceeded")
            
            # Check risk limits
            await self._check_order_risk(request)
            
            # Create order tracking
            tracking = OrderTracking(
                internal_id=internal_id,
                request=request,
                status=OrderStatus.NEW
            )
            
            self._active_orders[internal_id] = tracking
            self._stats["orders_submitted"] += 1
            
            # Determine best exchange for execution
            exchange = await self._select_exchange(request)
            
            # Create exchange order
            order = await self._create_exchange_order(request, exchange)
            
            # Submit to exchange
            submitted_order = await self._submit_to_exchange(order, exchange, tracking)
            
            if submitted_order:
                tracking.exchange_orders[exchange] = submitted_order
                tracking.status = OrderStatus.PARTIALLY_FILLED if submitted_order.status == OrderStatus.PARTIALLY_FILLED else OrderStatus.NEW
                
                # Publish order event
                await self._publish_order_event("order.placed", tracking)
                
                self._logger.info("Order submitted successfully",
                                order_id=internal_id,
                                symbol=request.symbol,
                                side=request.side.value,
                                quantity=str(request.quantity),
                                exchange=exchange)
            else:
                tracking.status = OrderStatus.REJECTED
                tracking.error_message = "Failed to submit to exchange"
                self._stats["orders_rejected"] += 1
            
            return internal_id
            
        except Exception as e:
            self._logger.error("Order submission failed",
                             order_id=internal_id,
                             error=str(e))
            
            if internal_id in self._active_orders:
                self._active_orders[internal_id].status = OrderStatus.REJECTED
                self._active_orders[internal_id].error_message = str(e)
            
            self._stats["orders_failed"] += 1
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Internal order ID
            
        Returns:
            True if cancellation successful
        """
        if order_id not in self._active_orders:
            self._logger.warning("Order not found for cancellation", order_id=order_id)
            return False
        
        tracking = self._active_orders[order_id]
        
        # Cancel on all exchanges
        cancel_tasks = []
        for exchange, order in tracking.exchange_orders.items():
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                task = self._cancel_on_exchange(order, exchange)
                cancel_tasks.append(task)
        
        results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        # Update tracking status
        if any(result is True for result in results if not isinstance(result, Exception)):
            tracking.status = OrderStatus.CANCELLED
            await self._publish_order_event("order.cancelled", tracking)
            return True
        
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderTracking]:
        """Get order status and details."""
        # Check active orders first
        if order_id in self._active_orders:
            return self._active_orders[order_id]
        
        # Check history
        for order in self._order_history:
            if order.internal_id == order_id:
                return order
        
        return None
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[OrderTracking]:
        """Get all active orders, optionally filtered by symbol."""
        orders = list(self._active_orders.values())
        
        if symbol:
            orders = [o for o in orders if o.request.symbol == symbol]
        
        return orders
    
    async def get_positions(self) -> Dict[str, Dict[str, Decimal]]:
        """Get current positions across all exchanges."""
        return self._positions.copy()
    
    async def get_position(self, symbol: str) -> Decimal:
        """Get total position for a symbol across all exchanges."""
        if symbol not in self._positions:
            return Decimal("0")
        
        return sum(self._positions[symbol].values())
    
    async def _validate_order(self, request: OrderRequest):
        """Validate order request."""
        # Check required fields
        if not request.symbol:
            raise OrderValidationError("Symbol is required")
        
        if request.quantity <= 0:
            raise OrderValidationError("Quantity must be positive")
        
        # Check order type specific validations
        if request.order_type == OrderType.LIMIT and request.price is None:
            raise OrderValidationError("Price is required for limit orders")
        
        if request.price is not None and request.price <= 0:
            raise OrderValidationError("Price must be positive")
        
        # Check max open orders
        if len(self._active_orders) >= self._risk_limits["max_open_orders"]:
            raise OrderValidationError("Maximum open orders limit reached")
    
    async def _check_order_risk(self, request: OrderRequest):
        """Check order against risk limits."""
        # Estimate order value
        order_value = await self._estimate_order_value(request)
        
        # Check min/max order size
        if order_value < self._risk_limits["min_order_size"]:
            raise RiskLimitExceededError(f"Order value {order_value} below minimum {self._risk_limits['min_order_size']}")
        
        if order_value > self._risk_limits["max_order_size"]:
            raise RiskLimitExceededError(f"Order value {order_value} exceeds maximum {self._risk_limits['max_order_size']}")
        
        # Check position limits
        current_position = await self.get_position(request.symbol)
        new_position = current_position + (request.quantity if request.side == OrderSide.BUY else -request.quantity)
        
        position_value = abs(new_position) * (request.price or await self._get_market_price(request.symbol))
        
        if position_value > self._risk_limits["max_position_size"]:
            raise RiskLimitExceededError(f"Position value {position_value} would exceed maximum {self._risk_limits['max_position_size']}")
        
        # Check daily loss limit
        if self._stats["daily_pnl"] < -self._risk_limits["max_daily_loss"]:
            raise RiskLimitExceededError(f"Daily loss limit of {self._risk_limits['max_daily_loss']} exceeded")
    
    async def _estimate_order_value(self, request: OrderRequest) -> Decimal:
        """Estimate order value in USD."""
        if request.price:
            return request.quantity * request.price
        
        # Get market price for market orders
        market_price = await self._get_market_price(request.symbol)
        return request.quantity * market_price
    
    async def _get_market_price(self, symbol: str) -> Decimal:
        """Get current market price for a symbol."""
        # Try to get price from any available exchange
        for exchange_name in self.exchange_manager.get_connected_exchanges():
            try:
                exchange = self.exchange_manager.get_exchange(exchange_name)
                if exchange:
                    ticker = await exchange.get_ticker(symbol)
                    if ticker and ticker.last:
                        return Decimal(str(ticker.last))
            except Exception:
                continue
        
        raise ValueError(f"Unable to get market price for {symbol}")
    
    async def _select_exchange(self, request: OrderRequest) -> str:
        """Select best exchange for order execution."""
        # For now, use simple routing based on best price
        # In production, consider liquidity, fees, latency, etc.
        
        best_exchange = None
        best_price = None
        
        for exchange_name in self.exchange_manager.get_connected_exchanges():
            try:
                exchange = self.exchange_manager.get_exchange(exchange_name)
                if not exchange:
                    continue
                
                # Check if exchange supports the symbol
                ticker = await exchange.get_ticker(request.symbol)
                if not ticker:
                    continue
                
                # Get relevant price
                if request.side == OrderSide.BUY:
                    price = ticker.ask or ticker.last
                else:
                    price = ticker.bid or ticker.last
                
                if price and (best_price is None or 
                           (request.side == OrderSide.BUY and price < best_price) or
                           (request.side == OrderSide.SELL and price > best_price)):
                    best_price = price
                    best_exchange = exchange_name
                    
            except Exception as e:
                self._logger.warning(f"Error checking exchange {exchange_name}: {e}")
                continue
        
        if not best_exchange:
            # Fallback to first available exchange
            exchanges = self.exchange_manager.get_connected_exchanges()
            if exchanges:
                best_exchange = exchanges[0]
            else:
                raise ValueError("No exchanges available for order execution")
        
        return best_exchange
    
    async def _create_exchange_order(self, request: OrderRequest, exchange: str) -> Order:
        """Create exchange-specific order object."""
        return Order(
            id=str(uuid.uuid4()),
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            type=request.order_type,
            quantity=float(request.quantity),
            price=float(request.price) if request.price else None,
            status=OrderStatus.NEW,
            timestamp=datetime.utcnow()
        )
    
    async def _submit_to_exchange(self, order: Order, exchange_name: str, 
                                 tracking: OrderTracking) -> Optional[Order]:
        """Submit order to specific exchange."""
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not available")
        
        # Get or create circuit breaker for exchange
        if exchange_name not in self._circuit_breakers:
            self._circuit_breakers[exchange_name] = CircuitBreaker(
                name=f"oms_{exchange_name}",
                failure_threshold=5,
                recovery_timeout=30,
                expected_exception=Exception
            )
        
        circuit_breaker = self._circuit_breakers[exchange_name]
        
        try:
            # Submit through circuit breaker
            async def submit():
                return await exchange.place_order(order)
            
            submitted_order = await circuit_breaker.call_async(submit)
            
            if self._metrics:
                self._metrics.record_metric("oms.orders.submitted", 1, tags={
                    "exchange": exchange_name,
                    "symbol": order.symbol,
                    "side": order.side.value
                })
            
            return submitted_order
            
        except Exception as e:
            self._logger.error(f"Failed to submit order to {exchange_name}: {e}")
            
            if self._metrics:
                self._metrics.record_metric("oms.orders.failed", 1, tags={
                    "exchange": exchange_name,
                    "reason": "submission_error"
                })
            
            return None
    
    async def _cancel_on_exchange(self, order: Order, exchange_name: str) -> bool:
        """Cancel order on specific exchange."""
        try:
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if not exchange:
                return False
            
            return await exchange.cancel_order(order.id, order.symbol)
            
        except Exception as e:
            self._logger.error(f"Failed to cancel order on {exchange_name}: {e}")
            return False
    
    async def _monitor_orders(self):
        """Monitor active orders for fills and status updates."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                for order_id, tracking in list(self._active_orders.items()):
                    try:
                        await self._update_order_status(tracking)
                    except Exception as e:
                        self._logger.error(f"Error updating order {order_id}: {e}")
                
            except Exception as e:
                self._logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _update_order_status(self, tracking: OrderTracking):
        """Update order status from exchanges."""
        for exchange_name, order in tracking.exchange_orders.items():
            try:
                exchange = self.exchange_manager.get_exchange(exchange_name)
                if not exchange:
                    continue
                
                # Get updated order status
                updated_order = await exchange.get_order(order.id, order.symbol)
                if not updated_order:
                    continue
                
                # Check for fills
                if updated_order.filled > order.filled:
                    # New fill detected
                    fill_quantity = Decimal(str(updated_order.filled - order.filled))
                    fill_price = Decimal(str(updated_order.average_price or updated_order.price or 0))
                    
                    execution = OrderExecution(
                        order_id=tracking.internal_id,
                        executed_quantity=fill_quantity,
                        executed_price=fill_price,
                        commission=Decimal("0"),  # TODO: Get from exchange
                        timestamp=datetime.utcnow(),
                        exchange=exchange_name
                    )
                    
                    tracking.executions.append(execution)
                    tracking.updated_at = datetime.utcnow()
                    
                    # Update stats
                    self._stats["total_volume"] += fill_quantity * fill_price
                    
                    # Publish fill event
                    await self._publish_order_event("order.filled", tracking, execution)
                
                # Update order object
                tracking.exchange_orders[exchange_name] = updated_order
                
                # Update tracking status
                if updated_order.status == OrderStatus.FILLED:
                    tracking.status = OrderStatus.FILLED
                elif updated_order.status == OrderStatus.CANCELLED:
                    tracking.status = OrderStatus.CANCELLED
                elif updated_order.status == OrderStatus.REJECTED:
                    tracking.status = OrderStatus.REJECTED
                elif updated_order.filled > 0:
                    tracking.status = OrderStatus.PARTIALLY_FILLED
                    
            except Exception as e:
                self._logger.error(f"Error updating order from {exchange_name}: {e}")
        
        # Move completed orders to history
        if tracking.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._move_to_history(tracking)
    
    def _move_to_history(self, tracking: OrderTracking):
        """Move completed order to history."""
        if tracking.internal_id in self._active_orders:
            del self._active_orders[tracking.internal_id]
            
            self._order_history.append(tracking)
            
            # Trim history if too large
            if len(self._order_history) > self._max_history_size:
                self._order_history = self._order_history[-self._max_history_size:]
            
            # Update stats
            if tracking.status == OrderStatus.FILLED:
                self._stats["orders_executed"] += 1
                self._stats["total_commission"] += tracking.total_commission
    
    async def _update_positions(self):
        """Update position tracking."""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                # Recalculate positions from order history
                positions: Dict[str, Dict[str, Decimal]] = {}
                
                # Add fills from active orders
                for tracking in self._active_orders.values():
                    self._update_position_from_executions(positions, tracking)
                
                # Add fills from recent history
                for tracking in self._order_history[-1000:]:  # Last 1000 orders
                    self._update_position_from_executions(positions, tracking)
                
                self._positions = positions
                
            except Exception as e:
                self._logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(10)
    
    def _update_position_from_executions(self, positions: Dict[str, Dict[str, Decimal]], 
                                       tracking: OrderTracking):
        """Update positions based on order executions."""
        symbol = tracking.request.symbol
        
        if symbol not in positions:
            positions[symbol] = {}
        
        for execution in tracking.executions:
            exchange = execution.exchange
            
            if exchange not in positions[symbol]:
                positions[symbol][exchange] = Decimal("0")
            
            # Update position
            if tracking.request.side == OrderSide.BUY:
                positions[symbol][exchange] += execution.executed_quantity
            else:
                positions[symbol][exchange] -= execution.executed_quantity
    
    async def _check_risk_limits(self):
        """Periodic risk limit checks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check position limits
                for symbol, exchanges in self._positions.items():
                    total_position = sum(exchanges.values())
                    
                    if total_position != 0:
                        # Get market price
                        try:
                            market_price = await self._get_market_price(symbol)
                            position_value = abs(total_position) * market_price
                            
                            if position_value > self._risk_limits["max_position_size"]:
                                await self._publish_risk_event(
                                    "risk.limit_exceeded",
                                    {
                                        "limit_type": "position_size",
                                        "symbol": symbol,
                                        "current_value": str(position_value),
                                        "limit_value": str(self._risk_limits["max_position_size"])
                                    }
                                )
                        except Exception as e:
                            self._logger.error(f"Error checking position risk for {symbol}: {e}")
                
            except Exception as e:
                self._logger.error(f"Error in risk limit checks: {e}")
                await asyncio.sleep(60)
    
    async def _publish_order_event(self, event_type: str, tracking: OrderTracking, 
                                  execution: Optional[OrderExecution] = None):
        """Publish order-related event."""
        event_data = {
            "order_id": tracking.internal_id,
            "symbol": tracking.request.symbol,
            "side": tracking.request.side.value,
            "quantity": str(tracking.request.quantity),
            "status": tracking.status.value,
            "strategy_id": tracking.request.strategy_id
        }
        
        if execution:
            event_data.update({
                "executed_quantity": str(execution.executed_quantity),
                "executed_price": str(execution.executed_price),
                "commission": str(execution.commission),
                "exchange": execution.exchange
            })
        
        event = Event(
            type=event_type,
            data=event_data,
            source="order_manager",
            priority=EventPriority.HIGH
        )
        
        await self._event_bus.publish(event)
    
    async def _publish_risk_event(self, event_type: str, data: Dict[str, Any]):
        """Publish risk-related event."""
        event = Event(
            type=event_type,
            data=data,
            source="order_manager",
            priority=EventPriority.CRITICAL
        )
        
        await self._event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OMS statistics."""
        return {
            **self._stats,
            "active_orders": len(self._active_orders),
            "total_positions": sum(
                len(exchanges) for exchanges in self._positions.values()
            ),
            "circuit_breaker_states": {
                name: cb.state.value 
                for name, cb in self._circuit_breakers.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown the order manager."""
        # Cancel all active orders
        cancel_tasks = []
        for order_id in list(self._active_orders.keys()):
            task = self.cancel_order(order_id)
            cancel_tasks.append(task)
        
        await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._logger.info("Order manager shutdown complete")