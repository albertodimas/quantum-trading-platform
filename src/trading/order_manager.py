"""
Order Management System

Handles order lifecycle, validation, and execution tracking.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set
from uuid import uuid4

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderStatus, OrderSide, OrderType
from .models import Order, ExecutionReport

logger = get_logger(__name__)


@injectable
class OrderManager:
    """
    Order Manager handles all aspects of order lifecycle.
    
    Responsibilities:
    - Order creation and validation
    - Order state management
    - Execution tracking
    - Risk compliance
    """
    
    def __init__(self):
        """Initialize order manager."""
        # In-memory order storage (would be database in production)
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        
        # Order state tracking
        self._pending_orders: Set[str] = set()
        self._open_orders: Set[str] = set()
        self._completed_orders: Set[str] = set()
        
        # Execution tracking
        self._executions: Dict[str, List[ExecutionReport]] = {}
        
        logger.info("Order manager initialized")
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        metadata: Optional[Dict] = None
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: Type of order
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional order metadata
            
        Returns:
            Created order
        """
        # Generate order ID
        order_id = str(uuid4())
        
        # Validate order parameters
        await self._validate_order_params(
            symbol, side, order_type, quantity, price
        )
        
        # Create order object
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.PENDING,
            quantity=quantity,
            price=price,
            executed_qty=Decimal("0"),
            remaining_qty=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Store order
        self._orders[order_id] = order
        self._pending_orders.add(order_id)
        
        logger.info(
            "Order created",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            quantity=float(quantity),
            price=float(price) if price else None
        )
        
        return order
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        open_orders = []
        
        for order_id in self._open_orders:
            order = self._orders.get(order_id)
            if order and (not symbol or order.symbol == symbol):
                open_orders.append(order)
        
        return open_orders
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history, optionally filtered by symbol."""
        history = self._order_history[-limit:] if limit else self._order_history
        
        if symbol:
            history = [order for order in history if order.symbol == symbol]
        
        return history
    
    async def update_status(self, order_id: str, status: OrderStatus) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Order ID
            status: New status
            
        Returns:
            True if update was successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error("Order not found", order_id=order_id)
            return False
        
        old_status = order.status
        order.status = status
        order.updated_at = datetime.now(timezone.utc)
        
        # Update tracking sets
        self._update_order_tracking(order_id, old_status, status)
        
        logger.info(
            "Order status updated",
            order_id=order_id,
            old_status=old_status.value,
            new_status=status.value
        )
        
        return True
    
    async def update_from_exchange_order(self, order_id: str, exchange_order) -> bool:
        """
        Update order from exchange order data.
        
        Args:
            order_id: Internal order ID
            exchange_order: Order object from exchange
            
        Returns:
            True if update was successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error("Order not found", order_id=order_id)
            return False
        
        # Update order with exchange data
        if hasattr(exchange_order, 'id'):
            order.exchange_order_id = exchange_order.id
        
        if hasattr(exchange_order, 'status'):
            await self.update_status(order_id, exchange_order.status)
        
        if hasattr(exchange_order, 'executed_qty'):
            order.executed_qty = exchange_order.executed_qty
            order.remaining_qty = order.quantity - order.executed_qty
        
        if hasattr(exchange_order, 'price') and exchange_order.price:
            order.price = exchange_order.price
        
        order.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            "Order updated from exchange",
            order_id=order_id,
            exchange_order_id=getattr(exchange_order, 'id', None)
        )
        
        return True
    
    async def update_order_from_execution(self, execution: ExecutionReport) -> bool:
        """
        Update order from execution report.
        
        Args:
            execution: Execution report
            
        Returns:
            True if update was successful
        """
        order = self._orders.get(execution.order_id)
        if not order:
            logger.error("Order not found", order_id=execution.order_id)
            return False
        
        # Update order quantities
        order.executed_qty = execution.filled_quantity
        order.remaining_qty = order.quantity - execution.filled_quantity
        
        # Update status
        if execution.status:
            await self.update_status(execution.order_id, execution.status)
        
        # Store execution
        if execution.order_id not in self._executions:
            self._executions[execution.order_id] = []
        self._executions[execution.order_id].append(execution)
        
        logger.info(
            "Order updated from execution",
            order_id=execution.order_id,
            filled_quantity=float(execution.filled_quantity)
        )
        
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error("Order not found", order_id=order_id)
            return False
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.OPEN]:
            logger.warning(
                "Cannot cancel order in current status",
                order_id=order_id,
                status=order.status.value
            )
            return False
        
        await self.update_status(order_id, OrderStatus.CANCELLED)
        
        logger.info("Order cancelled", order_id=order_id)
        return True
    
    async def get_executions(self, order_id: str) -> List[ExecutionReport]:
        """Get execution reports for an order."""
        return self._executions.get(order_id, [])
    
    async def get_order_statistics(self) -> Dict:
        """Get order statistics."""
        total_orders = len(self._orders)
        pending_count = len(self._pending_orders)
        open_count = len(self._open_orders)
        completed_count = len(self._completed_orders)
        
        # Calculate volumes
        total_volume = Decimal("0")
        executed_volume = Decimal("0")
        
        for order in self._orders.values():
            if order.price and order.quantity:
                total_volume += order.price * order.quantity
            
            if order.price and order.executed_qty:
                executed_volume += order.price * order.executed_qty
        
        return {
            "total_orders": total_orders,
            "pending_orders": pending_count,
            "open_orders": open_count,
            "completed_orders": completed_count,
            "total_volume": float(total_volume),
            "executed_volume": float(executed_volume),
            "execution_rate": float(executed_volume / total_volume) if total_volume > 0 else 0
        }
    
    # Private methods
    
    async def _validate_order_params(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal]
    ) -> None:
        """Validate order parameters."""
        if not symbol:
            raise ValueError("Symbol is required")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if not price or price <= 0:
                raise ValueError(f"Price is required for {order_type.value} orders")
        
        # Additional validations can be added here
        # - Symbol existence check
        # - Minimum quantity check
        # - Price precision check
        # - Account balance check
    
    def _update_order_tracking(
        self,
        order_id: str,
        old_status: OrderStatus,
        new_status: OrderStatus
    ) -> None:
        """Update order tracking sets based on status change."""
        # Remove from old status set
        if old_status == OrderStatus.PENDING and order_id in self._pending_orders:
            self._pending_orders.remove(order_id)
        elif old_status in [OrderStatus.NEW, OrderStatus.OPEN] and order_id in self._open_orders:
            self._open_orders.remove(order_id)
        elif old_status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ] and order_id in self._completed_orders:
            self._completed_orders.remove(order_id)
        
        # Add to new status set
        if new_status == OrderStatus.PENDING:
            self._pending_orders.add(order_id)
        elif new_status in [OrderStatus.NEW, OrderStatus.OPEN]:
            self._open_orders.add(order_id)
        elif new_status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]:
            self._completed_orders.add(order_id)
            
            # Add to history
            order = self._orders.get(order_id)
            if order and order not in self._order_history:
                self._order_history.append(order)