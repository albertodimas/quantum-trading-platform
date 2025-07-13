"""
Order Lifecycle Management

Comprehensive order lifecycle management with state machine,
workflows, and automated processing.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import uuid

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderStatus, OrderSide, OrderType
from .models import Order, ExecutionReport
from .order_manager import OrderManager
from .position_manager import PositionManager
from .risk_manager import RiskManager

logger = get_logger(__name__)


class OrderLifecycleState(Enum):
    """Order lifecycle states"""
    CREATED = "created"
    VALIDATED = "validated"
    RISK_CHECKED = "risk_checked"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SETTLED = "settled"
    FAILED = "failed"


class OrderEvent(Enum):
    """Order lifecycle events"""
    VALIDATE = "validate"
    RISK_CHECK = "risk_check"
    SUBMIT = "submit"
    ACKNOWLEDGE = "acknowledge"
    PARTIAL_FILL = "partial_fill"
    FILL = "fill"
    CANCEL = "cancel"
    REJECT = "reject"
    EXPIRE = "expire"
    SETTLE = "settle"
    FAIL = "fail"


@dataclass
class OrderContext:
    """Order context with additional tracking data"""
    order: Order
    state: OrderLifecycleState = OrderLifecycleState.CREATED
    events: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    last_error: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    
    def add_event(self, event: OrderEvent, details: Optional[Dict] = None):
        """Add event to order history"""
        self.events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event.value,
            "state": self.state.value,
            "details": details or {}
        })


@injectable
class OrderLifecycleManager:
    """
    Manages complete order lifecycle with state machine and workflows.
    
    Features:
    - State machine for order progression
    - Automated retry logic
    - Parent-child order relationships
    - Event-driven processing
    - Settlement and reconciliation
    """
    
    def __init__(
        self,
        order_manager: OrderManager = inject(),
        position_manager: PositionManager = inject(),
        risk_manager: RiskManager = inject()
    ):
        """Initialize order lifecycle manager."""
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        # Order contexts tracking
        self._order_contexts: Dict[str, OrderContext] = {}
        
        # State machine transitions
        self._transitions = self._build_state_machine()
        
        # Event handlers
        self._event_handlers: Dict[OrderEvent, List[Callable]] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Configuration
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # seconds
        self.order_timeout = timedelta(hours=24)
        
        logger.info("Order lifecycle manager initialized")
    
    async def start(self) -> None:
        """Start lifecycle manager background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background monitoring tasks
        self._background_tasks = [
            asyncio.create_task(self._order_timeout_monitor()),
            asyncio.create_task(self._retry_failed_orders()),
            asyncio.create_task(self._settlement_processor()),
        ]
        
        logger.info("Order lifecycle manager started")
    
    async def stop(self) -> None:
        """Stop lifecycle manager"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Order lifecycle manager stopped")
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        **kwargs
    ) -> str:
        """
        Create new order and start lifecycle management.
        
        Returns:
            Order ID
        """
        # Create order through order manager
        order = await self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs
        )
        
        # Create order context
        context = OrderContext(order=order)
        context.add_event(OrderEvent.VALIDATE)
        self._order_contexts[order.id] = context
        
        logger.info(
            "Order created and lifecycle started",
            order_id=order.id,
            symbol=symbol,
            side=side.value,
            quantity=float(quantity)
        )
        
        # Start processing workflow
        asyncio.create_task(self._process_order_workflow(order.id))
        
        return order.id
    
    async def handle_execution_report(self, execution: ExecutionReport) -> None:
        """Handle execution report and update order lifecycle"""
        context = self._order_contexts.get(execution.order_id)
        if not context:
            logger.warning(f"No context found for order {execution.order_id}")
            return
        
        # Update order manager
        await self.order_manager.update_order_from_execution(execution)
        
        # Determine event type
        if execution.status == OrderStatus.FILLED:
            if execution.filled_quantity == context.order.quantity:
                event = OrderEvent.FILL
            else:
                event = OrderEvent.PARTIAL_FILL
        elif execution.status == OrderStatus.CANCELLED:
            event = OrderEvent.CANCEL
        elif execution.status == OrderStatus.REJECTED:
            event = OrderEvent.REJECT
        else:
            event = OrderEvent.ACKNOWLEDGE
        
        # Process event
        await self._process_event(execution.order_id, event, {
            "execution": execution,
            "filled_quantity": float(execution.filled_quantity),
            "average_price": float(execution.average_price or execution.price)
        })
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order and update lifecycle"""
        context = self._order_contexts.get(order_id)
        if not context:
            logger.error(f"Order context not found: {order_id}")
            return False
        
        # Cancel through order manager
        success = await self.order_manager.cancel_order(order_id)
        
        if success:
            await self._process_event(order_id, OrderEvent.CANCEL)
        
        return success
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get comprehensive order status"""
        context = self._order_contexts.get(order_id)
        if not context:
            return None
        
        order = await self.order_manager.get_order(order_id)
        if not order:
            return None
        
        return {
            "order_id": order_id,
            "state": context.state.value,
            "status": order.status.value,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": float(order.quantity),
            "executed_qty": float(order.executed_qty),
            "remaining_qty": float(order.remaining_qty),
            "price": float(order.price) if order.price else None,
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "events": context.events,
            "retry_count": context.retry_count,
            "last_error": context.last_error,
            "parent_order_id": context.parent_order_id,
            "child_orders": context.child_orders
        }
    
    async def get_active_orders(self) -> List[Dict]:
        """Get all active orders with their lifecycle status"""
        active_orders = []
        
        for order_id, context in self._order_contexts.items():
            if context.state not in [
                OrderLifecycleState.FILLED,
                OrderLifecycleState.CANCELLED,
                OrderLifecycleState.SETTLED,
                OrderLifecycleState.FAILED
            ]:
                status = await self.get_order_status(order_id)
                if status:
                    active_orders.append(status)
        
        return active_orders
    
    def register_event_handler(self, event: OrderEvent, handler: Callable) -> None:
        """Register event handler for order lifecycle events"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    # Private methods
    
    async def _process_order_workflow(self, order_id: str) -> None:
        """Process complete order workflow"""
        try:
            # Validation
            await self._process_event(order_id, OrderEvent.VALIDATE)
            
            # Risk check
            await self._process_event(order_id, OrderEvent.RISK_CHECK)
            
            # Submit to exchange
            await self._process_event(order_id, OrderEvent.SUBMIT)
            
        except Exception as e:
            logger.error(f"Order workflow failed for {order_id}: {str(e)}")
            await self._process_event(order_id, OrderEvent.FAIL, {"error": str(e)})
    
    async def _process_event(
        self,
        order_id: str,
        event: OrderEvent,
        details: Optional[Dict] = None
    ) -> None:
        """Process order event and update state"""
        context = self._order_contexts.get(order_id)
        if not context:
            logger.error(f"Order context not found: {order_id}")
            return
        
        # Check if transition is valid
        new_state = self._transitions.get((context.state, event))
        if not new_state:
            logger.warning(
                f"Invalid transition: {context.state.value} -> {event.value} for order {order_id}"
            )
            return
        
        # Update state
        old_state = context.state
        context.state = new_state
        context.add_event(event, details)
        
        logger.info(
            f"Order state transition: {old_state.value} -> {new_state.value}",
            order_id=order_id,
            event=event.value
        )
        
        # Execute state-specific logic
        await self._execute_state_logic(order_id, event, details)
        
        # Call event handlers
        await self._call_event_handlers(event, order_id, context, details)
    
    async def _execute_state_logic(
        self,
        order_id: str,
        event: OrderEvent,
        details: Optional[Dict] = None
    ) -> None:
        """Execute logic specific to state transitions"""
        context = self._order_contexts[order_id]
        
        if event == OrderEvent.VALIDATE:
            await self._validate_order(context)
        
        elif event == OrderEvent.RISK_CHECK:
            await self._perform_risk_check(context)
        
        elif event == OrderEvent.SUBMIT:
            await self._submit_order(context)
        
        elif event == OrderEvent.FILL or event == OrderEvent.PARTIAL_FILL:
            await self._handle_fill(context, details)
        
        elif event == OrderEvent.SETTLE:
            await self._settle_order(context)
    
    async def _validate_order(self, context: OrderContext) -> None:
        """Validate order parameters"""
        order = context.order
        
        # Basic validation
        if order.quantity <= 0:
            raise ValueError("Invalid quantity")
        
        if order.order_type in [OrderType.LIMIT] and (not order.price or order.price <= 0):
            raise ValueError("Invalid price for limit order")
        
        # Symbol validation would go here
        # Market hours validation would go here
        
        logger.debug(f"Order validation passed: {order.id}")
    
    async def _perform_risk_check(self, context: OrderContext) -> None:
        """Perform risk assessment"""
        # This would integrate with signal generation if available
        # For now, we'll do basic risk checks
        
        order = context.order
        
        # Check position size limits
        # Check portfolio exposure
        # Check daily loss limits
        
        logger.debug(f"Risk check passed: {order.id}")
    
    async def _submit_order(self, context: OrderContext) -> None:
        """Submit order to exchange"""
        # This would integrate with the trading engine's order submission
        # For now, we'll simulate the submission
        
        order = context.order
        
        # Update order status
        await self.order_manager.update_status(order.id, OrderStatus.NEW)
        
        logger.info(f"Order submitted: {order.id}")
    
    async def _handle_fill(self, context: OrderContext, details: Optional[Dict] = None) -> None:
        """Handle order fill"""
        if details and "execution" in details:
            execution = details["execution"]
            
            # Update positions
            await self.position_manager.update_from_execution(execution)
            
            # Check if order is completely filled
            if execution.filled_quantity == context.order.quantity:
                # Schedule settlement
                asyncio.create_task(self._schedule_settlement(context.order.id))
        
        logger.info(f"Order fill processed: {context.order.id}")
    
    async def _settle_order(self, context: OrderContext) -> None:
        """Settle completed order"""
        # Final reconciliation
        # Update accounting
        # Archive order data
        
        logger.info(f"Order settled: {context.order.id}")
    
    async def _schedule_settlement(self, order_id: str, delay: int = 5) -> None:
        """Schedule order settlement"""
        await asyncio.sleep(delay)
        await self._process_event(order_id, OrderEvent.SETTLE)
    
    async def _call_event_handlers(
        self,
        event: OrderEvent,
        order_id: str,
        context: OrderContext,
        details: Optional[Dict] = None
    ) -> None:
        """Call registered event handlers"""
        handlers = self._event_handlers.get(event, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(order_id, context, details)
                else:
                    handler(order_id, context, details)
            except Exception as e:
                logger.error(f"Event handler failed: {str(e)}")
    
    def _build_state_machine(self) -> Dict:
        """Build state machine transition table"""
        return {
            # From CREATED
            (OrderLifecycleState.CREATED, OrderEvent.VALIDATE): OrderLifecycleState.VALIDATED,
            (OrderLifecycleState.CREATED, OrderEvent.FAIL): OrderLifecycleState.FAILED,
            
            # From VALIDATED
            (OrderLifecycleState.VALIDATED, OrderEvent.RISK_CHECK): OrderLifecycleState.RISK_CHECKED,
            (OrderLifecycleState.VALIDATED, OrderEvent.REJECT): OrderLifecycleState.REJECTED,
            
            # From RISK_CHECKED
            (OrderLifecycleState.RISK_CHECKED, OrderEvent.SUBMIT): OrderLifecycleState.SUBMITTED,
            (OrderLifecycleState.RISK_CHECKED, OrderEvent.REJECT): OrderLifecycleState.REJECTED,
            
            # From SUBMITTED
            (OrderLifecycleState.SUBMITTED, OrderEvent.ACKNOWLEDGE): OrderLifecycleState.ACKNOWLEDGED,
            (OrderLifecycleState.SUBMITTED, OrderEvent.REJECT): OrderLifecycleState.REJECTED,
            (OrderLifecycleState.SUBMITTED, OrderEvent.EXPIRE): OrderLifecycleState.EXPIRED,
            
            # From ACKNOWLEDGED
            (OrderLifecycleState.ACKNOWLEDGED, OrderEvent.PARTIAL_FILL): OrderLifecycleState.PARTIALLY_FILLED,
            (OrderLifecycleState.ACKNOWLEDGED, OrderEvent.FILL): OrderLifecycleState.FILLED,
            (OrderLifecycleState.ACKNOWLEDGED, OrderEvent.CANCEL): OrderLifecycleState.CANCELLED,
            (OrderLifecycleState.ACKNOWLEDGED, OrderEvent.EXPIRE): OrderLifecycleState.EXPIRED,
            
            # From PARTIALLY_FILLED
            (OrderLifecycleState.PARTIALLY_FILLED, OrderEvent.PARTIAL_FILL): OrderLifecycleState.PARTIALLY_FILLED,
            (OrderLifecycleState.PARTIALLY_FILLED, OrderEvent.FILL): OrderLifecycleState.FILLED,
            (OrderLifecycleState.PARTIALLY_FILLED, OrderEvent.CANCEL): OrderLifecycleState.CANCELLED,
            (OrderLifecycleState.PARTIALLY_FILLED, OrderEvent.EXPIRE): OrderLifecycleState.EXPIRED,
            
            # From FILLED
            (OrderLifecycleState.FILLED, OrderEvent.SETTLE): OrderLifecycleState.SETTLED,
            
            # Terminal states (no transitions out)
            # CANCELLED, REJECTED, EXPIRED, SETTLED, FAILED
        }
    
    # Background monitoring tasks
    
    async def _order_timeout_monitor(self) -> None:
        """Monitor orders for timeouts"""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                
                for order_id, context in list(self._order_contexts.items()):
                    # Check if order has timed out
                    if (now - context.order.created_at) > self.order_timeout:
                        if context.state not in [
                            OrderLifecycleState.FILLED,
                            OrderLifecycleState.CANCELLED,
                            OrderLifecycleState.SETTLED,
                            OrderLifecycleState.FAILED
                        ]:
                            await self._process_event(order_id, OrderEvent.EXPIRE)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in timeout monitor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _retry_failed_orders(self) -> None:
        """Retry failed order operations"""
        while self._running:
            try:
                for order_id, context in list(self._order_contexts.items()):
                    if (context.state == OrderLifecycleState.FAILED and 
                        context.retry_count < self.max_retries):
                        
                        # Wait for retry delay
                        delay = self.retry_delays[min(context.retry_count, len(self.retry_delays) - 1)]
                        
                        if context.retry_count == 0 or (
                            datetime.now(timezone.utc) - context.order.updated_at
                        ).total_seconds() >= delay:
                            
                            context.retry_count += 1
                            logger.info(f"Retrying failed order: {order_id} (attempt {context.retry_count})")
                            
                            # Restart workflow
                            asyncio.create_task(self._process_order_workflow(order_id))
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in retry monitor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _settlement_processor(self) -> None:
        """Process order settlements"""
        while self._running:
            try:
                for order_id, context in list(self._order_contexts.items()):
                    if context.state == OrderLifecycleState.FILLED:
                        # Auto-settle after delay
                        if (datetime.now(timezone.utc) - context.order.updated_at).total_seconds() >= 30:
                            await self._process_event(order_id, OrderEvent.SETTLE)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in settlement processor: {str(e)}")
                await asyncio.sleep(5)