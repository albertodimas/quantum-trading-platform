"""
Core trading engine implementation.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ..core.config import settings
from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import ExchangeManager, OrderSide, OrderType, OrderStatus
from .models import (
    ExecutionReport,
    Order,
    Position,
    Signal,
)
from .order_manager import OrderManager
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .execution_algorithms import ExecutionEngine, ExecutionParameters, ExecutionAlgorithm
from .order_lifecycle import OrderLifecycleManager

logger = get_logger(__name__)


class TradingEngine:
    """
    Core trading engine that orchestrates all trading operations.
    
    This engine manages order flow, position tracking, risk management,
    and coordinates with exchanges for execution.
    """
    
    def __init__(
        self,
        exchange_name: str,
        risk_manager: Optional[RiskManager] = None,
        exchange_manager: Optional[ExchangeManager] = None,
    ) -> None:
        """
        Initialize trading engine.
        
        Args:
            exchange_name: Name of the exchange to trade on
            risk_manager: Risk manager instance (creates default if None)
            exchange_manager: Exchange manager instance (creates default if None)
        """
        self.exchange_name = exchange_name
        self.risk_manager = risk_manager or RiskManager()
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.exchange_manager = exchange_manager
        
        # Advanced execution components
        self.execution_engine = ExecutionEngine(order_manager=self.order_manager)
        self.lifecycle_manager = OrderLifecycleManager(
            order_manager=self.order_manager,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager
        )
        
        # Engine state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self._metrics = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "total_volume": Decimal("0"),
            "total_commission": Decimal("0"),
        }
        
        logger.info(
            "Trading engine initialized",
            exchange=exchange_name,
            risk_limits=self.risk_manager.get_limits(),
            has_exchange_manager=self.exchange_manager is not None
        )
    
    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            logger.warning("Trading engine already running")
            return
        
        logger.info("Starting trading engine")
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._order_monitor()),
            asyncio.create_task(self._position_monitor()),
            asyncio.create_task(self._risk_monitor()),
        ]
        
        # Start advanced execution components
        await self.execution_engine.start()
        await self.lifecycle_manager.start()
        
        logger.info("Trading engine started successfully")
    
    async def stop(self) -> None:
        """Stop the trading engine gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping trading engine")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop advanced execution components
        await self.execution_engine.stop()
        await self.lifecycle_manager.stop()
        
        # Close all positions if configured
        if settings.close_positions_on_stop:
            await self._close_all_positions()
        
        logger.info("Trading engine stopped")
    
    async def process_signal(self, signal: Signal) -> Optional[str]:
        """
        Process a trading signal and create orders if appropriate.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            Order ID if order was created, None otherwise
        """
        logger.info(
            "Processing trading signal",
            symbol=signal.symbol,
            side=signal.side,
            confidence=signal.confidence,
        )
        
        # Risk checks
        risk_check = await self.risk_manager.check_signal(signal)
        if not risk_check.approved:
            logger.warning(
                "Signal rejected by risk manager",
                reason=risk_check.reason,
                signal=signal.dict(),
            )
            return None
        
        # Calculate position size
        position_size = await self._calculate_position_size(signal)
        if position_size <= 0:
            logger.warning("Invalid position size calculated", size=position_size)
            return None
        
        # Create order
        order = await self.order_manager.create_order(
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.LIMIT,
            quantity=position_size,
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata={
                "signal_id": signal.model_dump()["timestamp"],
                "strategy": signal.strategy,
                "confidence": signal.confidence,
            }
        )
        
        # Submit order
        try:
            await self._submit_order(order)
            self._metrics["orders_placed"] += 1
            
            logger.info(
                "Order submitted successfully",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
            )
            
            return order.id
            
        except Exception as e:
            logger.error(
                "Failed to submit order",
                error=str(e),
                order=order.dict(),
            )
            await self.order_manager.update_status(order.id, OrderStatus.REJECTED)
            self._metrics["orders_rejected"] += 1
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation was successful
        """
        order = await self.order_manager.get_order(order_id)
        if not order:
            logger.error("Order not found", order_id=order_id)
            return False
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN]:
            logger.warning(
                "Cannot cancel order in current status",
                order_id=order_id,
                status=order.status,
            )
            return False
        
        try:
            # Cancel on exchange
            await self._cancel_order_on_exchange(order)
            
            # Update local status
            await self.order_manager.update_status(order_id, OrderStatus.CANCELLED)
            
            logger.info("Order cancelled successfully", order_id=order_id)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cancel order",
                error=str(e),
                order_id=order_id,
            )
            return False
    
    async def execute_with_algorithm(
        self,
        signal: Signal,
        algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART,
        time_horizon: int = 60,  # minutes
        participation_rate: Optional[float] = None
    ) -> Optional[str]:
        """
        Execute signal using advanced execution algorithm.
        
        Args:
            signal: Trading signal to execute
            algorithm: Execution algorithm to use
            time_horizon: Time horizon for execution in minutes
            participation_rate: Target participation rate for POV algorithm
            
        Returns:
            Execution ID for tracking
        """
        logger.info(
            "Executing signal with advanced algorithm",
            symbol=signal.symbol,
            side=signal.side,
            algorithm=algorithm.value,
            time_horizon=time_horizon
        )
        
        # Risk checks
        risk_check = await self.risk_manager.check_signal(signal)
        if not risk_check.approved:
            logger.warning(
                "Signal rejected by risk manager",
                reason=risk_check.reason,
                signal=signal.dict(),
            )
            return None
        
        # Calculate position size
        position_size = await self._calculate_position_size(signal)
        if position_size <= 0:
            logger.warning("Invalid position size calculated", size=position_size)
            return None
        
        # Create execution parameters
        execution_params = ExecutionParameters(
            algorithm=algorithm,
            total_quantity=position_size,
            time_horizon=time_horizon,
            participation_rate=participation_rate,
            price_limit=signal.entry_price,
            urgency=0.5,  # Medium urgency
            risk_tolerance=0.3,  # Conservative
            metadata={
                "signal_id": signal.model_dump()["timestamp"],
                "strategy": signal.strategy,
                "confidence": signal.confidence,
            }
        )
        
        # Execute using execution engine
        try:
            execution_id = await self.execution_engine.execute_order(
                symbol=signal.symbol,
                side=signal.side,
                parameters=execution_params
            )
            
            logger.info(
                "Advanced execution started",
                execution_id=execution_id,
                algorithm=algorithm.value,
                quantity=float(position_size)
            )
            
            return execution_id
            
        except Exception as e:
            logger.error(
                "Failed to start advanced execution",
                error=str(e),
                signal=signal.dict(),
            )
            return None
    
    async def create_managed_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create order with full lifecycle management.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID for tracking
        """
        try:
            order_id = await self.lifecycle_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **kwargs
            )
            
            logger.info(
                "Managed order created",
                order_id=order_id,
                symbol=symbol,
                side=side.value,
                quantity=float(quantity)
            )
            
            return order_id
            
        except Exception as e:
            logger.error(
                "Failed to create managed order",
                error=str(e),
                symbol=symbol,
                side=side.value
            )
            return None
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get status of advanced execution"""
        return await self.execution_engine.get_execution_status(execution_id)
    
    async def get_order_lifecycle_status(self, order_id: str) -> Optional[Dict]:
        """Get order lifecycle status"""
        return await self.lifecycle_manager.get_order_status(order_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel advanced execution"""
        return await self.execution_engine.cancel_execution(execution_id)
    
    async def cancel_managed_order(self, order_id: str) -> bool:
        """Cancel managed order"""
        return await self.lifecycle_manager.cancel_order(order_id)
    
    async def get_portfolio_status(self) -> Dict:
        """Get current portfolio status including positions and P&L."""
        positions = await self.position_manager.get_all_positions()
        
        total_value = Decimal("0")
        total_pnl = Decimal("0")
        position_count = 0
        
        position_details = []
        for position in positions.values():
            if position.quantity > 0:
                position_count += 1
                total_value += position.quantity * position.current_price
                total_pnl += position.unrealized_pnl
                
                position_details.append({
                    "symbol": position.symbol,
                    "side": position.side,
                    "quantity": float(position.quantity),
                    "entry_price": float(position.entry_price),
                    "current_price": float(position.current_price),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "pnl_percentage": position.pnl_percentage,
                })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_value": float(total_value),
            "total_pnl": float(total_pnl),
            "position_count": position_count,
            "positions": position_details,
            "metrics": {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in self._metrics.items()
            },
        }
    
    # Private methods
    
    async def _calculate_position_size(self, signal: Signal) -> Decimal:
        """Calculate appropriate position size based on risk parameters."""
        if signal.quantity:
            # Use suggested quantity if provided
            return signal.quantity
        
        # Get account balance
        balance = await self._get_account_balance()
        
        # Calculate based on risk percentage
        risk_amount = balance * (settings.default_risk_percentage / 100)
        
        # Calculate based on stop loss distance
        if signal.stop_loss:
            price_diff = abs(signal.entry_price - signal.stop_loss)
            if price_diff > 0:
                return risk_amount / price_diff
        
        # Default to fixed percentage of balance
        return (balance * (settings.default_risk_percentage / 100)) / signal.entry_price
    
    async def _submit_order(self, order: Order) -> None:
        """Submit order to exchange."""
        if not self.exchange_manager:
            # Fallback to simulation if no exchange manager
            await asyncio.sleep(0.1)  # Simulate network delay
            await self.order_manager.update_status(order.id, OrderStatus.OPEN)
            return
        
        try:
            # Submit order through exchange manager
            exchange_order = await self.exchange_manager.route_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                time_in_force=getattr(order, 'time_in_force', None),
                client_order_id=order.id
            )
            
            # Update order with exchange details
            if exchange_order:
                # Take the first successfully placed order
                first_order = next(iter(exchange_order.values()))
                await self.order_manager.update_from_exchange_order(order.id, first_order)
            else:
                await self.order_manager.update_status(order.id, OrderStatus.REJECTED)
                
        except Exception as e:
            logger.error(f"Failed to submit order to exchange: {str(e)}")
            await self.order_manager.update_status(order.id, OrderStatus.REJECTED)
            raise
    
    async def _cancel_order_on_exchange(self, order: Order) -> None:
        """Cancel order on exchange."""
        if not self.exchange_manager:
            # Fallback to simulation if no exchange manager
            await asyncio.sleep(0.1)  # Simulate network delay
            return
        
        try:
            # Cancel order through exchange manager
            await self.exchange_manager.cancel_order(
                symbol=order.symbol,
                order_id=getattr(order, 'exchange_order_id', order.id)
            )
        except Exception as e:
            logger.error(f"Failed to cancel order on exchange: {str(e)}")
            raise
    
    async def _get_account_balance(self) -> Decimal:
        """Get current account balance."""
        if not self.exchange_manager:
            # Fallback to simulated balance if no exchange manager
            return Decimal("10000")  # Simulated balance
        
        try:
            # Get balance from exchange manager
            balances = await self.exchange_manager.get_balance()
            
            # Return balance in quote currency (default USDT)
            quote_currency = settings.default_quote_currency
            if quote_currency in balances:
                return balances[quote_currency].free
            
            # Fallback to first available balance
            if balances:
                first_balance = next(iter(balances.values()))
                return first_balance.free
            
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {str(e)}")
            return Decimal("10000")  # Fallback to simulated balance
    
    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        positions = await self.position_manager.get_all_positions()
        
        for position in positions.values():
            if position.quantity > 0:
                # Create market order to close
                close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
                
                order = await self.order_manager.create_order(
                    symbol=position.symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    metadata={"reason": "engine_shutdown"}
                )
                
                await self._submit_order(order)
    
    # Background monitoring tasks
    
    async def _order_monitor(self) -> None:
        """Monitor order status and handle fills."""
        while self._running:
            try:
                # Check for order updates
                open_orders = await self.order_manager.get_open_orders()
                
                for order in open_orders:
                    # Check order status from exchange if available
                    if self.exchange_manager:
                        try:
                            # Get real order status from exchange
                            exchange_order = await self.exchange_manager.get_order_status(
                                symbol=order.symbol,
                                order_id=getattr(order, 'exchange_order_id', order.id)
                            )
                            
                            if exchange_order and exchange_order.status != order.status:
                                # Create execution report from exchange order
                                execution = ExecutionReport(
                                    order_id=order.id,
                                    symbol=order.symbol,
                                    side=order.side,
                                    order_type=order.order_type,
                                    status=exchange_order.status,
                                    price=exchange_order.price or Decimal("0"),
                                    quantity=order.quantity,
                                    filled_quantity=exchange_order.executed_qty or Decimal("0"),
                                    remaining_quantity=order.quantity - (exchange_order.executed_qty or Decimal("0")),
                                    average_price=exchange_order.price,
                                    timestamp=datetime.utcnow(),
                                )
                                
                                await self._handle_execution(execution)
                                continue
                                
                        except Exception as e:
                            logger.warning(f"Failed to check order status from exchange: {str(e)}")
                    
                    # Fallback to simulation for orders without exchange integration
                    if order.created_at < datetime.utcnow().timestamp() - 5:
                        execution = ExecutionReport(
                            order_id=order.id,
                            symbol=order.symbol,
                            side=order.side,
                            order_type=order.order_type,
                            status=OrderStatus.FILLED,
                            price=order.price or Decimal("0"),
                            quantity=order.quantity,
                            filled_quantity=order.quantity,
                            remaining_quantity=Decimal("0"),
                            average_price=order.price,
                            timestamp=datetime.utcnow(),
                        )
                        
                        await self._handle_execution(execution)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error("Error in order monitor", error=str(e))
                await asyncio.sleep(5)
    
    async def _position_monitor(self) -> None:
        """Monitor positions and update prices."""
        while self._running:
            try:
                # Update position prices
                positions = await self.position_manager.get_all_positions()
                
                for position in positions.values():
                    # Get real market prices from exchange if available
                    if self.exchange_manager:
                        try:
                            ticker = await self.exchange_manager.get_ticker(position.symbol)
                            if ticker:
                                await self.position_manager.update_price(
                                    position.symbol,
                                    ticker.last_price
                                )
                                continue
                        except Exception as e:
                            logger.warning(f"Failed to get real price for {position.symbol}: {str(e)}")
                    
                    # Fallback to simulated price movements
                    current_price = position.current_price * Decimal("1.001")
                    await self.position_manager.update_price(
                        position.symbol,
                        current_price
                    )
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error("Error in position monitor", error=str(e))
                await asyncio.sleep(5)
    
    async def _risk_monitor(self) -> None:
        """Monitor risk metrics and enforce limits."""
        while self._running:
            try:
                # Check portfolio risk
                portfolio_status = await self.get_portfolio_status()
                risk_status = await self.risk_manager.check_portfolio_risk(
                    portfolio_status
                )
                
                if not risk_status.healthy:
                    logger.warning(
                        "Portfolio risk limit exceeded",
                        reason=risk_status.reason,
                        metrics=risk_status.metrics,
                    )
                    
                    # Take protective action if needed
                    if risk_status.action_required:
                        await self._handle_risk_breach(risk_status)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Error in risk monitor", error=str(e))
                await asyncio.sleep(5)
    
    async def _handle_execution(self, execution: ExecutionReport) -> None:
        """Handle order execution report."""
        logger.info(
            "Handling execution report",
            order_id=execution.order_id,
            status=execution.status,
            filled_quantity=execution.filled_quantity,
        )
        
        # Update order
        await self.order_manager.update_order_from_execution(execution)
        
        # Update position
        if execution.is_filled:
            await self.position_manager.update_from_execution(execution)
            self._metrics["orders_filled"] += 1
            self._metrics["total_volume"] += execution.filled_quantity * execution.average_price
            self._metrics["total_commission"] += execution.commission
    
    async def _handle_risk_breach(self, risk_status) -> None:
        """Handle risk limit breach."""
        logger.critical(
            "Handling risk breach",
            action=risk_status.action_required,
            metrics=risk_status.metrics,
        )
        
        # Implement risk breach handling
        # This could include:
        # - Closing positions
        # - Cancelling open orders
        # - Sending alerts
        # - Pausing trading