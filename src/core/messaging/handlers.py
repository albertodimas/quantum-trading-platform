"""
Trading-specific Event Handlers for the Quantum Trading Platform.

Features:
- Trading event processing and routing
- Market data handling and normalization
- Risk management event handling
- System monitoring and alerting
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .event_bus import EventHandler, Event, EventFilter, EventPriority, create_event_filter
from ..observability.logger import get_logger
from ..observability.metrics import get_metrics_collector
from ..observability.tracing import trace_async


logger = get_logger(__name__)


class TradingEventType(Enum):
    """Trading event types."""
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FAILED = "order.failed"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    BALANCE_UPDATED = "balance.updated"
    TRADE_EXECUTED = "trade.executed"
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    RISK_LIMIT_EXCEEDED = "risk.limit_exceeded"
    MARGIN_CALL = "risk.margin_call"


class MarketEventType(Enum):
    """Market data event types."""
    PRICE_UPDATE = "market.price_update"
    ORDERBOOK_UPDATE = "market.orderbook_update"
    TRADE_UPDATE = "market.trade_update"
    TICKER_UPDATE = "market.ticker_update"
    CANDLE_UPDATE = "market.candle_update"
    MARKET_STATUS_CHANGE = "market.status_change"
    VOLUME_SPIKE = "market.volume_spike"
    PRICE_ALERT = "market.price_alert"


class SystemEventType(Enum):
    """System event types."""
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    CONNECTION_LOST = "system.connection_lost"
    CONNECTION_RESTORED = "system.connection_restored"
    ERROR_OCCURRED = "system.error_occurred"
    HEALTH_CHECK_FAILED = "system.health_check_failed"
    RESOURCE_THRESHOLD_EXCEEDED = "system.resource_threshold_exceeded"


@dataclass
class TradingEventData:
    """Trading event data structure."""
    symbol: str
    exchange: str
    order_id: Optional[str] = None
    side: Optional[str] = None  # 'buy' or 'sell'
    quantity: Optional[float] = None
    price: Optional[float] = None
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None
    commission: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "order_id": self.order_id,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "strategy_id": self.strategy_id,
            "account_id": self.account_id,
            "commission": self.commission
        }


@dataclass
class MarketEventData:
    """Market event data structure."""
    symbol: str
    exchange: str
    price: Optional[float] = None
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "high": self.high,
            "low": self.low,
            "change": self.change,
            "change_percent": self.change_percent
        }


class TradingEventHandler(EventHandler):
    """
    Handler for trading-related events.
    
    Features:
    - Order lifecycle tracking
    - Position management
    - Trade execution monitoring
    - Performance analytics
    """
    
    def __init__(self, handler_id: Optional[str] = None):
        super().__init__(handler_id)
        self._active_orders: Dict[str, Dict[str, Any]] = {}
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._trade_history: List[Dict[str, Any]] = []
        self._performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_commission": 0.0
        }
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        trading_event_types = [e.value for e in TradingEventType]
        return event.type in trading_event_types
    
    def get_filter(self) -> Optional[EventFilter]:
        """Get event filter for trading events."""
        return create_event_filter(
            event_types=[e.value for e in TradingEventType],
            priority_levels=[EventPriority.HIGH, EventPriority.CRITICAL]
        )
    
    @trace_async(name="handle_trading_event", tags={"handler": "trading"})
    async def handle(self, event: Event) -> bool:
        """Handle trading events."""
        try:
            event_type = TradingEventType(event.type)
            
            if event_type == TradingEventType.ORDER_PLACED:
                return await self._handle_order_placed(event)
            elif event_type == TradingEventType.ORDER_FILLED:
                return await self._handle_order_filled(event)
            elif event_type == TradingEventType.ORDER_CANCELLED:
                return await self._handle_order_cancelled(event)
            elif event_type == TradingEventType.ORDER_FAILED:
                return await self._handle_order_failed(event)
            elif event_type == TradingEventType.POSITION_OPENED:
                return await self._handle_position_opened(event)
            elif event_type == TradingEventType.POSITION_CLOSED:
                return await self._handle_position_closed(event)
            elif event_type == TradingEventType.TRADE_EXECUTED:
                return await self._handle_trade_executed(event)
            else:
                self._logger.debug("Unhandled trading event type", event_type=event.type)
                return True
                
        except Exception as e:
            self._logger.error("Error handling trading event", 
                             event_id=event.id,
                             event_type=event.type,
                             error=str(e))
            return False
    
    async def _handle_order_placed(self, event: Event) -> bool:
        """Handle order placed event."""
        try:
            data = TradingEventData(**event.data)
            
            # Track active order
            if data.order_id:
                self._active_orders[data.order_id] = {
                    "event_id": event.id,
                    "symbol": data.symbol,
                    "exchange": data.exchange,
                    "side": data.side,
                    "quantity": data.quantity,
                    "price": data.price,
                    "strategy_id": data.strategy_id,
                    "placed_at": event.timestamp,
                    "status": "active"
                }
            
            if self._metrics:
                self._metrics.record_metric("trading.orders.placed", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange,
                    "side": data.side or "unknown"
                })
            
            self._logger.info("Order placed",
                            order_id=data.order_id,
                            symbol=data.symbol,
                            side=data.side,
                            quantity=data.quantity,
                            price=data.price)
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling order placed event", error=str(e))
            return False
    
    async def _handle_order_filled(self, event: Event) -> bool:
        """Handle order filled event."""
        try:
            data = TradingEventData(**event.data)
            
            # Update order status
            if data.order_id and data.order_id in self._active_orders:
                self._active_orders[data.order_id]["status"] = "filled"
                self._active_orders[data.order_id]["filled_at"] = event.timestamp
                self._active_orders[data.order_id]["fill_price"] = data.price
            
            # Record trade
            trade_record = {
                "event_id": event.id,
                "order_id": data.order_id,
                "symbol": data.symbol,
                "exchange": data.exchange,
                "side": data.side,
                "quantity": data.quantity,
                "price": data.price,
                "commission": data.commission or 0.0,
                "executed_at": event.timestamp,
                "strategy_id": data.strategy_id
            }
            
            self._trade_history.append(trade_record)
            self._update_performance_metrics(trade_record)
            
            if self._metrics:
                self._metrics.record_trade(
                    data.symbol,
                    data.side or "unknown",
                    data.quantity or 0.0,
                    data.price or 0.0
                )
            
            self._logger.info("Order filled",
                            order_id=data.order_id,
                            symbol=data.symbol,
                            side=data.side,
                            quantity=data.quantity,
                            price=data.price)
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling order filled event", error=str(e))
            return False
    
    async def _handle_order_cancelled(self, event: Event) -> bool:
        """Handle order cancelled event."""
        try:
            data = TradingEventData(**event.data)
            
            # Update order status
            if data.order_id and data.order_id in self._active_orders:
                self._active_orders[data.order_id]["status"] = "cancelled"
                self._active_orders[data.order_id]["cancelled_at"] = event.timestamp
            
            if self._metrics:
                self._metrics.record_metric("trading.orders.cancelled", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange
                })
            
            self._logger.info("Order cancelled", order_id=data.order_id)
            return True
            
        except Exception as e:
            self._logger.error("Error handling order cancelled event", error=str(e))
            return False
    
    async def _handle_order_failed(self, event: Event) -> bool:
        """Handle order failed event."""
        try:
            data = TradingEventData(**event.data)
            
            # Update order status
            if data.order_id and data.order_id in self._active_orders:
                self._active_orders[data.order_id]["status"] = "failed"
                self._active_orders[data.order_id]["failed_at"] = event.timestamp
                self._active_orders[data.order_id]["error"] = event.metadata.get("error", "Unknown error")
            
            if self._metrics:
                self._metrics.record_metric("trading.orders.failed", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange
                })
            
            self._logger.warning("Order failed",
                               order_id=data.order_id,
                               error=event.metadata.get("error"))
            return True
            
        except Exception as e:
            self._logger.error("Error handling order failed event", error=str(e))
            return False
    
    async def _handle_position_opened(self, event: Event) -> bool:
        """Handle position opened event."""
        try:
            data = TradingEventData(**event.data)
            
            position_key = f"{data.symbol}_{data.exchange}"
            self._positions[position_key] = {
                "symbol": data.symbol,
                "exchange": data.exchange,
                "side": data.side,
                "quantity": data.quantity,
                "entry_price": data.price,
                "opened_at": event.timestamp,
                "strategy_id": data.strategy_id,
                "unrealized_pnl": 0.0
            }
            
            self._logger.info("Position opened",
                            symbol=data.symbol,
                            side=data.side,
                            quantity=data.quantity,
                            price=data.price)
            return True
            
        except Exception as e:
            self._logger.error("Error handling position opened event", error=str(e))
            return False
    
    async def _handle_position_closed(self, event: Event) -> bool:
        """Handle position closed event."""
        try:
            data = TradingEventData(**event.data)
            
            position_key = f"{data.symbol}_{data.exchange}"
            if position_key in self._positions:
                position = self._positions[position_key]
                position["closed_at"] = event.timestamp
                position["exit_price"] = data.price
                
                # Calculate realized PnL
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                
                if data.side == "buy":
                    pnl = (data.price - entry_price) * quantity
                else:
                    pnl = (entry_price - data.price) * quantity
                
                position["realized_pnl"] = pnl
                self._performance_metrics["total_pnl"] += pnl
                
                del self._positions[position_key]
            
            self._logger.info("Position closed",
                            symbol=data.symbol,
                            price=data.price)
            return True
            
        except Exception as e:
            self._logger.error("Error handling position closed event", error=str(e))
            return False
    
    async def _handle_trade_executed(self, event: Event) -> bool:
        """Handle trade executed event."""
        try:
            data = TradingEventData(**event.data)
            
            # Update performance metrics
            self._performance_metrics["total_trades"] += 1
            
            if data.commission:
                self._performance_metrics["total_commission"] += data.commission
            
            if self._metrics:
                self._metrics.record_metric("trading.trades.executed", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange,
                    "strategy_id": data.strategy_id or "unknown"
                })
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling trade executed event", error=str(e))
            return False
    
    def _update_performance_metrics(self, trade_record: Dict[str, Any]):
        """Update performance metrics based on trade."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated P&L calculation
        pass
    
    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get active orders."""
        return {k: v for k, v in self._active_orders.items() if v["status"] == "active"}
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        return self._positions.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._performance_metrics.copy()


class MarketDataHandler(EventHandler):
    """
    Handler for market data events.
    
    Features:
    - Real-time price monitoring
    - Market data normalization
    - Price alert triggering
    - Market statistics calculation
    """
    
    def __init__(self, handler_id: Optional[str] = None):
        super().__init__(handler_id)
        self._price_data: Dict[str, Dict[str, Any]] = {}
        self._price_alerts: List[Dict[str, Any]] = []
        self._market_stats = {
            "updates_processed": 0,
            "symbols_tracked": 0,
            "price_alerts_triggered": 0
        }
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        market_event_types = [e.value for e in MarketEventType]
        return event.type in market_event_types
    
    def get_filter(self) -> Optional[EventFilter]:
        """Get event filter for market data events."""
        return create_event_filter(
            event_types=[e.value for e in MarketEventType]
        )
    
    @trace_async(name="handle_market_event", tags={"handler": "market_data"})
    async def handle(self, event: Event) -> bool:
        """Handle market data events."""
        try:
            event_type = MarketEventType(event.type)
            
            if event_type == MarketEventType.PRICE_UPDATE:
                return await self._handle_price_update(event)
            elif event_type == MarketEventType.TICKER_UPDATE:
                return await self._handle_ticker_update(event)
            elif event_type == MarketEventType.ORDERBOOK_UPDATE:
                return await self._handle_orderbook_update(event)
            elif event_type == MarketEventType.VOLUME_SPIKE:
                return await self._handle_volume_spike(event)
            else:
                self._logger.debug("Unhandled market event type", event_type=event.type)
                return True
                
        except Exception as e:
            self._logger.error("Error handling market event",
                             event_id=event.id,
                             event_type=event.type,
                             error=str(e))
            return False
    
    async def _handle_price_update(self, event: Event) -> bool:
        """Handle price update event."""
        try:
            data = MarketEventData(**event.data)
            
            symbol_key = f"{data.symbol}_{data.exchange}"
            
            # Update price data
            self._price_data[symbol_key] = {
                "symbol": data.symbol,
                "exchange": data.exchange,
                "price": data.price,
                "volume": data.volume,
                "updated_at": event.timestamp,
                "change": data.change,
                "change_percent": data.change_percent
            }
            
            # Check price alerts
            await self._check_price_alerts(data)
            
            self._market_stats["updates_processed"] += 1
            self._market_stats["symbols_tracked"] = len(self._price_data)
            
            if self._metrics:
                self._metrics.record_metric("market.price_updates", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange
                })
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling price update event", error=str(e))
            return False
    
    async def _handle_ticker_update(self, event: Event) -> bool:
        """Handle ticker update event."""
        try:
            data = MarketEventData(**event.data)
            
            symbol_key = f"{data.symbol}_{data.exchange}"
            
            # Update comprehensive ticker data
            ticker_data = {
                "symbol": data.symbol,
                "exchange": data.exchange,
                "price": data.price,
                "bid": data.bid,
                "ask": data.ask,
                "high": data.high,
                "low": data.low,
                "volume": data.volume,
                "change": data.change,
                "change_percent": data.change_percent,
                "updated_at": event.timestamp
            }
            
            self._price_data[symbol_key] = ticker_data
            
            if self._metrics:
                self._metrics.record_metric("market.ticker_updates", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange
                })
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling ticker update event", error=str(e))
            return False
    
    async def _handle_orderbook_update(self, event: Event) -> bool:
        """Handle orderbook update event."""
        try:
            # Process orderbook data
            # This would typically involve updating bid/ask spreads and depth
            
            if self._metrics:
                self._metrics.record_metric("market.orderbook_updates", 1)
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling orderbook update event", error=str(e))
            return False
    
    async def _handle_volume_spike(self, event: Event) -> bool:
        """Handle volume spike event."""
        try:
            data = MarketEventData(**event.data)
            
            self._logger.warning("Volume spike detected",
                               symbol=data.symbol,
                               volume=data.volume,
                               threshold=event.metadata.get("threshold"))
            
            if self._metrics:
                self._metrics.record_metric("market.volume_spikes", 1, tags={
                    "symbol": data.symbol,
                    "exchange": data.exchange
                })
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling volume spike event", error=str(e))
            return False
    
    async def _check_price_alerts(self, data: MarketEventData):
        """Check and trigger price alerts."""
        # Simplified price alert checking
        # In practice, you'd have a more sophisticated alert system
        for alert in self._price_alerts:
            if (alert["symbol"] == data.symbol and 
                alert["exchange"] == data.exchange and
                data.price is not None):
                
                if alert["condition"] == "above" and data.price > alert["price"]:
                    await self._trigger_price_alert(alert, data.price)
                elif alert["condition"] == "below" and data.price < alert["price"]:
                    await self._trigger_price_alert(alert, data.price)
    
    async def _trigger_price_alert(self, alert: Dict[str, Any], current_price: float):
        """Trigger a price alert."""
        self._market_stats["price_alerts_triggered"] += 1
        
        self._logger.info("Price alert triggered",
                        symbol=alert["symbol"],
                        condition=alert["condition"],
                        target_price=alert["price"],
                        current_price=current_price)
    
    def add_price_alert(self, symbol: str, exchange: str, price: float, condition: str):
        """Add a price alert."""
        alert = {
            "symbol": symbol,
            "exchange": exchange,
            "price": price,
            "condition": condition,  # "above" or "below"
            "created_at": datetime.utcnow()
        }
        self._price_alerts.append(alert)
    
    def get_market_data(self, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for symbol."""
        symbol_key = f"{symbol}_{exchange}"
        return self._price_data.get(symbol_key)
    
    def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics."""
        return self._market_stats.copy()


class RiskEventHandler(EventHandler):
    """
    Handler for risk management events.
    
    Features:
    - Risk limit monitoring
    - Margin call handling
    - Portfolio risk assessment
    - Risk alert system
    """
    
    def __init__(self, handler_id: Optional[str] = None):
        super().__init__(handler_id)
        self._risk_violations: List[Dict[str, Any]] = []
        self._risk_limits = {
            "max_position_size": 100000.0,
            "max_daily_loss": 5000.0,
            "max_leverage": 10.0,
            "max_concentration": 0.2  # 20% max per symbol
        }
        
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        return event.type in [
            TradingEventType.RISK_LIMIT_EXCEEDED.value,
            TradingEventType.MARGIN_CALL.value
        ]
    
    def get_filter(self) -> Optional[EventFilter]:
        """Get event filter for risk events."""
        return create_event_filter(
            event_types=[
                TradingEventType.RISK_LIMIT_EXCEEDED.value,
                TradingEventType.MARGIN_CALL.value
            ],
            priority_levels=[EventPriority.CRITICAL]
        )
    
    async def handle(self, event: Event) -> bool:
        """Handle risk events."""
        try:
            if event.type == TradingEventType.RISK_LIMIT_EXCEEDED.value:
                return await self._handle_risk_limit_exceeded(event)
            elif event.type == TradingEventType.MARGIN_CALL.value:
                return await self._handle_margin_call(event)
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling risk event", error=str(e))
            return False
    
    async def _handle_risk_limit_exceeded(self, event: Event) -> bool:
        """Handle risk limit exceeded event."""
        try:
            violation = {
                "event_id": event.id,
                "timestamp": event.timestamp,
                "limit_type": event.metadata.get("limit_type"),
                "current_value": event.metadata.get("current_value"),
                "limit_value": event.metadata.get("limit_value"),
                "symbol": event.metadata.get("symbol"),
                "account_id": event.metadata.get("account_id")
            }
            
            self._risk_violations.append(violation)
            
            self._logger.critical("Risk limit exceeded",
                                limit_type=violation["limit_type"],
                                current_value=violation["current_value"],
                                limit_value=violation["limit_value"],
                                symbol=violation["symbol"])
            
            # Trigger risk management actions
            await self._trigger_risk_actions(violation)
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling risk limit exceeded event", error=str(e))
            return False
    
    async def _handle_margin_call(self, event: Event) -> bool:
        """Handle margin call event."""
        try:
            self._logger.critical("Margin call triggered",
                                account_id=event.metadata.get("account_id"),
                                margin_level=event.metadata.get("margin_level"),
                                required_margin=event.metadata.get("required_margin"))
            
            # Trigger emergency risk actions
            await self._trigger_emergency_actions(event)
            
            return True
            
        except Exception as e:
            self._logger.error("Error handling margin call event", error=str(e))
            return False
    
    async def _trigger_risk_actions(self, violation: Dict[str, Any]):
        """Trigger risk management actions."""
        # In a real implementation, this would:
        # - Close risky positions
        # - Reduce position sizes
        # - Notify risk managers
        # - Update risk parameters
        
        self._logger.info("Risk actions triggered", violation_id=violation["event_id"])
    
    async def _trigger_emergency_actions(self, event: Event):
        """Trigger emergency risk actions."""
        # In a real implementation, this would:
        # - Immediately close all positions
        # - Stop all trading
        # - Notify emergency contacts
        # - Implement capital preservation measures
        
        self._logger.critical("Emergency risk actions triggered", event_id=event.id)


class SystemEventHandler(EventHandler):
    """
    Handler for system events.
    
    Features:
    - System health monitoring
    - Service lifecycle tracking
    - Error aggregation and alerting
    - Performance monitoring
    """
    
    def __init__(self, handler_id: Optional[str] = None):
        super().__init__(handler_id)
        self._system_status: Dict[str, str] = {}
        self._error_log: List[Dict[str, Any]] = []
        self._max_error_log_size = 1000
        
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        system_event_types = [e.value for e in SystemEventType]
        return event.type in system_event_types
    
    def get_filter(self) -> Optional[EventFilter]:
        """Get event filter for system events."""
        return create_event_filter(
            event_types=[e.value for e in SystemEventType]
        )
    
    async def handle(self, event: Event) -> bool:
        """Handle system events."""
        try:
            event_type = SystemEventType(event.type)
            
            if event_type == SystemEventType.SERVICE_STARTED:
                return await self._handle_service_started(event)
            elif event_type == SystemEventType.SERVICE_STOPPED:
                return await self._handle_service_stopped(event)
            elif event_type == SystemEventType.ERROR_OCCURRED:
                return await self._handle_error_occurred(event)
            elif event_type == SystemEventType.CONNECTION_LOST:
                return await self._handle_connection_lost(event)
            elif event_type == SystemEventType.CONNECTION_RESTORED:
                return await self._handle_connection_restored(event)
            else:
                self._logger.debug("Unhandled system event type", event_type=event.type)
                return True
                
        except Exception as e:
            self._logger.error("Error handling system event", error=str(e))
            return False
    
    async def _handle_service_started(self, event: Event) -> bool:
        """Handle service started event."""
        service_name = event.metadata.get("service_name", "unknown")
        self._system_status[service_name] = "running"
        
        self._logger.info("Service started", service_name=service_name)
        return True
    
    async def _handle_service_stopped(self, event: Event) -> bool:
        """Handle service stopped event."""
        service_name = event.metadata.get("service_name", "unknown")
        self._system_status[service_name] = "stopped"
        
        self._logger.warning("Service stopped", service_name=service_name)
        return True
    
    async def _handle_error_occurred(self, event: Event) -> bool:
        """Handle error occurred event."""
        error_record = {
            "timestamp": event.timestamp,
            "service": event.metadata.get("service", "unknown"),
            "error_type": event.metadata.get("error_type", "unknown"),
            "error_message": event.metadata.get("error_message", ""),
            "severity": event.metadata.get("severity", "low")
        }
        
        self._error_log.append(error_record)
        
        # Trim error log if too large
        if len(self._error_log) > self._max_error_log_size:
            self._error_log = self._error_log[-self._max_error_log_size:]
        
        self._logger.error("System error occurred",
                         service=error_record["service"],
                         error_type=error_record["error_type"],
                         severity=error_record["severity"])
        
        return True
    
    async def _handle_connection_lost(self, event: Event) -> bool:
        """Handle connection lost event."""
        connection_name = event.metadata.get("connection", "unknown")
        
        self._logger.warning("Connection lost", connection=connection_name)
        
        # Trigger connection recovery if needed
        await self._trigger_connection_recovery(connection_name)
        
        return True
    
    async def _handle_connection_restored(self, event: Event) -> bool:
        """Handle connection restored event."""
        connection_name = event.metadata.get("connection", "unknown")
        
        self._logger.info("Connection restored", connection=connection_name)
        return True
    
    async def _trigger_connection_recovery(self, connection_name: str):
        """Trigger connection recovery procedures."""
        # In a real implementation, this would attempt to restore connections
        self._logger.info("Triggering connection recovery", connection=connection_name)
    
    def get_system_status(self) -> Dict[str, str]:
        """Get current system status."""
        return self._system_status.copy()
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self._error_log[-limit:] if self._error_log else []