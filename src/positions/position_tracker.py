"""
Position Tracking System for the Quantum Trading Platform.

Features:
- Real-time position tracking across multiple exchanges
- P&L calculation (realized and unrealized)
- Position aggregation and netting
- Historical position tracking
- Risk metrics calculation
- Position reconciliation
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import uuid
from collections import defaultdict

from ..core.observability.logger import get_logger
from ..core.observability.metrics import get_metrics_collector
from ..core.observability.tracing import trace_async
from ..core.messaging.event_bus import get_event_bus, Event, EventPriority
from ..core.cache.cache_manager import CacheManager
from ..exchange.exchange_interface import Order, OrderSide, OrderStatus


logger = get_logger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    order_id: str
    symbol: str
    exchange: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position information for a symbol on an exchange."""
    symbol: str
    exchange: str
    quantity: Decimal  # Positive for long, negative for short
    average_entry_price: Decimal
    current_price: Optional[Decimal] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # P&L fields
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    
    # Risk metrics
    market_value: Decimal = Decimal("0")
    exposure: Decimal = Decimal("0")
    
    # Trade history
    trades: List[Trade] = field(default_factory=list)
    open_trades: List[Trade] = field(default_factory=list)  # FIFO queue
    
    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    @property
    def abs_quantity(self) -> Decimal:
        """Get absolute quantity."""
        return abs(self.quantity)
    
    @property
    def total_pnl(self) -> Decimal:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def net_pnl(self) -> Decimal:
        """Get net P&L after commissions."""
        return self.total_pnl - self.total_commission
    
    def update_unrealized_pnl(self, current_price: Decimal):
        """Update unrealized P&L based on current price."""
        self.current_price = current_price
        
        if self.quantity != 0:
            if self.side == PositionSide.LONG:
                self.unrealized_pnl = (current_price - self.average_entry_price) * self.quantity
            else:  # SHORT
                self.unrealized_pnl = (self.average_entry_price - current_price) * abs(self.quantity)
            
            self.market_value = abs(self.quantity) * current_price
            self.exposure = self.market_value
        else:
            self.unrealized_pnl = Decimal("0")
            self.market_value = Decimal("0")
            self.exposure = Decimal("0")
        
        self.last_updated = datetime.utcnow()


@dataclass
class PositionSnapshot:
    """Point-in-time position snapshot."""
    timestamp: datetime
    positions: Dict[str, Position]  # symbol -> Position
    total_market_value: Decimal
    total_unrealized_pnl: Decimal
    total_realized_pnl: Decimal
    total_exposure: Decimal


class PositionTracker:
    """
    Position tracking system for real-time position management.
    
    Features:
    - Multi-exchange position tracking
    - FIFO-based P&L calculation
    - Real-time position updates
    - Historical snapshots
    - Position reconciliation
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self._logger = get_logger(self.__class__.__name__)
        self._metrics = get_metrics_collector().get_collector("trading")
        self._event_bus = get_event_bus()
        self._cache = cache_manager
        
        # Position storage: {symbol: {exchange: Position}}
        self._positions: Dict[str, Dict[str, Position]] = defaultdict(dict)
        
        # Trade history
        self._trades: List[Trade] = []
        self._max_trade_history = 100000
        
        # Position snapshots
        self._snapshots: List[PositionSnapshot] = []
        self._max_snapshots = 1000
        
        # Price cache for unrealized P&L calculation
        self._price_cache: Dict[str, Decimal] = {}
        
        # Statistics
        self._stats = {
            "trades_processed": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "total_realized_pnl": Decimal("0"),
            "total_commission": Decimal("0")
        }
        
        # Background tasks
        self._tasks = [
            asyncio.create_task(self._update_unrealized_pnl()),
            asyncio.create_task(self._create_snapshots())
        ]
    
    @trace_async(name="process_trade", tags={"component": "position_tracker"})
    async def process_trade(self, trade: Trade) -> Position:
        """
        Process a trade and update positions.
        
        Args:
            trade: Trade to process
            
        Returns:
            Updated position
        """
        symbol = trade.symbol
        exchange = trade.exchange
        
        # Get or create position
        if exchange not in self._positions[symbol]:
            self._positions[symbol][exchange] = Position(
                symbol=symbol,
                exchange=exchange,
                quantity=Decimal("0"),
                average_entry_price=Decimal("0")
            )
        
        position = self._positions[symbol][exchange]
        old_quantity = position.quantity
        
        # Update position based on trade
        if trade.side == OrderSide.BUY:
            await self._process_buy_trade(position, trade)
        else:
            await self._process_sell_trade(position, trade)
        
        # Add trade to history
        position.trades.append(trade)
        self._trades.append(trade)
        
        # Trim trade history if needed
        if len(self._trades) > self._max_trade_history:
            self._trades = self._trades[-self._max_trade_history:]
        
        # Update commission
        position.total_commission += trade.commission
        self._stats["total_commission"] += trade.commission
        
        # Update statistics
        self._stats["trades_processed"] += 1
        
        # Check if position was opened or closed
        if old_quantity == 0 and position.quantity != 0:
            self._stats["positions_opened"] += 1
            await self._publish_position_event("position.opened", position)
        elif old_quantity != 0 and position.quantity == 0:
            self._stats["positions_closed"] += 1
            await self._publish_position_event("position.closed", position)
        else:
            await self._publish_position_event("position.updated", position)
        
        # Update metrics
        if self._metrics:
            self._metrics.record_metric("positions.trades_processed", 1, tags={
                "symbol": symbol,
                "exchange": exchange,
                "side": trade.side.value
            })
        
        # Cache position
        if self._cache:
            cache_key = f"position:{symbol}:{exchange}"
            await self._cache.set_async(cache_key, position.__dict__, ttl=300)
        
        self._logger.info("Trade processed",
                         trade_id=trade.trade_id,
                         symbol=symbol,
                         side=trade.side.value,
                         quantity=str(trade.quantity),
                         price=str(trade.price),
                         position_quantity=str(position.quantity))
        
        return position
    
    async def _process_buy_trade(self, position: Position, trade: Trade):
        """Process a buy trade."""
        if position.side == PositionSide.LONG or position.side == PositionSide.FLAT:
            # Adding to long position or opening new long
            total_value = (position.quantity * position.average_entry_price + 
                          trade.quantity * trade.price)
            position.quantity += trade.quantity
            position.average_entry_price = total_value / position.quantity if position.quantity > 0 else Decimal("0")
            
            # Add to open trades (FIFO)
            position.open_trades.append(trade)
            
        else:  # SHORT position
            # Closing short position
            if trade.quantity >= abs(position.quantity):
                # Fully closing short position
                realized_pnl = self._calculate_realized_pnl_short(position, trade.price, abs(position.quantity))
                position.realized_pnl += realized_pnl
                self._stats["total_realized_pnl"] += realized_pnl
                
                # Clear open trades
                position.open_trades.clear()
                
                # Remaining quantity becomes long
                remaining = trade.quantity - abs(position.quantity)
                position.quantity = remaining
                position.average_entry_price = trade.price if remaining > 0 else Decimal("0")
                
                if remaining > 0:
                    # Add remaining as new open trade
                    remaining_trade = Trade(
                        trade_id=f"{trade.trade_id}_remaining",
                        order_id=trade.order_id,
                        symbol=trade.symbol,
                        exchange=trade.exchange,
                        side=trade.side,
                        quantity=remaining,
                        price=trade.price,
                        commission=Decimal("0"),
                        timestamp=trade.timestamp
                    )
                    position.open_trades.append(remaining_trade)
            else:
                # Partially closing short position
                realized_pnl = self._calculate_realized_pnl_short(position, trade.price, trade.quantity)
                position.realized_pnl += realized_pnl
                self._stats["total_realized_pnl"] += realized_pnl
                
                position.quantity += trade.quantity  # Reduce short position
                
                # Update open trades (FIFO)
                self._update_open_trades_short(position, trade.quantity)
    
    async def _process_sell_trade(self, position: Position, trade: Trade):
        """Process a sell trade."""
        if position.side == PositionSide.SHORT or position.side == PositionSide.FLAT:
            # Adding to short position or opening new short
            total_value = (abs(position.quantity) * position.average_entry_price + 
                          trade.quantity * trade.price)
            position.quantity -= trade.quantity
            position.average_entry_price = total_value / abs(position.quantity) if position.quantity != 0 else Decimal("0")
            
            # Add to open trades (FIFO)
            position.open_trades.append(trade)
            
        else:  # LONG position
            # Closing long position
            if trade.quantity >= position.quantity:
                # Fully closing long position
                realized_pnl = self._calculate_realized_pnl_long(position, trade.price, position.quantity)
                position.realized_pnl += realized_pnl
                self._stats["total_realized_pnl"] += realized_pnl
                
                # Clear open trades
                position.open_trades.clear()
                
                # Remaining quantity becomes short
                remaining = trade.quantity - position.quantity
                position.quantity = -remaining
                position.average_entry_price = trade.price if remaining > 0 else Decimal("0")
                
                if remaining > 0:
                    # Add remaining as new open trade
                    remaining_trade = Trade(
                        trade_id=f"{trade.trade_id}_remaining",
                        order_id=trade.order_id,
                        symbol=trade.symbol,
                        exchange=trade.exchange,
                        side=trade.side,
                        quantity=remaining,
                        price=trade.price,
                        commission=Decimal("0"),
                        timestamp=trade.timestamp
                    )
                    position.open_trades.append(remaining_trade)
            else:
                # Partially closing long position
                realized_pnl = self._calculate_realized_pnl_long(position, trade.price, trade.quantity)
                position.realized_pnl += realized_pnl
                self._stats["total_realized_pnl"] += realized_pnl
                
                position.quantity -= trade.quantity  # Reduce long position
                
                # Update open trades (FIFO)
                self._update_open_trades_long(position, trade.quantity)
    
    def _calculate_realized_pnl_long(self, position: Position, exit_price: Decimal, 
                                    quantity: Decimal) -> Decimal:
        """Calculate realized P&L for closing long position (FIFO)."""
        realized_pnl = Decimal("0")
        remaining_quantity = quantity
        
        for trade in list(position.open_trades):
            if remaining_quantity <= 0:
                break
            
            if trade.side == OrderSide.BUY:
                close_quantity = min(trade.quantity, remaining_quantity)
                pnl = (exit_price - trade.price) * close_quantity
                realized_pnl += pnl
                remaining_quantity -= close_quantity
        
        return realized_pnl
    
    def _calculate_realized_pnl_short(self, position: Position, exit_price: Decimal,
                                     quantity: Decimal) -> Decimal:
        """Calculate realized P&L for closing short position (FIFO)."""
        realized_pnl = Decimal("0")
        remaining_quantity = quantity
        
        for trade in list(position.open_trades):
            if remaining_quantity <= 0:
                break
            
            if trade.side == OrderSide.SELL:
                close_quantity = min(trade.quantity, remaining_quantity)
                pnl = (trade.price - exit_price) * close_quantity
                realized_pnl += pnl
                remaining_quantity -= close_quantity
        
        return realized_pnl
    
    def _update_open_trades_long(self, position: Position, closed_quantity: Decimal):
        """Update open trades after partially closing long position (FIFO)."""
        remaining_to_close = closed_quantity
        new_open_trades = []
        
        for trade in position.open_trades:
            if remaining_to_close <= 0:
                new_open_trades.append(trade)
            elif trade.side == OrderSide.BUY:
                if trade.quantity > remaining_to_close:
                    # Partially close this trade
                    updated_trade = Trade(
                        trade_id=trade.trade_id,
                        order_id=trade.order_id,
                        symbol=trade.symbol,
                        exchange=trade.exchange,
                        side=trade.side,
                        quantity=trade.quantity - remaining_to_close,
                        price=trade.price,
                        commission=trade.commission,
                        timestamp=trade.timestamp,
                        metadata=trade.metadata
                    )
                    new_open_trades.append(updated_trade)
                    remaining_to_close = Decimal("0")
                else:
                    # Fully close this trade
                    remaining_to_close -= trade.quantity
            else:
                new_open_trades.append(trade)
        
        position.open_trades = new_open_trades
    
    def _update_open_trades_short(self, position: Position, closed_quantity: Decimal):
        """Update open trades after partially closing short position (FIFO)."""
        remaining_to_close = closed_quantity
        new_open_trades = []
        
        for trade in position.open_trades:
            if remaining_to_close <= 0:
                new_open_trades.append(trade)
            elif trade.side == OrderSide.SELL:
                if trade.quantity > remaining_to_close:
                    # Partially close this trade
                    updated_trade = Trade(
                        trade_id=trade.trade_id,
                        order_id=trade.order_id,
                        symbol=trade.symbol,
                        exchange=trade.exchange,
                        side=trade.side,
                        quantity=trade.quantity - remaining_to_close,
                        price=trade.price,
                        commission=trade.commission,
                        timestamp=trade.timestamp,
                        metadata=trade.metadata
                    )
                    new_open_trades.append(updated_trade)
                    remaining_to_close = Decimal("0")
                else:
                    # Fully close this trade
                    remaining_to_close -= trade.quantity
            else:
                new_open_trades.append(trade)
        
        position.open_trades = new_open_trades
    
    async def get_position(self, symbol: str, exchange: Optional[str] = None) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Optional exchange filter
            
        Returns:
            Position if exists, None otherwise
        """
        if symbol not in self._positions:
            return None
        
        if exchange:
            return self._positions[symbol].get(exchange)
        
        # Aggregate positions across exchanges
        positions = list(self._positions[symbol].values())
        if not positions:
            return None
        
        if len(positions) == 1:
            return positions[0]
        
        # Create aggregated position
        total_quantity = sum(p.quantity for p in positions)
        
        # Calculate weighted average entry price
        if total_quantity != 0:
            total_value = sum(p.quantity * p.average_entry_price for p in positions)
            avg_entry_price = abs(total_value / total_quantity)
        else:
            avg_entry_price = Decimal("0")
        
        aggregated = Position(
            symbol=symbol,
            exchange="AGGREGATE",
            quantity=total_quantity,
            average_entry_price=avg_entry_price,
            realized_pnl=sum(p.realized_pnl for p in positions),
            unrealized_pnl=sum(p.unrealized_pnl for p in positions),
            total_commission=sum(p.total_commission for p in positions),
            market_value=sum(p.market_value for p in positions),
            exposure=sum(p.exposure for p in positions)
        )
        
        return aggregated
    
    async def get_all_positions(self, exchange: Optional[str] = None) -> Dict[str, Position]:
        """
        Get all positions.
        
        Args:
            exchange: Optional exchange filter
            
        Returns:
            Dictionary of symbol -> Position
        """
        result = {}
        
        for symbol, exchanges in self._positions.items():
            if exchange:
                if exchange in exchanges:
                    result[symbol] = exchanges[exchange]
            else:
                # Get aggregated position
                position = await self.get_position(symbol)
                if position:
                    result[symbol] = position
        
        return result
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary statistics."""
        all_positions = await self.get_all_positions()
        
        long_positions = [p for p in all_positions.values() if p.side == PositionSide.LONG]
        short_positions = [p for p in all_positions.values() if p.side == PositionSide.SHORT]
        
        total_market_value = sum(p.market_value for p in all_positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in all_positions.values())
        total_realized_pnl = sum(p.realized_pnl for p in all_positions.values())
        total_exposure = sum(p.exposure for p in all_positions.values())
        
        return {
            "total_positions": len(all_positions),
            "long_positions": len(long_positions),
            "short_positions": len(short_positions),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_unrealized_pnl + total_realized_pnl),
            "total_exposure": str(total_exposure),
            "total_commission": str(self._stats["total_commission"]),
            "net_pnl": str(total_unrealized_pnl + total_realized_pnl - self._stats["total_commission"])
        }
    
    async def update_market_price(self, symbol: str, price: Decimal):
        """Update market price for a symbol."""
        self._price_cache[symbol] = price
        
        # Update unrealized P&L for all positions of this symbol
        if symbol in self._positions:
            for exchange, position in self._positions[symbol].items():
                position.update_unrealized_pnl(price)
    
    async def _update_unrealized_pnl(self):
        """Background task to update unrealized P&L."""
        while True:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                for symbol, exchanges in self._positions.items():
                    if symbol in self._price_cache:
                        price = self._price_cache[symbol]
                        
                        for position in exchanges.values():
                            if position.quantity != 0:
                                position.update_unrealized_pnl(price)
                
            except Exception as e:
                self._logger.error(f"Error updating unrealized P&L: {e}")
                await asyncio.sleep(30)
    
    async def _create_snapshots(self):
        """Background task to create position snapshots."""
        while True:
            try:
                await asyncio.sleep(300)  # Snapshot every 5 minutes
                
                all_positions = await self.get_all_positions()
                
                if all_positions:
                    snapshot = PositionSnapshot(
                        timestamp=datetime.utcnow(),
                        positions=all_positions.copy(),
                        total_market_value=sum(p.market_value for p in all_positions.values()),
                        total_unrealized_pnl=sum(p.unrealized_pnl for p in all_positions.values()),
                        total_realized_pnl=sum(p.realized_pnl for p in all_positions.values()),
                        total_exposure=sum(p.exposure for p in all_positions.values())
                    )
                    
                    self._snapshots.append(snapshot)
                    
                    # Trim snapshots if needed
                    if len(self._snapshots) > self._max_snapshots:
                        self._snapshots = self._snapshots[-self._max_snapshots:]
                    
                    self._logger.info("Position snapshot created",
                                    positions=len(all_positions),
                                    total_exposure=str(snapshot.total_exposure))
                
            except Exception as e:
                self._logger.error(f"Error creating position snapshot: {e}")
                await asyncio.sleep(60)
    
    async def get_snapshots(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[PositionSnapshot]:
        """Get position snapshots within time range."""
        snapshots = self._snapshots
        
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        return snapshots
    
    async def reconcile_positions(self, exchange_positions: Dict[str, Dict[str, Decimal]]) -> Dict[str, Any]:
        """
        Reconcile positions with exchange-reported positions.
        
        Args:
            exchange_positions: {exchange: {symbol: quantity}}
            
        Returns:
            Reconciliation report
        """
        discrepancies = []
        
        for exchange, symbols in exchange_positions.items():
            for symbol, exchange_quantity in symbols.items():
                # Get our tracked position
                our_position = None
                if symbol in self._positions and exchange in self._positions[symbol]:
                    our_position = self._positions[symbol][exchange]
                
                our_quantity = our_position.quantity if our_position else Decimal("0")
                
                # Check for discrepancy
                diff = abs(our_quantity - exchange_quantity)
                if diff > Decimal("0.00001"):  # Small tolerance for rounding
                    discrepancies.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "our_quantity": str(our_quantity),
                        "exchange_quantity": str(exchange_quantity),
                        "difference": str(diff)
                    })
        
        # Check for positions we have but exchange doesn't report
        for symbol, exchanges in self._positions.items():
            for exchange, position in exchanges.items():
                if position.quantity != 0:
                    if exchange not in exchange_positions or symbol not in exchange_positions[exchange]:
                        discrepancies.append({
                            "exchange": exchange,
                            "symbol": symbol,
                            "our_quantity": str(position.quantity),
                            "exchange_quantity": "0",
                            "difference": str(abs(position.quantity))
                        })
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_discrepancies": len(discrepancies),
            "discrepancies": discrepancies,
            "reconciled": len(discrepancies) == 0
        }
        
        if discrepancies:
            self._logger.warning("Position discrepancies found",
                               count=len(discrepancies))
            
            # Publish reconciliation event
            await self._publish_reconciliation_event(report)
        
        return report
    
    async def _publish_position_event(self, event_type: str, position: Position):
        """Publish position-related event."""
        event_data = {
            "symbol": position.symbol,
            "exchange": position.exchange,
            "quantity": str(position.quantity),
            "side": position.side.value,
            "average_entry_price": str(position.average_entry_price),
            "unrealized_pnl": str(position.unrealized_pnl),
            "realized_pnl": str(position.realized_pnl),
            "total_pnl": str(position.total_pnl),
            "market_value": str(position.market_value),
            "exposure": str(position.exposure)
        }
        
        event = Event(
            type=event_type,
            data=event_data,
            source="position_tracker",
            priority=EventPriority.NORMAL
        )
        
        await self._event_bus.publish(event)
    
    async def _publish_reconciliation_event(self, report: Dict[str, Any]):
        """Publish reconciliation event."""
        event = Event(
            type="position.reconciliation_failed",
            data=report,
            source="position_tracker",
            priority=EventPriority.HIGH
        )
        
        await self._event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get position tracker statistics."""
        return {
            **self._stats,
            "active_positions": sum(
                1 for positions in self._positions.values()
                for p in positions.values()
                if p.quantity != 0
            ),
            "total_symbols": len(self._positions),
            "snapshot_count": len(self._snapshots),
            "trade_history_size": len(self._trades)
        }
    
    async def shutdown(self):
        """Shutdown the position tracker."""
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Create final snapshot
        all_positions = await self.get_all_positions()
        if all_positions:
            final_snapshot = PositionSnapshot(
                timestamp=datetime.utcnow(),
                positions=all_positions,
                total_market_value=sum(p.market_value for p in all_positions.values()),
                total_unrealized_pnl=sum(p.unrealized_pnl for p in all_positions.values()),
                total_realized_pnl=sum(p.realized_pnl for p in all_positions.values()),
                total_exposure=sum(p.exposure for p in all_positions.values())
            )
            self._snapshots.append(final_snapshot)
        
        self._logger.info("Position tracker shutdown complete")