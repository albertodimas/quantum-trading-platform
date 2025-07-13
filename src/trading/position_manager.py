"""
Position Management System

Handles position tracking, P&L calculation, and portfolio management.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderSide
from .models import Position, ExecutionReport
from .fifo_calculator import FIFOCalculator, TransactionType

logger = get_logger(__name__)


@injectable
class PositionManager:
    """
    Position Manager handles position tracking and P&L calculation.
    
    Uses FIFO (First In, First Out) method for position calculation
    and supports multiple symbols and exchanges.
    """
    
    def __init__(self):
        """Initialize position manager."""
        # Advanced FIFO calculator for precise P&L tracking
        self.fifo_calculator = FIFOCalculator()
        
        # Legacy position storage for compatibility (will be phased out)
        self._positions: Dict[str, Position] = {}
        
        # Position snapshots for historical tracking
        self._position_snapshots: List[Dict] = []
        
        # Metrics
        self._total_realized_pnl = Decimal("0")
        self._total_unrealized_pnl = Decimal("0")
        
        logger.info("Advanced position manager initialized with FIFO calculator")
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        # Get position from FIFO calculator
        fifo_summary = await self.fifo_calculator.get_position_summary(symbol)
        
        if not fifo_summary:
            return None
        
        # Convert FIFO summary to Position model for compatibility
        return Position(
            symbol=symbol,
            side=OrderSide.BUY if fifo_summary.total_quantity > 0 else OrderSide.SELL,
            quantity=abs(fifo_summary.total_quantity),
            entry_price=fifo_summary.average_price,
            current_price=fifo_summary.current_price,
            unrealized_pnl=fifo_summary.unrealized_pnl,
            realized_pnl=fifo_summary.total_realized_pnl,
            pnl_percentage=float(fifo_summary.unrealized_pnl_percent),
            created_at=fifo_summary.oldest_lot_date or datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        positions = {}
        
        # Get all positions from FIFO calculator
        all_fifo_positions = await self.fifo_calculator.get_all_positions()
        
        for symbol, fifo_summary in all_fifo_positions.items():
            if fifo_summary.total_quantity != 0:  # Only include non-zero positions
                positions[symbol] = Position(
                    symbol=symbol,
                    side=OrderSide.BUY if fifo_summary.total_quantity > 0 else OrderSide.SELL,
                    quantity=abs(fifo_summary.total_quantity),
                    entry_price=fifo_summary.average_price,
                    current_price=fifo_summary.current_price,
                    unrealized_pnl=fifo_summary.unrealized_pnl,
                    realized_pnl=fifo_summary.total_realized_pnl,
                    pnl_percentage=float(fifo_summary.unrealized_pnl_percent),
                    created_at=fifo_summary.oldest_lot_date or datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
        
        return positions
    
    async def update_from_execution(self, execution: ExecutionReport) -> None:
        """
        Update position from execution report using advanced FIFO calculator.
        
        Args:
            execution: Execution report from order fill
        """
        symbol = execution.symbol
        
        # Add transaction to FIFO calculator
        transaction_id = await self.fifo_calculator.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.BUY if execution.side == OrderSide.BUY else TransactionType.SELL,
            side=execution.side,
            quantity=execution.filled_quantity,
            price=execution.average_price or execution.price,
            timestamp=execution.timestamp,
            commission=execution.commission or Decimal("0"),
            metadata={
                "order_id": execution.order_id,
                "commission_asset": execution.commission_asset or "",
                "execution_type": "trade"
            }
        )
        
        logger.info(
            "Position updated from execution using FIFO calculator",
            symbol=symbol,
            side=execution.side.value,
            quantity=float(execution.filled_quantity),
            price=float(execution.average_price or execution.price),
            transaction_id=transaction_id
        )
    
    async def update_price(self, symbol: str, current_price: Decimal) -> None:
        """
        Update current market price for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        # Update price in FIFO calculator
        await self.fifo_calculator.update_current_price(symbol, current_price)
        
        # Get updated position summary
        position_summary = await self.fifo_calculator.get_position_summary(symbol)
        
        if position_summary:
            logger.debug(
                "Position price updated via FIFO calculator",
                symbol=symbol,
                new_price=float(current_price),
                unrealized_pnl=float(position_summary.unrealized_pnl),
                unrealized_pnl_percent=float(position_summary.unrealized_pnl_percent)
            )
        
        # Update total unrealized P&L
        await self._update_total_unrealized_pnl()
    
    async def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
        price: Optional[Decimal] = None
    ) -> Decimal:
        """
        Close position (partially or fully) using FIFO calculator.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to close (None for full close)
            price: Closing price (current market price if None)
            
        Returns:
            Realized P&L from closing
        """
        position_summary = await self.fifo_calculator.get_position_summary(symbol)
        if not position_summary or position_summary.total_quantity <= 0:
            logger.warning("No position to close", symbol=symbol)
            return Decimal("0")
        
        close_qty = quantity or abs(position_summary.total_quantity)
        close_price = price or position_summary.current_price
        
        if close_qty > abs(position_summary.total_quantity):
            close_qty = abs(position_summary.total_quantity)
        
        # Determine close side
        close_side = OrderSide.SELL if position_summary.total_quantity > 0 else OrderSide.BUY
        
        # Record closing transaction in FIFO calculator
        transaction_id = await self.fifo_calculator.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.SELL if close_side == OrderSide.SELL else TransactionType.BUY,
            side=close_side,
            quantity=close_qty,
            price=close_price,
            timestamp=datetime.now(timezone.utc),
            commission=Decimal("0"),
            metadata={
                "order_id": "position_close",
                "execution_type": "position_close"
            }
        )
        
        # Get updated position to calculate realized P&L
        updated_summary = await self.fifo_calculator.get_position_summary(symbol)
        realized_pnl = (updated_summary.total_realized_pnl - position_summary.total_realized_pnl) if updated_summary else Decimal("0")
        
        logger.info(
            "Position closed using FIFO calculator",
            symbol=symbol,
            quantity=float(close_qty),
            price=float(close_price),
            realized_pnl=float(realized_pnl),
            transaction_id=transaction_id
        )
        
        return realized_pnl
    
    async def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with key metrics using FIFO calculator."""
        total_value = Decimal("0")
        total_cost = Decimal("0")
        total_realized_pnl = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        position_count = 0
        
        positions_data = []
        
        # Get all positions from FIFO calculator
        all_positions = await self.fifo_calculator.get_all_positions()
        
        for symbol, position_summary in all_positions.items():
            if position_summary.total_quantity != 0:
                position_count += 1
                position_value = abs(position_summary.total_quantity) * position_summary.current_price
                position_cost = position_summary.total_cost_basis
                
                total_value += position_value
                total_cost += position_cost
                total_realized_pnl += position_summary.total_realized_pnl
                total_unrealized_pnl += position_summary.unrealized_pnl
                
                positions_data.append({
                    "symbol": symbol,
                    "side": "BUY" if position_summary.total_quantity > 0 else "SELL",
                    "quantity": float(abs(position_summary.total_quantity)),
                    "entry_price": float(position_summary.average_price),
                    "current_price": float(position_summary.current_price),
                    "value": float(position_value),
                    "cost": float(position_cost),
                    "unrealized_pnl": float(position_summary.unrealized_pnl),
                    "pnl_percentage": float(position_summary.unrealized_pnl_percent),
                    "realized_pnl": float(position_summary.total_realized_pnl),
                    "lot_count": position_summary.lot_count,
                    "oldest_lot_date": position_summary.oldest_lot_date.isoformat() if position_summary.oldest_lot_date else None,
                    "short_term_quantity": float(position_summary.holding_periods.get("short_term", Decimal("0"))),
                    "long_term_quantity": float(position_summary.holding_periods.get("long_term", Decimal("0")))
                })
        
        total_pnl = total_realized_pnl + total_unrealized_pnl
        total_pnl_percentage = float((total_pnl / total_cost * 100)) if total_cost > 0 else 0
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "position_count": position_count,
            "total_value": float(total_value),
            "total_cost": float(total_cost),
            "total_pnl": float(total_pnl),
            "total_pnl_percentage": total_pnl_percentage,
            "realized_pnl": float(total_realized_pnl),
            "unrealized_pnl": float(total_unrealized_pnl),
            "positions": positions_data
        }
    
    async def take_snapshot(self) -> Dict:
        """Take a snapshot of current positions."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": {},
            "total_value": Decimal("0"),
            "total_pnl": self._total_realized_pnl + self._total_unrealized_pnl
        }
        
        for symbol, position in self._positions.items():
            if position.quantity > 0:
                position_value = position.quantity * position.current_price
                snapshot["total_value"] += position_value
                
                snapshot["positions"][symbol] = {
                    "side": position.side.value,
                    "quantity": float(position.quantity),
                    "entry_price": float(position.entry_price),
                    "current_price": float(position.current_price),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "pnl_percentage": position.pnl_percentage
                }
        
        snapshot["total_value"] = float(snapshot["total_value"])
        self._position_snapshots.append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(self._position_snapshots) > 1000:
            self._position_snapshots = self._position_snapshots[-1000:]
        
        return snapshot
    
    async def get_trade_history(self, symbol: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get trade history from FIFO calculator, optionally filtered by symbol."""
        if symbol:
            # Get transaction history for specific symbol
            transactions = []
            # Note: FIFO calculator doesn't expose transactions directly for security
            # This would require adding a method to FIFOCalculator if needed
            return {symbol: transactions}
        
        # Return empty for now - would need FIFO calculator enhancement
        return {}
        
    async def get_detailed_pnl_report(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """Get detailed P&L report from FIFO calculator."""
        pnl_report = await self.fifo_calculator.generate_pnl_report(symbol, start_date, end_date)
        
        return {
            "report_id": pnl_report.report_id,
            "symbol": pnl_report.symbol,
            "period_start": pnl_report.period_start.isoformat(),
            "period_end": pnl_report.period_end.isoformat(),
            "total_realized_pnl": float(pnl_report.total_realized_pnl),
            "total_unrealized_pnl": float(pnl_report.total_unrealized_pnl),
            "total_pnl": float(pnl_report.total_pnl),
            "total_commissions": float(pnl_report.total_commissions),
            "total_trades": pnl_report.total_trades,
            "winning_trades": pnl_report.winning_trades,
            "losing_trades": pnl_report.losing_trades,
            "win_rate": float(pnl_report.win_rate),
            "average_win": float(pnl_report.average_win),
            "average_loss": float(pnl_report.average_loss),
            "profit_factor": float(pnl_report.profit_factor),
            "max_consecutive_wins": pnl_report.max_consecutive_wins,
            "max_consecutive_losses": pnl_report.max_consecutive_losses,
            "short_term_gains": float(pnl_report.short_term_gains),
            "long_term_gains": float(pnl_report.long_term_gains)
        }
    
    async def get_tax_lots(self, symbol: str) -> List[Dict]:
        """Get tax lot breakdown for a symbol."""
        tax_lots = await self.fifo_calculator.get_tax_lots(symbol)
        
        return [{
            "lot_id": lot.lot_id,
            "symbol": lot.symbol,
            "quantity": float(lot.quantity),
            "remaining_quantity": float(lot.remaining_quantity),
            "entry_price": float(lot.entry_price),
            "entry_date": lot.entry_date.isoformat(),
            "status": lot.status.value,
            "cost_basis": float(lot.cost_basis),
            "metadata": lot.metadata
        } for lot in tax_lots]
    
    async def calculate_wash_sales(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Calculate potential wash sales for tax optimization."""
        return await self.fifo_calculator.calculate_wash_sales(symbol, start_date, end_date)
    
    # Private methods
    
    async def _update_total_unrealized_pnl(self) -> None:
        """Update total unrealized P&L across all positions using FIFO calculator."""
        total = Decimal("0")
        all_positions = await self.fifo_calculator.get_all_positions()
        
        for position_summary in all_positions.values():
            if position_summary.total_quantity != 0:
                total += position_summary.unrealized_pnl
        
        self._total_unrealized_pnl = total
    
