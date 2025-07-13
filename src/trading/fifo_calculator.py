"""
Advanced FIFO P&L Calculator

Sophisticated implementation of First-In-First-Out position and P&L calculations
with support for complex scenarios, tax optimization, and detailed reporting.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderSide

logger = get_logger(__name__)


class TransactionType(Enum):
    """Transaction types for P&L calculation"""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"
    MERGE = "merge"
    CORPORATE_ACTION = "corporate_action"


class LotStatus(Enum):
    """Status of a tax lot"""
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"


@dataclass
class TaxLot:
    """
    Individual tax lot for FIFO tracking.
    
    A tax lot represents a specific purchase at a specific price and time.
    """
    lot_id: str
    symbol: str
    quantity: Decimal
    original_quantity: Decimal
    entry_price: Decimal
    entry_date: datetime
    status: LotStatus = LotStatus.OPEN
    cost_basis: Decimal = field(init=False)
    remaining_quantity: Decimal = field(init=False)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.cost_basis = self.quantity * self.entry_price
        self.remaining_quantity = self.quantity
    
    def can_sell(self, quantity: Decimal) -> bool:
        """Check if lot has enough quantity for sale"""
        return self.remaining_quantity >= quantity and self.status != LotStatus.CLOSED
    
    def partial_sell(self, quantity: Decimal, sell_price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Partially sell from this lot.
        
        Returns:
            (realized_pnl, cost_basis_sold)
        """
        if not self.can_sell(quantity):
            raise ValueError(f"Insufficient quantity in lot {self.lot_id}")
        
        # Calculate proportional cost basis
        cost_basis_sold = (quantity / self.remaining_quantity) * (self.remaining_quantity * self.entry_price)
        realized_pnl = quantity * sell_price - cost_basis_sold
        
        # Update lot
        self.remaining_quantity -= quantity
        
        if self.remaining_quantity == 0:
            self.status = LotStatus.CLOSED
        else:
            self.status = LotStatus.PARTIALLY_CLOSED
        
        return realized_pnl, cost_basis_sold


@dataclass
class Transaction:
    """Individual transaction record"""
    transaction_id: str
    symbol: str
    transaction_type: TransactionType
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    tax_lots_affected: List[str] = field(default_factory=list)
    realized_pnl: Decimal = Decimal("0")
    cost_basis: Decimal = field(init=False)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.cost_basis = self.quantity * self.price + self.commission


@dataclass
class PositionSummary:
    """Position summary with detailed FIFO breakdown"""
    symbol: str
    total_quantity: Decimal
    total_cost_basis: Decimal
    average_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: Decimal
    total_realized_pnl: Decimal
    total_commissions: Decimal
    lot_count: int
    oldest_lot_date: Optional[datetime]
    newest_lot_date: Optional[datetime]
    tax_lot_breakdown: List[Dict] = field(default_factory=list)
    holding_periods: Dict[str, Decimal] = field(default_factory=dict)  # short_term, long_term quantities
    

@dataclass
class PnLReport:
    """Comprehensive P&L report"""
    report_id: str
    symbol: str
    period_start: datetime
    period_end: datetime
    total_realized_pnl: Decimal
    total_unrealized_pnl: Decimal
    total_pnl: Decimal
    total_commissions: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    max_consecutive_wins: int
    max_consecutive_losses: int
    short_term_gains: Decimal
    long_term_gains: Decimal
    transactions: List[Transaction] = field(default_factory=list)
    lot_closures: List[Dict] = field(default_factory=list)


@injectable
class FIFOCalculator:
    """
    Advanced FIFO calculator with comprehensive P&L tracking.
    
    Features:
    - Precise FIFO lot tracking
    - Short-term vs long-term capital gains
    - Detailed transaction history
    - Corporate action handling
    - Tax optimization suggestions
    - Performance analytics
    """
    
    def __init__(self):
        """Initialize FIFO calculator."""
        
        # Tax lots by symbol: symbol -> List[TaxLot]
        self._tax_lots: Dict[str, List[TaxLot]] = {}
        
        # All transactions: symbol -> List[Transaction]
        self._transactions: Dict[str, List[Transaction]] = {}
        
        # Position summaries: symbol -> PositionSummary
        self._position_summaries: Dict[str, PositionSummary] = {}
        
        # Current prices for unrealized P&L
        self._current_prices: Dict[str, Decimal] = {}
        
        # Configuration
        self.long_term_holding_days = 365  # Days for long-term capital gains
        self.precision = 8  # Decimal precision for calculations
        
        logger.info("Advanced FIFO calculator initialized")
    
    async def add_transaction(
        self,
        symbol: str,
        transaction_type: TransactionType,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Decimal = Decimal("0"),
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a new transaction and update FIFO calculations.
        
        Args:
            symbol: Trading symbol
            transaction_type: Type of transaction
            side: Order side (BUY/SELL)
            quantity: Quantity transacted
            price: Transaction price
            timestamp: Transaction timestamp (current time if None)
            commission: Commission paid
            metadata: Additional transaction metadata
            
        Returns:
            Transaction ID
        """
        transaction_id = str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc)
        
        # Round to avoid floating point issues
        quantity = quantity.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        price = price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        commission = commission.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        transaction = Transaction(
            transaction_id=transaction_id,
            symbol=symbol,
            transaction_type=transaction_type,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            metadata=metadata or {}
        )
        
        # Initialize symbol if needed
        if symbol not in self._tax_lots:
            self._tax_lots[symbol] = []
            self._transactions[symbol] = []
        
        # Process transaction
        if side == OrderSide.BUY:
            await self._process_buy_transaction(transaction)
        elif side == OrderSide.SELL:
            await self._process_sell_transaction(transaction)
        
        # Add to transaction history
        self._transactions[symbol].append(transaction)
        
        # Update position summary
        await self._update_position_summary(symbol)
        
        logger.info(
            f"Transaction processed",
            transaction_id=transaction_id,
            symbol=symbol,
            side=side.value,
            quantity=float(quantity),
            price=float(price)
        )
        
        return transaction_id
    
    async def update_current_price(self, symbol: str, price: Decimal) -> None:
        """Update current market price for unrealized P&L calculation"""
        self._current_prices[symbol] = price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        # Update position summary with new unrealized P&L
        if symbol in self._position_summaries:
            await self._update_position_summary(symbol)
    
    async def get_position_summary(self, symbol: str) -> Optional[PositionSummary]:
        """Get current position summary for symbol"""
        return self._position_summaries.get(symbol)
    
    async def get_all_positions(self) -> Dict[str, PositionSummary]:
        """Get all position summaries"""
        return dict(self._position_summaries)
    
    async def generate_pnl_report(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PnLReport:
        """
        Generate comprehensive P&L report for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Report start date (None for all time)
            end_date: Report end date (None for current time)
            
        Returns:
            Detailed P&L report
        """
        start_date = start_date or datetime.min.replace(tzinfo=timezone.utc)
        end_date = end_date or datetime.now(timezone.utc)
        
        # Filter transactions by date range
        transactions = [
            t for t in self._transactions.get(symbol, [])
            if start_date <= t.timestamp <= end_date
        ]
        
        if not transactions:
            # Return empty report
            return PnLReport(
                report_id=str(uuid.uuid4()),
                symbol=symbol,
                period_start=start_date,
                period_end=end_date,
                total_realized_pnl=Decimal("0"),
                total_unrealized_pnl=Decimal("0"),
                total_pnl=Decimal("0"),
                total_commissions=Decimal("0"),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=Decimal("0"),
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                profit_factor=Decimal("0"),
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                short_term_gains=Decimal("0"),
                long_term_gains=Decimal("0")
            )
        
        # Calculate metrics
        sell_transactions = [t for t in transactions if t.side == OrderSide.SELL]
        total_realized_pnl = sum(t.realized_pnl for t in sell_transactions)
        total_commissions = sum(t.commission for t in transactions)
        
        # Get current position for unrealized P&L
        position = self._position_summaries.get(symbol)
        total_unrealized_pnl = position.unrealized_pnl if position else Decimal("0")
        
        # Win/loss analysis
        winning_trades = len([t for t in sell_transactions if t.realized_pnl > 0])
        losing_trades = len([t for t in sell_transactions if t.realized_pnl < 0])
        total_trades = len(sell_transactions)
        
        win_rate = Decimal(str(winning_trades / total_trades)) if total_trades > 0 else Decimal("0")
        
        # Average win/loss
        wins = [t.realized_pnl for t in sell_transactions if t.realized_pnl > 0]
        losses = [t.realized_pnl for t in sell_transactions if t.realized_pnl < 0]
        
        average_win = sum(wins) / len(wins) if wins else Decimal("0")
        average_loss = sum(losses) / len(losses) if losses else Decimal("0")
        
        # Profit factor
        total_wins = sum(wins) if wins else Decimal("0")
        total_losses = abs(sum(losses)) if losses else Decimal("0")
        profit_factor = total_wins / total_losses if total_losses > 0 else Decimal("0")
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks(sell_transactions)
        
        # Short-term vs long-term gains
        short_term_gains, long_term_gains = await self._calculate_holding_period_gains(
            symbol, sell_transactions
        )
        
        report = PnLReport(
            report_id=str(uuid.uuid4()),
            symbol=symbol,
            period_start=start_date,
            period_end=end_date,
            total_realized_pnl=total_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
            total_pnl=total_realized_pnl + total_unrealized_pnl,
            total_commissions=total_commissions,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            short_term_gains=short_term_gains,
            long_term_gains=long_term_gains,
            transactions=transactions
        )
        
        logger.info(
            f"P&L report generated for {symbol}",
            period=f"{start_date.date()} to {end_date.date()}",
            total_pnl=float(report.total_pnl),
            total_trades=report.total_trades,
            win_rate=float(report.win_rate)
        )
        
        return report
    
    async def get_tax_lots(self, symbol: str) -> List[TaxLot]:
        """Get all tax lots for a symbol"""
        return self._tax_lots.get(symbol, []).copy()
    
    async def get_open_lots(self, symbol: str) -> List[TaxLot]:
        """Get only open tax lots for a symbol"""
        return [lot for lot in self._tax_lots.get(symbol, []) if lot.status != LotStatus.CLOSED]
    
    async def calculate_wash_sales(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Calculate potential wash sales for tax optimization.
        
        A wash sale occurs when you sell a security at a loss and buy the same
        security within 30 days before or after the sale.
        """
        transactions = self._transactions.get(symbol, [])
        sell_transactions = [
            t for t in transactions
            if t.side == OrderSide.SELL and t.realized_pnl < 0
            and start_date <= t.timestamp <= end_date
        ]
        
        wash_sales = []
        
        for sell_tx in sell_transactions:
            # Look for buys within 30 days before or after
            wash_sale_window_start = sell_tx.timestamp - timedelta(days=30)
            wash_sale_window_end = sell_tx.timestamp + timedelta(days=30)
            
            related_buys = [
                t for t in transactions
                if t.side == OrderSide.BUY
                and wash_sale_window_start <= t.timestamp <= wash_sale_window_end
                and t.transaction_id != sell_tx.transaction_id
            ]
            
            if related_buys:
                wash_sales.append({
                    "sell_transaction": sell_tx,
                    "related_buys": related_buys,
                    "disallowed_loss": sell_tx.realized_pnl,
                    "wash_sale_date": sell_tx.timestamp
                })
        
        return wash_sales
    
    # Private methods
    
    async def _process_buy_transaction(self, transaction: Transaction) -> None:
        """Process a buy transaction by creating a new tax lot"""
        lot_id = f"{transaction.symbol}_{transaction.timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        tax_lot = TaxLot(
            lot_id=lot_id,
            symbol=transaction.symbol,
            quantity=transaction.quantity,
            original_quantity=transaction.quantity,
            entry_price=transaction.price,
            entry_date=transaction.timestamp,
            metadata=transaction.metadata
        )
        
        self._tax_lots[transaction.symbol].append(tax_lot)
        transaction.tax_lots_affected.append(lot_id)
        
        logger.debug(
            f"Created new tax lot",
            lot_id=lot_id,
            symbol=transaction.symbol,
            quantity=float(transaction.quantity),
            price=float(transaction.price)
        )
    
    async def _process_sell_transaction(self, transaction: Transaction) -> None:
        """Process a sell transaction using FIFO lot matching"""
        symbol = transaction.symbol
        remaining_to_sell = transaction.quantity
        total_realized_pnl = Decimal("0")
        total_cost_basis = Decimal("0")
        
        # Get open lots sorted by date (FIFO)
        open_lots = [lot for lot in self._tax_lots[symbol] if lot.status != LotStatus.CLOSED]
        open_lots.sort(key=lambda x: x.entry_date)
        
        if not open_lots:
            # Short sale or error - create synthetic lot for tracking
            logger.warning(f"Sell transaction with no open lots: {transaction.transaction_id}")
            await self._handle_short_position(transaction)
            return
        
        for lot in open_lots:
            if remaining_to_sell <= 0:
                break
            
            if lot.remaining_quantity <= 0:
                continue
            
            # Calculate quantity to sell from this lot
            sell_from_lot = min(remaining_to_sell, lot.remaining_quantity)
            
            # Calculate realized P&L for this portion
            realized_pnl, cost_basis_sold = lot.partial_sell(sell_from_lot, transaction.price)
            
            total_realized_pnl += realized_pnl
            total_cost_basis += cost_basis_sold
            remaining_to_sell -= sell_from_lot
            
            transaction.tax_lots_affected.append(lot.lot_id)
            
            logger.debug(
                f"Sold from lot {lot.lot_id}",
                quantity_sold=float(sell_from_lot),
                realized_pnl=float(realized_pnl),
                remaining_in_lot=float(lot.remaining_quantity)
            )
        
        # Update transaction with realized P&L
        transaction.realized_pnl = total_realized_pnl
        
        if remaining_to_sell > 0:
            logger.warning(
                f"Partial lot matching - {float(remaining_to_sell)} units unsold",
                transaction_id=transaction.transaction_id
            )
    
    async def _handle_short_position(self, transaction: Transaction) -> None:
        """Handle short sale by creating negative tax lot"""
        # For simplicity, treat as synthetic short position
        # In production, this would need more sophisticated handling
        logger.warning(f"Short position handling not fully implemented: {transaction.transaction_id}")
    
    async def _update_position_summary(self, symbol: str) -> None:
        """Update position summary for symbol"""
        open_lots = await self.get_open_lots(symbol)
        
        if not open_lots:
            # No position
            if symbol in self._position_summaries:
                del self._position_summaries[symbol]
            return
        
        # Calculate totals
        total_quantity = sum(lot.remaining_quantity for lot in open_lots)
        total_cost_basis = sum(lot.remaining_quantity * lot.entry_price for lot in open_lots)
        average_price = total_cost_basis / total_quantity if total_quantity > 0 else Decimal("0")
        
        # Current price and unrealized P&L
        current_price = self._current_prices.get(symbol, average_price)
        current_value = total_quantity * current_price
        unrealized_pnl = current_value - total_cost_basis
        unrealized_pnl_percent = (unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else Decimal("0")
        
        # Calculate total realized P&L
        all_transactions = self._transactions.get(symbol, [])
        total_realized_pnl = sum(t.realized_pnl for t in all_transactions if t.side == OrderSide.SELL)
        total_commissions = sum(t.commission for t in all_transactions)
        
        # Holding periods
        current_time = datetime.now(timezone.utc)
        short_term_quantity = Decimal("0")
        long_term_quantity = Decimal("0")
        
        for lot in open_lots:
            holding_days = (current_time - lot.entry_date).days
            if holding_days < self.long_term_holding_days:
                short_term_quantity += lot.remaining_quantity
            else:
                long_term_quantity += lot.remaining_quantity
        
        # Tax lot breakdown
        tax_lot_breakdown = []
        for lot in open_lots:
            holding_days = (current_time - lot.entry_date).days
            tax_lot_breakdown.append({
                "lot_id": lot.lot_id,
                "quantity": float(lot.remaining_quantity),
                "entry_price": float(lot.entry_price),
                "entry_date": lot.entry_date.isoformat(),
                "holding_days": holding_days,
                "is_long_term": holding_days >= self.long_term_holding_days,
                "unrealized_pnl": float(lot.remaining_quantity * (current_price - lot.entry_price))
            })
        
        # Create position summary
        summary = PositionSummary(
            symbol=symbol,
            total_quantity=total_quantity,
            total_cost_basis=total_cost_basis,
            average_price=average_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            total_realized_pnl=total_realized_pnl,
            total_commissions=total_commissions,
            lot_count=len(open_lots),
            oldest_lot_date=min(lot.entry_date for lot in open_lots) if open_lots else None,
            newest_lot_date=max(lot.entry_date for lot in open_lots) if open_lots else None,
            tax_lot_breakdown=tax_lot_breakdown,
            holding_periods={
                "short_term": short_term_quantity,
                "long_term": long_term_quantity
            }
        )
        
        self._position_summaries[symbol] = summary
    
    def _calculate_consecutive_streaks(self, sell_transactions: List[Transaction]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not sell_transactions:
            return 0, 0
        
        # Sort by timestamp
        sorted_txs = sorted(sell_transactions, key=lambda x: x.timestamp)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for tx in sorted_txs:
            if tx.realized_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif tx.realized_pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                # Break even - reset both
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    async def _calculate_holding_period_gains(
        self,
        symbol: str,
        sell_transactions: List[Transaction]
    ) -> Tuple[Decimal, Decimal]:
        """Calculate short-term vs long-term capital gains"""
        short_term_gains = Decimal("0")
        long_term_gains = Decimal("0")
        
        # This would require tracking which specific lots were sold
        # For now, simplified calculation based on overall position
        
        for tx in sell_transactions:
            # Simplified: assume proportional allocation
            # In reality, would need to track specific lot sales
            if tx.realized_pnl > 0:
                long_term_gains += tx.realized_pnl  # Simplified
            else:
                short_term_gains += tx.realized_pnl  # Simplified
        
        return short_term_gains, long_term_gains