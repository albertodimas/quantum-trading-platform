"""
Portfolio Simulator for Backtesting

Simulates portfolio state, positions, cash management, and P&L tracking
during backtesting operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.observability import get_logger
from ..positions.position_tracker import Position

logger = get_logger(__name__)

@dataclass
class SimulatedPosition:
    """Represents a position in the simulated portfolio"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price    
    @property
    def total_pnl(self) -> float:
        """Total P&L including realized and unrealized"""
        return self.realized_pnl + self.unrealized_pnl
        
    def update_price(self, new_price: float):
        """Update position with new market price"""
        self.current_price = new_price
        if self.quantity != 0:
            self.unrealized_pnl = (new_price - self.average_price) * self.quantity

@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time"""
    timestamp: datetime
    cash: float
    positions: Dict[str, SimulatedPosition]
    total_value: float
    equity: float
    margin_used: float = 0.0
    free_margin: float = 0.0
    
class SimulatedPortfolio:
    """Manages portfolio state during backtesting"""
    
    def __init__(
        self,
        initial_capital: float,
        commission: float = 0.001,
        allow_shorting: bool = True,
        margin_ratio: float = 0.5,
        base_currency: str = "USD"
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission
        self.allow_shorting = allow_shorting
        self.margin_ratio = margin_ratio
        self.base_currency = base_currency        
        # Portfolio state
        self.positions: Dict[str, SimulatedPosition] = {}
        self.closed_positions: List[SimulatedPosition] = []
        
        # Trade history
        self.trades: List[Dict[str, Any]] = []
        self.orders: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Risk metrics
        self.max_positions = 0
        self.max_exposure = 0.0
        self.total_commission_paid = 0.0
        
        self.logger = logger
        
    def get_position(self, symbol: str) -> Optional[SimulatedPosition]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
        
    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol"""
        return symbol in self.positions and self.positions[symbol].quantity != 0
        
    def get_cash_available(self) -> float:
        """Get available cash considering margin"""
        if not self.allow_shorting:
            return self.cash            
        # Calculate margin used
        margin_used = 0.0
        for position in self.positions.values():
            if position.quantity < 0:  # Short position
                margin_used += abs(position.market_value) * self.margin_ratio
                
        return self.cash - margin_used
        
    def can_afford_order(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if portfolio can afford the order"""
        order_value = abs(quantity * price)
        commission = order_value * self.commission_rate
        
        if quantity > 0:  # Buy order
            required_cash = order_value + commission
            return self.get_cash_available() >= required_cash
        else:  # Sell order
            if not self.allow_shorting and not self.has_position(symbol):
                return False
                
            if self.allow_shorting:
                # Check margin requirements
                required_margin = order_value * self.margin_ratio
                return self.get_cash_available() >= required_margin + commission
                
            # For closing positions, just need commission
            return self.cash >= commission            
    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        order_id: Optional[str] = None
    ) -> bool:
        """Execute a trade and update portfolio"""
        # Calculate costs
        trade_value = quantity * price
        commission = abs(trade_value) * self.commission_rate
        
        # Check if we can afford it
        if not self.can_afford_order(symbol, quantity, price):
            self.logger.warning(
                f"Insufficient funds for trade: {symbol} {quantity}@{price}"
            )
            return False
            
        # Update cash
        self.cash -= (trade_value + commission)
        self.total_commission_paid += commission
        
        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            old_quantity = position.quantity
            
            # FIFO calculation for realized P&L
            if (old_quantity > 0 and quantity < 0) or (old_quantity < 0 and quantity > 0):
                # Closing or reducing position
                closed_quantity = min(abs(old_quantity), abs(quantity))
                if old_quantity > 0:
                    realized_pnl = closed_quantity * (price - position.average_price)
                else:
                    realized_pnl = closed_quantity * (position.average_price - price)                    
                position.realized_pnl += realized_pnl
                
            # Update position
            new_quantity = old_quantity + quantity
            
            if new_quantity == 0:
                # Position closed
                self.closed_positions.append(position)
                del self.positions[symbol]
            else:
                # Update average price
                if (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                    # Adding to position
                    total_cost = (old_quantity * position.average_price) + trade_value
                    position.average_price = total_cost / new_quantity
                else:
                    # Reversed position
                    position.average_price = price
                    
                position.quantity = new_quantity
                position.commission_paid += commission
                position.last_updated = timestamp
        else:
            # New position
            self.positions[symbol] = SimulatedPosition(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price,
                commission_paid=commission,
                opened_at=timestamp,
                last_updated=timestamp
            )            
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'commission': commission,
            'order_id': order_id,
            'cash_after': self.cash,
            'portfolio_value': self.get_total_value()
        })
        
        # Update max metrics
        self.max_positions = max(self.max_positions, len(self.positions))
        total_exposure = sum(abs(p.market_value) for p in self.positions.values())
        self.max_exposure = max(self.max_exposure, total_exposure)
        
        return True
        
    def update_market_prices(self, price_data: Dict[str, float], timestamp: datetime):
        """Update all positions with latest market prices"""
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
                
        # Record equity curve point
        total_value = self.get_total_value()
        self.equity_curve.append((timestamp, total_value))        
    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
        
    def get_equity(self) -> float:
        """Calculate portfolio equity (considers unrealized P&L)"""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(p.realized_pnl for p in self.positions.values())
        return self.initial_capital + total_realized + total_unrealized
        
    def take_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Create a snapshot of current portfolio state"""
        # Deep copy positions
        positions_copy = {
            symbol: SimulatedPosition(
                symbol=pos.symbol,
                quantity=pos.quantity,
                average_price=pos.average_price,
                current_price=pos.current_price,
                realized_pnl=pos.realized_pnl,
                unrealized_pnl=pos.unrealized_pnl,
                commission_paid=pos.commission_paid,
                opened_at=pos.opened_at,
                last_updated=pos.last_updated
            )
            for symbol, pos in self.positions.items()
        }        
        total_value = self.get_total_value()
        equity = self.get_equity()
        
        # Calculate margin
        margin_used = 0.0
        for pos in self.positions.values():
            if pos.quantity < 0:
                margin_used += abs(pos.market_value) * self.margin_ratio
                
        free_margin = self.cash - margin_used
        
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=positions_copy,
            total_value=total_value,
            equity=equity,
            margin_used=margin_used,
            free_margin=free_margin
        )
        
    def record_snapshot(self, timestamp: datetime):
        """Record portfolio snapshot to history"""
        snapshot = self.take_snapshot(timestamp)
        self.portfolio_history.append(snapshot)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        if not self.equity_curve:
            return {}
            
        equity_values = [v for _, v in self.equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()        
        total_pnl = self.get_total_value() - self.initial_capital
        total_return = total_pnl / self.initial_capital
        
        # Calculate win/loss statistics
        winning_trades = [t for t in self.trades if t['value'] < 0 and t['quantity'] < 0]  # Sells at profit
        losing_trades = [t for t in self.trades if t['value'] < 0 and t['quantity'] > 0]  # Buys at loss
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.get_total_value(),
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_commission': self.total_commission_paid,
            'max_positions': self.max_positions,
            'max_exposure': self.max_exposure,
            'average_trade_size': np.mean([abs(t['value']) for t in self.trades]) if self.trades else 0,
            'largest_trade': max([abs(t['value']) for t in self.trades]) if self.trades else 0,
        }
        
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades history to DataFrame"""
        if not self.trades:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.trades)
        df.set_index('timestamp', inplace=True)
        return df        
    def get_positions_dataframe(self) -> pd.DataFrame:
        """Get all positions (open and closed) as DataFrame"""
        all_positions = []
        
        # Current positions
        for pos in self.positions.values():
            all_positions.append({
                'symbol': pos.symbol,
                'status': 'open',
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'realized_pnl': pos.realized_pnl,
                'unrealized_pnl': pos.unrealized_pnl,
                'total_pnl': pos.total_pnl,
                'commission_paid': pos.commission_paid,
                'opened_at': pos.opened_at,
                'closed_at': None
            })
            
        # Closed positions
        for pos in self.closed_positions:
            all_positions.append({
                'symbol': pos.symbol,
                'status': 'closed',
                'quantity': 0,
                'average_price': pos.average_price,
                'current_price': pos.current_price,
                'market_value': 0,
                'realized_pnl': pos.realized_pnl,
                'unrealized_pnl': 0,
                'total_pnl': pos.realized_pnl,
                'commission_paid': pos.commission_paid,
                'opened_at': pos.opened_at,
                'closed_at': pos.last_updated
            })            
        if not all_positions:
            return pd.DataFrame()
            
        return pd.DataFrame(all_positions)
        
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'total_value'])
        df.set_index('timestamp', inplace=True)
        
        # Add additional metrics
        df['returns'] = df['total_value'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        df['drawdown'] = df['total_value'] / df['total_value'].cummax() - 1
        
        return df
        
    def get_portfolio_values(self) -> pd.DataFrame:
        """Get detailed portfolio values over time"""
        if not self.portfolio_history:
            return pd.DataFrame()
            
        data = []
        for snapshot in self.portfolio_history:
            row = {
                'timestamp': snapshot.timestamp,
                'cash': snapshot.cash,
                'positions_value': snapshot.total_value - snapshot.cash,
                'total_value': snapshot.total_value,
                'equity': snapshot.equity,
                'margin_used': snapshot.margin_used,
                'free_margin': snapshot.free_margin,
                'num_positions': len(snapshot.positions)
            }            
            # Add position details
            for symbol, pos in snapshot.positions.items():
                row[f'{symbol}_quantity'] = pos.quantity
                row[f'{symbol}_value'] = pos.market_value
                row[f'{symbol}_pnl'] = pos.total_pnl
                
            data.append(row)
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

class PortfolioSimulator:
    """
    High-level portfolio simulator that manages the simulated portfolio
    """
    
    def __init__(
        self,
        initial_capital: float,
        commission: float = 0.001,
        allow_shorting: bool = True,
        margin_ratio: float = 0.5
    ):
        self.portfolio = SimulatedPortfolio(
            initial_capital=initial_capital,
            commission=commission,
            allow_shorting=allow_shorting,
            margin_ratio=margin_ratio
        )
        self.logger = logger        
    async def update_market_data(self, market_data: Dict[str, Any]):
        """Update portfolio with latest market data"""
        # Extract prices from market data
        prices = {}
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'close' in data:
                prices[symbol] = data['close']
            elif isinstance(data, (int, float)):
                prices[symbol] = data
                
        timestamp = datetime.now()  # Or extract from market data
        self.portfolio.update_market_prices(prices, timestamp)
        
    async def process_fill(self, fill: Dict[str, Any]) -> bool:
        """Process an order fill"""
        return self.portfolio.execute_trade(
            symbol=fill['symbol'],
            quantity=fill['quantity'],
            price=fill['price'],
            timestamp=fill['timestamp'],
            order_id=fill.get('order_id')
        )
        
    async def record_snapshot(self, timestamp: datetime):
        """Record portfolio snapshot"""
        self.portfolio.record_snapshot(timestamp)
        
    def get_portfolio_history(self) -> List[PortfolioSnapshot]:
        """Get portfolio history"""
        return self.portfolio.portfolio_history        
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        return self.portfolio.get_trades_dataframe()
        
    def get_positions_dataframe(self) -> pd.DataFrame:
        """Get positions as DataFrame"""
        return self.portfolio.get_positions_dataframe()
        
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve"""
        return self.portfolio.get_equity_curve()
        
    def get_portfolio_values(self) -> pd.DataFrame:
        """Get portfolio values over time"""
        return self.portfolio.get_portfolio_values()
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.portfolio.get_performance_summary()