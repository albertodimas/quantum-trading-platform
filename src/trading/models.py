"""
Trading data models and enums.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    """Order side enumeration."""
    
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class Signal(BaseModel):
    """Trading signal from strategy or AI agent."""
    
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    side: OrderSide = Field(..., description="Buy or sell signal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    entry_price: Decimal = Field(..., description="Suggested entry price")
    stop_loss: Optional[Decimal] = Field(None, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, description="Take profit price")
    quantity: Optional[Decimal] = Field(None, description="Suggested quantity")
    strategy: str = Field(..., description="Strategy that generated the signal")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict, description="Additional signal data")
    
    @field_validator("stop_loss", "take_profit")
    @classmethod
    def validate_risk_levels(cls, v: Optional[Decimal], values: dict) -> Optional[Decimal]:
        """Validate stop loss and take profit levels."""
        if v is None:
            return v
            
        entry_price = values.get("entry_price")
        side = values.get("side")
        
        if entry_price and side:
            if side == OrderSide.BUY:
                if "stop_loss" in values and v < entry_price:
                    raise ValueError("Stop loss must be below entry for buy orders")
                if "take_profit" in values and v > entry_price:
                    raise ValueError("Take profit must be above entry for buy orders")
            else:  # SELL
                if "stop_loss" in values and v > entry_price:
                    raise ValueError("Stop loss must be above entry for sell orders")
                if "take_profit" in values and v < entry_price:
                    raise ValueError("Take profit must be below entry for sell orders")
        
        return v


class MarketData(BaseModel):
    """Market data snapshot."""
    
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    open_24h: Decimal
    change_24h: Decimal
    change_percent_24h: float
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = (self.bid + self.ask) / 2
        return float((self.spread / mid) * 100) if mid > 0 else 0.0


class ExecutionReport(BaseModel):
    """Order execution report."""
    
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    price: Decimal
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal
    average_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")
    commission_asset: Optional[str] = None
    timestamp: datetime
    exchange_order_id: Optional[str] = None
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return float((self.filled_quantity / self.quantity) * 100)


class PositionUpdate(BaseModel):
    """Position update event."""
    
    symbol: str
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    timestamp: datetime
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return float((self.unrealized_pnl / (self.entry_price * self.quantity)) * 100)