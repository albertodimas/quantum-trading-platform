"""
Exchange Interface

Abstract base class and data models for exchange integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field

from ..core.observability import get_logger

logger = get_logger(__name__)


# Enums for order types and statuses
class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    LIMIT_MAKER = "limit_maker"


class OrderSide(str, Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force for orders"""
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTX = "gtx"  # Good Till Crossing
    GTD = "gtd"  # Good Till Date


# Data models
class Balance(BaseModel):
    """Account balance for a specific asset"""
    asset: str
    free: Decimal = Field(decimal_places=8)
    locked: Decimal = Field(decimal_places=8)
    total: Decimal = Field(decimal_places=8)
    
    class Config:
        arbitrary_types_allowed = True


class Ticker(BaseModel):
    """Market ticker data"""
    symbol: str
    bid_price: Decimal = Field(decimal_places=8)
    bid_qty: Decimal = Field(decimal_places=8)
    ask_price: Decimal = Field(decimal_places=8)
    ask_qty: Decimal = Field(decimal_places=8)
    last_price: Decimal = Field(decimal_places=8)
    volume_24h: Decimal = Field(decimal_places=8)
    high_24h: Decimal = Field(decimal_places=8)
    low_24h: Decimal = Field(decimal_places=8)
    open_24h: Decimal = Field(decimal_places=8)
    timestamp: datetime
    
    class Config:
        arbitrary_types_allowed = True


class OrderBook(BaseModel):
    """Order book data"""
    symbol: str
    bids: List[List[Decimal]]  # [[price, quantity], ...]
    asks: List[List[Decimal]]  # [[price, quantity], ...]
    timestamp: datetime
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Get bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class Order(BaseModel):
    """Order data model"""
    id: str  # Exchange order ID
    client_order_id: Optional[str] = None
    symbol: str
    type: OrderType
    side: OrderSide
    status: OrderStatus
    price: Optional[Decimal] = Field(None, decimal_places=8)
    quantity: Decimal = Field(decimal_places=8)
    executed_qty: Decimal = Field(default=Decimal("0"), decimal_places=8)
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: datetime
    updated_at: datetime
    fills: List[Dict[str, Any]] = []
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def remaining_qty(self) -> Decimal:
        """Get remaining quantity"""
        return self.quantity - self.executed_qty


class Trade(BaseModel):
    """Trade/fill data model"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal = Field(decimal_places=8)
    quantity: Decimal = Field(decimal_places=8)
    fee: Decimal = Field(decimal_places=8)
    fee_asset: str
    timestamp: datetime
    is_maker: bool
    
    class Config:
        arbitrary_types_allowed = True


class Position(BaseModel):
    """Position data model"""
    symbol: str
    side: OrderSide
    quantity: Decimal = Field(decimal_places=8)
    entry_price: Decimal = Field(decimal_places=8)
    mark_price: Decimal = Field(decimal_places=8)
    unrealized_pnl: Decimal = Field(decimal_places=8)
    realized_pnl: Decimal = Field(decimal_places=8)
    margin_type: str = "cross"  # cross or isolated
    leverage: int = 1
    
    class Config:
        arbitrary_types_allowed = True


class MarketData(BaseModel):
    """Market data container"""
    symbol: str
    ticker: Optional[Ticker] = None
    orderbook: Optional[OrderBook] = None
    trades: List[Trade] = []
    timestamp: datetime
    
    class Config:
        arbitrary_types_allowed = True


class ExchangeInterface(ABC):
    """Abstract base class for exchange implementations"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self._callbacks: Dict[str, List[Callable]] = {}
        logger.info(f"Initializing {name} exchange interface")
    
    # Connection management
    @abstractmethod
    async def connect(self) -> None:
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def is_alive(self) -> bool:
        """Check if connection is alive"""
        pass
    
    # Account information
    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions"""
        pass
    
    # Market data
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data for symbol"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book for symbol"""
        pass
    
    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for symbol"""
        pass
    
    # Order management
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> Order:
        """Create new order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel existing order"""
        pass
    
    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order details"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_order_history(
        self, 
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history"""
        pass
    
    # WebSocket streams
    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker updates"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to order book updates"""
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade updates"""
        pass
    
    @abstractmethod
    async def subscribe_orders(self, callback: Callable) -> None:
        """Subscribe to order updates"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from stream"""
        pass
    
    # Utility methods
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register event callback"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")
    
    def emit_event(self, event: str, data: Any) -> None:
        """Emit event to registered callbacks"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for event {event}: {str(e)}")
    
    async def ping(self) -> float:
        """Ping exchange and return latency in ms"""
        start = datetime.now()
        await self.is_alive()
        latency = (datetime.now() - start).total_seconds() * 1000
        return latency