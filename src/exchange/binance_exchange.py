"""
Binance Exchange Implementation

Real WebSocket and REST API integration for Binance exchange.
"""

import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlencode
import aiohttp
import websockets
from websockets.exceptions import WebSocketException

from .exchange_interface import (
    ExchangeInterface,
    OrderType, OrderSide, OrderStatus, TimeInForce,
    Order, Trade, Position, MarketData, OrderBook, Ticker, Balance
)
from ..core.observability import get_logger
from ..core.architecture import CircuitBreaker, RateLimiter
from ..core.config import settings

logger = get_logger(__name__)


class BinanceExchange(ExchangeInterface):
    """Binance exchange implementation with WebSocket support"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        super().__init__("Binance")
        
        # API credentials
        self.api_key = api_key or settings.binance_api_key or ""
        self.api_secret = api_secret or settings.binance_api_secret or ""
        self.testnet = testnet
        
        # URLs based on environment
        if testnet:
            self.rest_url = "https://testnet.binance.vision/api/v3"
            self.ws_url = "wss://testnet.binance.vision/ws"
            self.ws_stream_url = "wss://testnet.binance.vision/stream"
        else:
            self.rest_url = "https://api.binance.com/api/v3"
            self.ws_url = "wss://stream.binance.com:9443/ws"
            self.ws_stream_url = "wss://stream.binance.com:9443/stream"
        
        # Connection management
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.user_stream_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.listen_key: Optional[str] = None
        self.subscriptions: Dict[str, Set[str]] = {}
        self.subscription_id_counter = 0
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Circuit breaker and rate limiter
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=(aiohttp.ClientError, WebSocketException)
        )
        self.rate_limiter = RateLimiter(
            rate=1200,  # Binance limit: 1200 requests per minute
            per=60
        )
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info(f"Initialized Binance exchange (testnet={testnet})")
    
    # Connection management
    async def connect(self) -> None:
        """Connect to Binance WebSocket and REST API"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Start user data stream
            await self._start_user_stream()
            
            # Connect to market data WebSocket
            self.ws_connection = await websockets.connect(self.ws_stream_url)
            
            # Start background tasks
            self._background_tasks.add(
                asyncio.create_task(self._ws_message_handler())
            )
            self._background_tasks.add(
                asyncio.create_task(self._user_stream_handler())
            )
            self._background_tasks.add(
                asyncio.create_task(self._keepalive_task())
            )
            
            self.is_connected = True
            logger.info("Connected to Binance exchange")
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Binance"""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            
            # Close WebSocket connections
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
            
            if self.user_stream_connection:
                await self.user_stream_connection.close()
                self.user_stream_connection = None
            
            # Stop user stream
            await self._stop_user_stream()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.is_connected = False
            logger.info("Disconnected from Binance exchange")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
    
    async def is_alive(self) -> bool:
        """Check if connection is alive"""
        try:
            # Ping REST API
            async with self.circuit_breaker:
                await self._make_request("GET", "/ping")
            
            # Check WebSocket connections
            ws_alive = self.ws_connection and not self.ws_connection.closed
            user_stream_alive = self.user_stream_connection and not self.user_stream_connection.closed
            
            return ws_alive and user_stream_alive
            
        except Exception:
            return False
    
    # Account information
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance"""
        try:
            # Get account info
            account_info = await self._make_request("GET", "/account", signed=True)
            
            balances = {}
            for balance_data in account_info["balances"]:
                symbol = balance_data["asset"]
                free = Decimal(balance_data["free"])
                locked = Decimal(balance_data["locked"])
                
                if asset and symbol != asset:
                    continue
                
                if free > 0 or locked > 0:
                    balances[symbol] = Balance(
                        asset=symbol,
                        free=free,
                        locked=locked,
                        total=free + locked
                    )
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balance: {str(e)}")
            raise
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions (spot trading doesn't have positions, return empty)"""
        # Note: For spot trading, we track positions separately
        # This would be different for futures trading
        return []
    
    # Market data
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data for symbol"""
        try:
            ticker_data = await self._make_request("GET", "/ticker/24hr", params={"symbol": symbol})
            
            return Ticker(
                symbol=ticker_data["symbol"],
                bid_price=Decimal(ticker_data["bidPrice"]),
                bid_qty=Decimal(ticker_data["bidQty"]),
                ask_price=Decimal(ticker_data["askPrice"]),
                ask_qty=Decimal(ticker_data["askQty"]),
                last_price=Decimal(ticker_data["lastPrice"]),
                volume_24h=Decimal(ticker_data["volume"]),
                high_24h=Decimal(ticker_data["highPrice"]),
                low_24h=Decimal(ticker_data["lowPrice"]),
                open_24h=Decimal(ticker_data["openPrice"]),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {str(e)}")
            raise
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book for symbol"""
        try:
            orderbook_data = await self._make_request(
                "GET", 
                "/depth", 
                params={"symbol": symbol, "limit": depth}
            )
            
            bids = [[Decimal(price), Decimal(qty)] for price, qty in orderbook_data["bids"]]
            asks = [[Decimal(price), Decimal(qty)] for price, qty in orderbook_data["asks"]]
            
            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {str(e)}")
            raise
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for symbol"""
        try:
            trades_data = await self._make_request(
                "GET", 
                "/trades", 
                params={"symbol": symbol, "limit": limit}
            )
            
            trades = []
            for trade_data in trades_data:
                trades.append(Trade(
                    id=str(trade_data["id"]),
                    order_id="",  # Not available in public trades
                    symbol=symbol,
                    side=OrderSide.BUY if trade_data["isBuyerMaker"] else OrderSide.SELL,
                    price=Decimal(trade_data["price"]),
                    quantity=Decimal(trade_data["qty"]),
                    fee=Decimal("0"),  # Not available in public trades
                    fee_asset="",
                    timestamp=datetime.fromtimestamp(trade_data["time"] / 1000, tz=timezone.utc),
                    is_maker=trade_data["isBuyerMaker"]
                ))
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {str(e)}")
            raise
    
    # Order management
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
        try:
            # Build order parameters
            params = {
                "symbol": symbol,
                "side": side.value.upper(),
                "type": self._map_order_type(type),
                "quantity": str(quantity)
            }
            
            # Add price for limit orders
            if type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                if not price:
                    raise ValueError(f"Price required for {type} orders")
                params["price"] = str(price)
            
            # Add time in force
            if type == OrderType.LIMIT:
                params["timeInForce"] = self._map_time_in_force(time_in_force)
            
            # Add client order ID
            if client_order_id:
                params["newClientOrderId"] = client_order_id
            
            # Add stop price for stop orders
            if "stop_price" in kwargs:
                params["stopPrice"] = str(kwargs["stop_price"])
            
            # Create order
            order_data = await self._make_request("POST", "/order", params=params, signed=True)
            
            # Convert to Order object
            return self._parse_order(order_data)
            
        except Exception as e:
            logger.error(f"Failed to create order: {str(e)}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel existing order"""
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            
            order_data = await self._make_request("DELETE", "/order", params=params, signed=True)
            return self._parse_order(order_data)
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise
    
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order details"""
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            
            order_data = await self._make_request("GET", "/order", params=params, signed=True)
            return self._parse_order(order_data)
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {str(e)}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            
            orders_data = await self._make_request("GET", "/openOrders", params=params, signed=True)
            
            orders = []
            for order_data in orders_data:
                orders.append(self._parse_order(order_data))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {str(e)}")
            raise
    
    async def get_order_history(
        self, 
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history"""
        try:
            params = {"limit": limit}
            if symbol:
                params["symbol"] = symbol
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            orders_data = await self._make_request("GET", "/allOrders", params=params, signed=True)
            
            orders = []
            for order_data in orders_data:
                orders.append(self._parse_order(order_data))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get order history: {str(e)}")
            raise
    
    # WebSocket streams
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker updates"""
        stream = f"{symbol.lower()}@ticker"
        await self._subscribe_stream(stream, callback, "ticker")
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to order book updates"""
        stream = f"{symbol.lower()}@depth20@100ms"
        await self._subscribe_stream(stream, callback, "orderbook")
    
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade updates"""
        stream = f"{symbol.lower()}@trade"
        await self._subscribe_stream(stream, callback, "trades")
    
    async def subscribe_orders(self, callback: Callable) -> None:
        """Subscribe to order updates"""
        # Order updates come through user data stream
        self.register_callback("order_update", callback)
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from stream"""
        # Find and remove subscription
        for stream, subs in self.subscriptions.items():
            if subscription_id in subs:
                subs.remove(subscription_id)
                if not subs:
                    # No more subscriptions for this stream
                    await self._unsubscribe_stream(stream)
                break
    
    # Private helper methods
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None, 
        signed: bool = False
    ) -> Any:
        """Make REST API request"""
        if not self.session:
            raise RuntimeError("Not connected to exchange")
        
        # Apply rate limiting
        async with self.rate_limiter:
            # Prepare request
            url = f"{self.rest_url}{endpoint}"
            headers = {}
            
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            
            # Add signature for signed requests
            if signed:
                if not params:
                    params = {}
                params["timestamp"] = int(time.time() * 1000)
                
                # Create signature
                query_string = urlencode(params)
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params["signature"] = signature
            
            # Make request
            async with self.circuit_breaker:
                async with self.session.request(method, url, params=params, headers=headers) as response:
                    data = await response.json()
                    
                    if response.status != 200:
                        error_msg = data.get("msg", f"HTTP {response.status}")
                        raise Exception(f"Binance API error: {error_msg}")
                    
                    return data
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse order data from API response"""
        return Order(
            id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            type=self._parse_order_type(data["type"]),
            side=OrderSide.BUY if data["side"] == "BUY" else OrderSide.SELL,
            status=self._parse_order_status(data["status"]),
            price=Decimal(data["price"]) if data.get("price") and data["price"] != "0.00000000" else None,
            quantity=Decimal(data["origQty"]),
            executed_qty=Decimal(data["executedQty"]),
            time_in_force=self._parse_time_in_force(data.get("timeInForce", "GTC")),
            created_at=datetime.fromtimestamp(data["time"] / 1000, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(data["updateTime"] / 1000, tz=timezone.utc),
            fills=[]
        )
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Binance type"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS",
            OrderType.STOP_LOSS_LIMIT: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
            OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT",
            OrderType.LIMIT_MAKER: "LIMIT_MAKER"
        }
        return mapping.get(order_type, "LIMIT")
    
    def _parse_order_type(self, binance_type: str) -> OrderType:
        """Parse Binance order type to internal type"""
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS_LIMIT,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT_LIMIT,
            "LIMIT_MAKER": OrderType.LIMIT_MAKER
        }
        return mapping.get(binance_type, OrderType.LIMIT)
    
    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map internal TIF to Binance TIF"""
        mapping = {
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK"
        }
        return mapping.get(tif, "GTC")
    
    def _parse_time_in_force(self, binance_tif: str) -> TimeInForce:
        """Parse Binance TIF to internal TIF"""
        mapping = {
            "GTC": TimeInForce.GTC,
            "IOC": TimeInForce.IOC,
            "FOK": TimeInForce.FOK
        }
        return mapping.get(binance_tif, TimeInForce.GTC)
    
    def _parse_order_status(self, binance_status: str) -> OrderStatus:
        """Parse Binance order status to internal status"""
        mapping = {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        return mapping.get(binance_status, OrderStatus.NEW)
    
    async def _subscribe_stream(self, stream: str, callback: Callable, data_type: str) -> None:
        """Subscribe to WebSocket stream"""
        if not self.ws_connection:
            raise RuntimeError("WebSocket not connected")
        
        # Generate subscription ID
        sub_id = f"{data_type}_{self.subscription_id_counter}"
        self.subscription_id_counter += 1
        
        # Add to subscriptions
        if stream not in self.subscriptions:
            self.subscriptions[stream] = set()
            
            # Send subscription message
            sub_message = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": self.subscription_id_counter
            }
            await self.ws_connection.send(json.dumps(sub_message))
        
        self.subscriptions[stream].add(sub_id)
        
        # Register callback
        self.register_callback(f"{stream}_{data_type}", callback)
    
    async def _unsubscribe_stream(self, stream: str) -> None:
        """Unsubscribe from WebSocket stream"""
        if not self.ws_connection:
            return
        
        # Send unsubscription message
        unsub_message = {
            "method": "UNSUBSCRIBE",
            "params": [stream],
            "id": self.subscription_id_counter
        }
        self.subscription_id_counter += 1
        
        await self.ws_connection.send(json.dumps(unsub_message))
        
        # Remove from subscriptions
        if stream in self.subscriptions:
            del self.subscriptions[stream]
    
    async def _ws_message_handler(self) -> None:
        """Handle WebSocket messages"""
        while self.is_connected and self.ws_connection:
            try:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                
                # Handle different message types
                if "stream" in data:
                    stream = data["stream"]
                    stream_data = data["data"]
                    
                    # Dispatch to appropriate handler
                    if "@ticker" in stream:
                        self._handle_ticker_update(stream, stream_data)
                    elif "@depth" in stream:
                        self._handle_orderbook_update(stream, stream_data)
                    elif "@trade" in stream:
                        self._handle_trade_update(stream, stream_data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting reconnect...")
                await self._reconnect_websocket()
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                await asyncio.sleep(0.1)
    
    def _handle_ticker_update(self, stream: str, data: Dict) -> None:
        """Handle ticker update"""
        try:
            ticker = Ticker(
                symbol=data["s"],
                bid_price=Decimal(data["b"]),
                bid_qty=Decimal(data["B"]),
                ask_price=Decimal(data["a"]),
                ask_qty=Decimal(data["A"]),
                last_price=Decimal(data["c"]),
                volume_24h=Decimal(data["v"]),
                high_24h=Decimal(data["h"]),
                low_24h=Decimal(data["l"]),
                open_24h=Decimal(data["o"]),
                timestamp=datetime.now(timezone.utc)
            )
            
            self.emit_event(f"{stream}_ticker", ticker)
            
        except Exception as e:
            logger.error(f"Error handling ticker update: {str(e)}")
    
    def _handle_orderbook_update(self, stream: str, data: Dict) -> None:
        """Handle orderbook update"""
        try:
            bids = [[Decimal(price), Decimal(qty)] for price, qty in data["bids"]]
            asks = [[Decimal(price), Decimal(qty)] for price, qty in data["asks"]]
            
            orderbook = OrderBook(
                symbol=stream.split("@")[0].upper(),
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.emit_event(f"{stream}_orderbook", orderbook)
            
        except Exception as e:
            logger.error(f"Error handling orderbook update: {str(e)}")
    
    def _handle_trade_update(self, stream: str, data: Dict) -> None:
        """Handle trade update"""
        try:
            trade = Trade(
                id=str(data["t"]),
                order_id="",
                symbol=data["s"],
                side=OrderSide.BUY if data["m"] else OrderSide.SELL,
                price=Decimal(data["p"]),
                quantity=Decimal(data["q"]),
                fee=Decimal("0"),
                fee_asset="",
                timestamp=datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
                is_maker=data["m"]
            )
            
            self.emit_event(f"{stream}_trades", trade)
            
        except Exception as e:
            logger.error(f"Error handling trade update: {str(e)}")
    
    async def _start_user_stream(self) -> None:
        """Start user data stream"""
        if not self.api_key:
            logger.warning("API key not provided, skipping user stream")
            return
        
        try:
            # Get listen key
            response = await self._make_request("POST", "/userDataStream")
            self.listen_key = response["listenKey"]
            
            # Connect to user stream
            user_stream_url = f"{self.ws_url}/{self.listen_key}"
            self.user_stream_connection = await websockets.connect(user_stream_url)
            
            logger.info("Started user data stream")
            
        except Exception as e:
            logger.error(f"Failed to start user stream: {str(e)}")
    
    async def _stop_user_stream(self) -> None:
        """Stop user data stream"""
        if self.listen_key:
            try:
                await self._make_request(
                    "DELETE", 
                    "/userDataStream",
                    params={"listenKey": self.listen_key}
                )
                self.listen_key = None
                logger.info("Stopped user data stream")
            except Exception as e:
                logger.error(f"Failed to stop user stream: {str(e)}")
    
    async def _user_stream_handler(self) -> None:
        """Handle user stream messages"""
        if not self.user_stream_connection:
            return
        
        while self.is_connected and self.user_stream_connection:
            try:
                message = await self.user_stream_connection.recv()
                data = json.loads(message)
                
                # Handle different event types
                event_type = data.get("e")
                
                if event_type == "executionReport":
                    # Order update
                    self._handle_order_update(data)
                elif event_type == "outboundAccountPosition":
                    # Balance update
                    self._handle_balance_update(data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("User stream connection closed, attempting reconnect...")
                await self._reconnect_user_stream()
            except Exception as e:
                logger.error(f"Error handling user stream message: {str(e)}")
                await asyncio.sleep(0.1)
    
    def _handle_order_update(self, data: Dict) -> None:
        """Handle order update from user stream"""
        try:
            order = Order(
                id=str(data["i"]),
                client_order_id=data.get("c"),
                symbol=data["s"],
                type=self._parse_order_type(data["o"]),
                side=OrderSide.BUY if data["S"] == "BUY" else OrderSide.SELL,
                status=self._parse_order_status(data["X"]),
                price=Decimal(data["p"]) if data["p"] != "0.00000000" else None,
                quantity=Decimal(data["q"]),
                executed_qty=Decimal(data["z"]),
                time_in_force=self._parse_time_in_force(data.get("f", "GTC")),
                created_at=datetime.fromtimestamp(data["O"] / 1000, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
                fills=[]
            )
            
            self.emit_event("order_update", order)
            
        except Exception as e:
            logger.error(f"Error handling order update: {str(e)}")
    
    def _handle_balance_update(self, data: Dict) -> None:
        """Handle balance update from user stream"""
        try:
            balances = {}
            for balance_data in data["B"]:
                asset = balance_data["a"]
                free = Decimal(balance_data["f"])
                locked = Decimal(balance_data["l"])
                
                if free > 0 or locked > 0:
                    balances[asset] = Balance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=free + locked
                    )
            
            self.emit_event("balance_update", balances)
            
        except Exception as e:
            logger.error(f"Error handling balance update: {str(e)}")
    
    async def _keepalive_task(self) -> None:
        """Keep connections alive"""
        while self.is_connected:
            try:
                # Ping WebSocket
                if self.ws_connection and not self.ws_connection.closed:
                    await self.ws_connection.ping()
                
                # Extend user stream listen key
                if self.listen_key:
                    await self._make_request(
                        "PUT",
                        "/userDataStream",
                        params={"listenKey": self.listen_key}
                    )
                
                # Wait 30 minutes before next keepalive
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in keepalive task: {str(e)}")
                await asyncio.sleep(60)
    
    async def _reconnect_websocket(self) -> None:
        """Reconnect WebSocket"""
        try:
            # Close existing connection
            if self.ws_connection:
                await self.ws_connection.close()
            
            # Reconnect
            self.ws_connection = await websockets.connect(self.ws_stream_url)
            
            # Resubscribe to all streams
            for stream in self.subscriptions:
                sub_message = {
                    "method": "SUBSCRIBE",
                    "params": [stream],
                    "id": self.subscription_id_counter
                }
                self.subscription_id_counter += 1
                await self.ws_connection.send(json.dumps(sub_message))
            
            logger.info("Reconnected WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to reconnect WebSocket: {str(e)}")
            await asyncio.sleep(5)
    
    async def _reconnect_user_stream(self) -> None:
        """Reconnect user stream"""
        try:
            # Close existing connection
            if self.user_stream_connection:
                await self.user_stream_connection.close()
            
            # Stop and restart user stream
            await self._stop_user_stream()
            await self._start_user_stream()
            
            logger.info("Reconnected user stream")
            
        except Exception as e:
            logger.error(f"Failed to reconnect user stream: {str(e)}")
            await asyncio.sleep(5)