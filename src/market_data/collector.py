"""
Data Collector for Market Data

Collects raw market data from exchanges through various protocols
(WebSocket, REST, FIX, etc.)
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum

from ..core.observability import get_logger
from ..core.architecture import CircuitBreaker
from ..exchange.exchange_interface import ExchangeInterface

logger = get_logger(__name__)

class ConnectionState(Enum):
    """Connection states for data collectors"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class DataCollector(ABC):
    """Abstract base class for data collectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ConnectionState.DISCONNECTED
        self.logger = logger
        self.is_running = False
        
    @abstractmethod
    async def connect(self):
        """Establish connection to data source"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass
        
    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[str]):
        """Subscribe to specific data streams"""
        pass        
    @abstractmethod
    async def unsubscribe(self, symbols: List[str], data_types: List[str]):
        """Unsubscribe from data streams"""
        pass
        
    @property
    def is_connected(self) -> bool:
        """Check if collector is connected"""
        return self.state == ConnectionState.CONNECTED

class ExchangeDataCollector(DataCollector):
    """
    Collects market data from a specific exchange
    """
    
    def __init__(
        self,
        exchange: ExchangeInterface,
        symbols: List[str],
        data_types: List[str],
        callback: Callable
    ):
        super().__init__(f"{exchange.name}_collector")
        self.exchange = exchange
        self.symbols = set(symbols)
        self.data_types = set(data_types)
        self.callback = callback
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        
        # REST polling tasks
        self.polling_tasks: List[asyncio.Task] = []
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            success_threshold=3
        )
        
        # Subscription state
        self.active_subscriptions: Set[str] = set()
        
        # Reconnection settings
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10        
    async def start(self):
        """Start collecting data"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info(f"Starting {self.name}")
        
        # Connect to exchange
        await self.connect()
        
        # Subscribe to data streams
        await self.subscribe(list(self.symbols), list(self.data_types))
        
        # Start monitoring
        asyncio.create_task(self._monitor_connection())
        
    async def stop(self):
        """Stop collecting data"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.logger.info(f"Stopping {self.name}")
        
        # Cancel polling tasks
        for task in self.polling_tasks:
            task.cancel()
            
        # Unsubscribe and disconnect
        await self.unsubscribe(list(self.symbols), list(self.data_types))
        await self.disconnect()
        
    async def connect(self):
        """Connect to exchange data feeds"""
        self.state = ConnectionState.CONNECTING
        
        try:
            # Use circuit breaker for connection
            await self.circuit_breaker.call(self._establish_connections)
            self.state = ConnectionState.CONNECTED
            self.logger.info(f"{self.name} connected successfully")
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.logger.error(f"{self.name} connection failed: {str(e)}")
            raise            
    async def _establish_connections(self):
        """Establish connections based on data types"""
        # Check which protocols to use
        use_websocket = any(dt in ['ticker', 'orderbook', 'trades'] for dt in self.data_types)
        use_rest = any(dt in ['candles', 'funding'] for dt in self.data_types)
        
        if use_websocket and hasattr(self.exchange, 'ws_connect'):
            # Connect WebSocket
            ws = await self.exchange.ws_connect()
            self.ws_connections['main'] = ws
            
            # Setup message handler
            asyncio.create_task(self._handle_ws_messages(ws))
            
        if use_rest:
            # Start REST polling for specific data types
            if 'candles' in self.data_types:
                task = asyncio.create_task(self._poll_candles())
                self.polling_tasks.append(task)
                
    async def disconnect(self):
        """Disconnect from exchange"""
        self.state = ConnectionState.DISCONNECTED
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            if hasattr(ws, 'close'):
                await ws.close()
                
        self.ws_connections.clear()
        
    async def subscribe(self, symbols: List[str], data_types: List[str]):
        """Subscribe to market data streams"""
        # Build subscription messages
        subscriptions = []
        
        for symbol in symbols:
            for data_type in data_types:
                if data_type == 'ticker':
                    subscriptions.append({
                        'method': 'SUBSCRIBE',
                        'params': [f"{symbol.lower()}@ticker"],
                        'id': len(subscriptions) + 1
                    })                elif data_type == 'orderbook':
                    subscriptions.append({
                        'method': 'SUBSCRIBE',
                        'params': [f"{symbol.lower()}@depth20@100ms"],
                        'id': len(subscriptions) + 1
                    })
                elif data_type == 'trades':
                    subscriptions.append({
                        'method': 'SUBSCRIBE',
                        'params': [f"{symbol.lower()}@trade"],
                        'id': len(subscriptions) + 1
                    })
                    
        # Send subscriptions
        if subscriptions and 'main' in self.ws_connections:
            ws = self.ws_connections['main']
            for sub in subscriptions:
                await ws.send_json(sub)
                self.active_subscriptions.add(f"{sub['params'][0]}")
                
        self.logger.info(f"Subscribed to {len(subscriptions)} streams")
        
    async def unsubscribe(self, symbols: List[str], data_types: List[str]):
        """Unsubscribe from data streams"""
        # Similar to subscribe but with UNSUBSCRIBE method
        pass        
    async def _handle_ws_messages(self, ws):
        """Handle incoming WebSocket messages"""
        try:
            while self.is_running and self.state == ConnectionState.CONNECTED:
                msg = await ws.receive()
                
                if msg.type == 'text':
                    data = msg.json()
                    await self._process_ws_message(data)
                elif msg.type == 'error':
                    self.logger.error(f"WebSocket error: {msg.data}")
                elif msg.type == 'close':
                    self.logger.warning("WebSocket connection closed")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error handling WebSocket messages: {str(e)}")
            
        # Trigger reconnection if needed
        if self.is_running:
            self.state = ConnectionState.RECONNECTING
            await self._reconnect()
            
    async def _process_ws_message(self, data: Dict[str, Any]):
        """Process WebSocket message and call callback"""
        # Extract message type and data
        if 'stream' in data:
            stream = data['stream']
            stream_data = data['data']
            
            # Parse stream type
            if '@ticker' in stream:
                symbol = stream.split('@')[0].upper()
                await self.callback(
                    self.exchange.name,
                    symbol,
                    'ticker',
                    stream_data,
                    datetime.now()
                )            elif '@depth' in stream:
                symbol = stream.split('@')[0].upper()
                await self.callback(
                    self.exchange.name,
                    symbol,
                    'orderbook',
                    stream_data,
                    datetime.now()
                )
            elif '@trade' in stream:
                symbol = stream.split('@')[0].upper()
                await self.callback(
                    self.exchange.name,
                    symbol,
                    'trades',
                    stream_data,
                    datetime.now()
                )
                
    async def _poll_candles(self):
        """Poll for candle data via REST"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # Fetch candles from exchange
                    candles = await self.exchange.fetch_ohlcv(
                        symbol, '1m', limit=100
                    )
                    
                    if candles:
                        await self.callback(
                            self.exchange.name,
                            symbol,
                            'candles',
                            candles,
                            datetime.now()
                        )
                        
                # Poll every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error polling candles: {str(e)}")
                await asyncio.sleep(5)  # Retry after short delay                
    async def _monitor_connection(self):
        """Monitor connection health"""
        while self.is_running:
            try:
                # Check WebSocket health
                for name, ws in self.ws_connections.items():
                    if hasattr(ws, 'ping'):
                        await ws.ping()
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Connection health check failed: {str(e)}")
                if self.state == ConnectionState.CONNECTED:
                    self.state = ConnectionState.RECONNECTING
                    await self._reconnect()
                    
    async def _reconnect(self):
        """Reconnect with exponential backoff"""
        attempt = 0
        
        while self.is_running and attempt < self.max_reconnect_attempts:
            attempt += 1
            delay = min(self.reconnect_delay * (2 ** (attempt - 1)), 300)  # Max 5 minutes
            
            self.logger.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts} in {delay}s")
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                await self.subscribe(list(self.symbols), list(self.data_types))
                self.logger.info("Reconnection successful")
                return
                
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt} failed: {str(e)}")
                
        self.logger.error("Max reconnection attempts reached")
        self.state = ConnectionState.ERRORclass WebSocketCollector(DataCollector):
    """
    Generic WebSocket data collector
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        callback: Callable
    ):
        super().__init__(name)
        self.url = url
        self.callback = callback
        self.ws = None
        
    async def connect(self):
        """Connect to WebSocket endpoint"""
        import aiohttp
        
        self.state = ConnectionState.CONNECTING
        session = aiohttp.ClientSession()
        self.ws = await session.ws_connect(self.url)
        self.state = ConnectionState.CONNECTED
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
        
    async def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.state = ConnectionState.DISCONNECTED
        
    async def subscribe(self, symbols: List[str], data_types: List[str]):
        """Send subscription message"""
        if self.ws:
            sub_msg = {
                'action': 'subscribe',
                'symbols': symbols,
                'types': data_types
            }
            await self.ws.send_json(sub_msg)            
    async def unsubscribe(self, symbols: List[str], data_types: List[str]):
        """Send unsubscribe message"""
        if self.ws:
            unsub_msg = {
                'action': 'unsubscribe',
                'symbols': symbols,
                'types': data_types
            }
            await self.ws.send_json(unsub_msg)
            
    async def _handle_messages(self):
        """Handle incoming messages"""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    await self.callback(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {self.ws.exception()}')
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in message handler: {str(e)}")
            
        self.state = ConnectionState.DISCONNECTED

class RESTCollector(DataCollector):
    """
    REST API data collector with polling
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        endpoints: Dict[str, str],
        poll_interval: int,
        callback: Callable
    ):
        super().__init__(name)
        self.base_url = base_url
        self.endpoints = endpoints
        self.poll_interval = poll_interval
        self.callback = callback
        self.polling_tasks = []        
    async def connect(self):
        """Start REST polling"""
        self.state = ConnectionState.CONNECTED
        
    async def disconnect(self):
        """Stop REST polling"""
        for task in self.polling_tasks:
            task.cancel()
        self.polling_tasks.clear()
        self.state = ConnectionState.DISCONNECTED
        
    async def subscribe(self, symbols: List[str], data_types: List[str]):
        """Start polling for specified data"""
        for data_type in data_types:
            if data_type in self.endpoints:
                for symbol in symbols:
                    task = asyncio.create_task(
                        self._poll_endpoint(symbol, data_type)
                    )
                    self.polling_tasks.append(task)
                    
    async def unsubscribe(self, symbols: List[str], data_types: List[str]):
        """Stop polling - handled by disconnect"""
        pass
        
    async def _poll_endpoint(self, symbol: str, data_type: str):
        """Poll a specific endpoint"""
        import aiohttp
        
        endpoint = self.endpoints[data_type]
        url = f"{self.base_url}{endpoint}".format(symbol=symbol)
        
        async with aiohttp.ClientSession() as session:
            while self.is_running:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self.callback(symbol, data_type, data)
                            
                    await asyncio.sleep(self.poll_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error polling {url}: {str(e)}")
                    await asyncio.sleep(self.poll_interval * 2)  # Backoff on error