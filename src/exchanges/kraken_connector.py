"""
Conector para Kraken Exchange
Implementa la API REST y WebSocket de Kraken
"""

import asyncio
import json
import base64
import urllib.parse
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import websockets
import logging

from .base import (
    ExchangeBase, ExchangeError,
    OrderBook, Ticker, Balance, Order, Trade
)

logger = logging.getLogger(__name__)

class KrakenConnector(ExchangeBase):
    """Conector para Kraken Exchange"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        
        # URLs base
        self.rest_url = "https://api.kraken.com"
        self.ws_url = "wss://ws.kraken.com"
        
        # Mapeo de pares de Kraken
        self.symbol_map = {
            'BTC/USD': 'XBTUSD',
            'ETH/USD': 'ETHUSD',
            'BTC/EUR': 'XBTEUR',
            'ETH/EUR': 'ETHEUR',
            'LTC/USD': 'LTCUSD',
            'XRP/USD': 'XRPUSD',
        }
        
        self.ws_public = None
        self.ws_private = None
        self._ws_tasks = []
        self.token = None
        
    async def connect(self):
        """Establecer conexión con Kraken"""
        logger.info("Conectando con Kraken...")
        
        # Verificar conectividad
        await self._request('GET', f"{self.rest_url}/0/public/Time")
        
        # Obtener token para WebSocket privado
        if self.api_key and self.api_secret:
            response = await self._request(
                'POST',
                f"{self.rest_url}/0/private/GetWebSocketsToken",
                signed=True
            )
            if response.get('error'):
                raise ExchangeError(f"Error obteniendo token WS: {response['error']}")
            self.token = response['result']['token']
        
        logger.info("Conectado con Kraken exitosamente")
        
    async def disconnect(self):
        """Cerrar conexión con Kraken"""
        logger.info("Desconectando de Kraken...")
        
        # Cancelar tareas
        for task in self._ws_tasks:
            task.cancel()
            
        # Cerrar websockets
        if self.ws_public:
            await self.ws_public.close()
        if self.ws_private:
            await self.ws_private.close()
            
        # Cerrar sesión HTTP
        if self.session:
            await self.session.close()
            
        logger.info("Desconectado de Kraken")
        
    async def get_ticker(self, symbol: str) -> Ticker:
        """Obtener ticker de un símbolo"""
        kraken_symbol = self._convert_symbol(symbol)
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/0/public/Ticker",
            params={'pair': kraken_symbol}
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error obteniendo ticker: {response['error']}")
            
        data = response['result'][kraken_symbol]
        
        return Ticker(
            symbol=symbol,
            bid=self._parse_decimal(data['b'][0]),
            ask=self._parse_decimal(data['a'][0]),
            last=self._parse_decimal(data['c'][0]),
            volume=self._parse_decimal(data['v'][1]),  # Volumen 24h
            timestamp=datetime.now()
        )
        
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Obtener libro de órdenes"""
        kraken_symbol = self._convert_symbol(symbol)
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/0/public/Depth",
            params={'pair': kraken_symbol, 'count': limit}
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error obteniendo orderbook: {response['error']}")
            
        data = response['result'][kraken_symbol]
        
        bids = [
            (self._parse_decimal(price), self._parse_decimal(volume))
            for price, volume, _ in data['bids']
        ]
        
        asks = [
            (self._parse_decimal(price), self._parse_decimal(volume))
            for price, volume, _ in data['asks']
        ]
        
        return OrderBook(
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
        
    async def get_balance(self) -> Dict[str, Balance]:
        """Obtener balances de la cuenta"""
        response = await self._request(
            'POST',
            f"{self.rest_url}/0/private/Balance",
            signed=True
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error obteniendo balance: {response['error']}")
            
        balances = {}
        for currency, amount in response['result'].items():
            amount = self._parse_decimal(amount)
            if amount > 0:
                # Kraken no separa free/locked en el balance básico
                balances[currency] = Balance(
                    currency=currency,
                    free=amount,
                    locked=Decimal('0'),
                    total=amount
                )
                
        return balances
        
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Order:
        """Crear una nueva orden"""
        kraken_symbol = self._convert_symbol(symbol)
        
        params = {
            'pair': kraken_symbol,
            'type': side.lower(),
            'ordertype': self._map_order_type(order_type),
            'volume': str(quantity)
        }
        
        # Añadir precio para órdenes limit
        if order_type.lower() == 'limit':
            if not price:
                raise ExchangeError("Precio requerido para órdenes limit")
            params['price'] = str(price)
            
        response = await self._request(
            'POST',
            f"{self.rest_url}/0/private/AddOrder",
            data=params,
            signed=True
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error creando orden: {response['error']}")
            
        # Kraken devuelve solo el ID, necesitamos obtener la orden completa
        order_id = response['result']['txid'][0]
        return await self.get_order(order_id, symbol)
        
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancelar una orden"""
        try:
            response = await self._request(
                'POST',
                f"{self.rest_url}/0/private/CancelOrder",
                data={'txid': order_id},
                signed=True
            )
            
            if response.get('error'):
                return False
                
            return True
        except ExchangeError:
            return False
            
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Obtener información de una orden"""
        response = await self._request(
            'POST',
            f"{self.rest_url}/0/private/QueryOrders",
            data={'txid': order_id},
            signed=True
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error obteniendo orden: {response['error']}")
            
        if order_id not in response['result']:
            raise ExchangeError(f"Orden {order_id} no encontrada")
            
        data = response['result'][order_id]
        return self._parse_order(order_id, data, symbol)
        
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Obtener órdenes abiertas"""
        response = await self._request(
            'POST',
            f"{self.rest_url}/0/private/OpenOrders",
            signed=True
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error obteniendo órdenes: {response['error']}")
            
        orders = []
        for order_id, data in response['result']['open'].items():
            pair = data['descr']['pair']
            # Convertir par de Kraken a formato estándar
            converted_symbol = self._convert_from_kraken_symbol(pair)
            
            if not symbol or converted_symbol == symbol:
                orders.append(self._parse_order(order_id, data, converted_symbol))
                
        return orders
        
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Obtener historial de trades"""
        response = await self._request(
            'POST',
            f"{self.rest_url}/0/private/TradesHistory",
            signed=True
        )
        
        if response.get('error'):
            raise ExchangeError(f"Error obteniendo trades: {response['error']}")
            
        trades = []
        kraken_symbol = self._convert_symbol(symbol)
        
        for trade_id, data in response['result']['trades'].items():
            if data['pair'] == kraken_symbol:
                trades.append(Trade(
                    id=trade_id,
                    order_id=data['ordertxid'],
                    symbol=symbol,
                    side=data['type'],
                    price=self._parse_decimal(data['price']),
                    quantity=self._parse_decimal(data['vol']),
                    fee=self._parse_decimal(data['fee']),
                    fee_currency=data['pair'][-3:],  # Últimos 3 caracteres como moneda
                    timestamp=datetime.fromtimestamp(float(data['time']))
                ))
                
                if len(trades) >= limit:
                    break
                    
        return trades
        
    async def subscribe_ticker(self, symbol: str, callback):
        """Suscribirse a actualizaciones del ticker"""
        kraken_symbol = self._convert_symbol(symbol)
        
        if not self.ws_public:
            self.ws_public = await websockets.connect(self.ws_url)
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_public, callback, 'ticker'))
            )
            
        # Suscribirse al ticker
        subscribe_msg = {
            "event": "subscribe",
            "pair": [kraken_symbol],
            "subscription": {"name": "ticker"}
        }
        
        await self.ws_public.send(json.dumps(subscribe_msg))
        
    async def subscribe_order_book(self, symbol: str, callback):
        """Suscribirse a actualizaciones del libro de órdenes"""
        kraken_symbol = self._convert_symbol(symbol)
        
        if not self.ws_public:
            self.ws_public = await websockets.connect(self.ws_url)
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_public, callback, 'orderbook'))
            )
            
        # Suscribirse al orderbook
        subscribe_msg = {
            "event": "subscribe",
            "pair": [kraken_symbol],
            "subscription": {"name": "book", "depth": 10}
        }
        
        await self.ws_public.send(json.dumps(subscribe_msg))
        
    async def subscribe_trades(self, callback):
        """Suscribirse a actualizaciones de trades propios"""
        if not self.token:
            raise ExchangeError("Token WebSocket requerido para trades privados")
            
        if not self.ws_private:
            self.ws_private = await websockets.connect(f"{self.ws_url}?token={self.token}")
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_private, callback, 'trades'))
            )
            
        # Suscribirse a órdenes propias
        subscribe_msg = {
            "event": "subscribe",
            "subscription": {"name": "ownTrades", "token": self.token}
        }
        
        await self.ws_private.send(json.dumps(subscribe_msg))
        
    # Métodos privados
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convertir símbolo al formato de Kraken"""
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        # Formato por defecto para Kraken
        return symbol.replace('/', '')
        
    def _convert_from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convertir símbolo de Kraken al formato estándar"""
        for std_symbol, kraken_sym in self.symbol_map.items():
            if kraken_sym == kraken_symbol:
                return std_symbol
        # Por defecto, insertar / en el medio
        if len(kraken_symbol) == 6:
            return f"{kraken_symbol[:3]}/{kraken_symbol[3:]}"
        return kraken_symbol
        
    def _map_order_type(self, order_type: str) -> str:
        """Mapear tipo de orden al formato de Kraken"""
        mapping = {
            'limit': 'limit',
            'market': 'market',
            'stop': 'stop-loss',
            'stop_limit': 'stop-loss-limit'
        }
        return mapping.get(order_type.lower(), order_type.lower())
        
    def _parse_order(self, order_id: str, data: Dict, symbol: str) -> Order:
        """Parsear respuesta de orden"""
        return Order(
            id=order_id,
            symbol=symbol,
            side=data['descr']['type'],
            type=data['descr']['ordertype'],
            price=self._parse_decimal(data['descr']['price']) if data['descr']['price'] else None,
            quantity=self._parse_decimal(data['vol']),
            status=self._map_order_status(data['status']),
            timestamp=datetime.fromtimestamp(float(data['opentm'])),
            filled=self._parse_decimal(data['vol_exec']),
            remaining=self._parse_decimal(data['vol']) - self._parse_decimal(data['vol_exec'])
        )
        
    def _map_order_status(self, status: str) -> str:
        """Mapear estado de orden"""
        mapping = {
            'pending': 'open',
            'open': 'open',
            'closed': 'filled',
            'canceled': 'cancelled',
            'expired': 'cancelled'
        }
        return mapping.get(status, status.lower())
        
    def _generate_signature(self, url_path: str, data: str) -> str:
        """Generar firma para Kraken"""
        postdata = urllib.parse.urlencode(data) if isinstance(data, dict) else data
        encoded = (str(data['nonce']) + postdata).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512
        )
        
        return base64.b64encode(mac.digest()).decode()
        
    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """Realizar petición HTTP con firma de Kraken"""
        if signed:
            if not data:
                data = {}
            data['nonce'] = str(int(time.time() * 1000))
            
            # Generar firma
            url_path = url.replace(self.rest_url, '')
            signature = self._generate_signature(url_path, data)
            
            headers = headers or {}
            headers.update({
                'API-Key': self.api_key,
                'API-Sign': signature
            })
            
        return await super()._request(method, url, headers, params, data, signed=False)
        
    async def _handle_ws_messages(self, ws, callback, msg_type: str):
        """Manejar mensajes WebSocket"""
        try:
            async for message in ws:
                data = json.loads(message)
                
                # Ignorar mensajes de sistema
                if isinstance(data, dict):
                    continue
                    
                if msg_type == 'ticker' and len(data) > 3:
                    ticker_data = data[1]
                    symbol = data[3]
                    
                    ticker = Ticker(
                        symbol=self._convert_from_kraken_symbol(symbol),
                        bid=self._parse_decimal(ticker_data['b'][0]),
                        ask=self._parse_decimal(ticker_data['a'][0]),
                        last=self._parse_decimal(ticker_data['c'][0]),
                        volume=self._parse_decimal(ticker_data['v'][1]),
                        timestamp=datetime.now()
                    )
                    await callback(ticker)
                    
                elif msg_type == 'orderbook':
                    await callback(data)
                    
                elif msg_type == 'trades':
                    await callback(data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket de Kraken cerrado")
        except Exception as e:
            logger.error(f"Error en WebSocket de Kraken: {e}")

import hashlib
import hmac
import time