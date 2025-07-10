"""
Conector para Coinbase Pro Exchange
Implementa la API REST y WebSocket de Coinbase Pro
"""

import asyncio
import json
import base64
import hmac
import hashlib
import time
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

class CoinbaseConnector(ExchangeBase):
    """Conector para Coinbase Pro Exchange"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        
        self.passphrase = passphrase
        
        # URLs base
        if testnet:
            self.rest_url = "https://api-public.sandbox.pro.coinbase.com"
            self.ws_url = "wss://ws-feed-public.sandbox.pro.coinbase.com"
        else:
            self.rest_url = "https://api.pro.coinbase.com"
            self.ws_url = "wss://ws-feed.pro.coinbase.com"
            
        self.ws_public = None
        self.ws_private = None
        self._ws_tasks = []
        
    async def connect(self):
        """Establecer conexión con Coinbase Pro"""
        logger.info("Conectando con Coinbase Pro...")
        
        # Verificar conectividad
        await self._request('GET', f"{self.rest_url}/time")
        
        logger.info("Conectado con Coinbase Pro exitosamente")
        
    async def disconnect(self):
        """Cerrar conexión con Coinbase Pro"""
        logger.info("Desconectando de Coinbase Pro...")
        
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
            
        logger.info("Desconectado de Coinbase Pro")
        
    async def get_ticker(self, symbol: str) -> Ticker:
        """Obtener ticker de un símbolo"""
        cb_symbol = self._convert_symbol(symbol)
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/products/{cb_symbol}/ticker"
        )
        
        # Coinbase Pro devuelve datos del ticker directamente
        return Ticker(
            symbol=symbol,
            bid=self._parse_decimal(response['bid']),
            ask=self._parse_decimal(response['ask']),
            last=self._parse_decimal(response['price']),
            volume=self._parse_decimal(response['volume']),
            timestamp=datetime.fromisoformat(response['time'].replace('Z', '+00:00'))
        )
        
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Obtener libro de órdenes"""
        cb_symbol = self._convert_symbol(symbol)
        
        # Coinbase Pro usa niveles (1, 2, 3) en lugar de límite
        level = 2  # Nivel 2 proporciona hasta 50 niveles por lado
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/products/{cb_symbol}/book",
            params={'level': level}
        )
        
        bids = [
            (self._parse_decimal(price), self._parse_decimal(size))
            for price, size, _ in response['bids'][:limit]
        ]
        
        asks = [
            (self._parse_decimal(price), self._parse_decimal(size))
            for price, size, _ in response['asks'][:limit]
        ]
        
        return OrderBook(
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
        
    async def get_balance(self) -> Dict[str, Balance]:
        """Obtener balances de la cuenta"""
        response = await self._request(
            'GET',
            f"{self.rest_url}/accounts",
            signed=True
        )
        
        balances = {}
        for account in response:
            currency = account['currency']
            balance = self._parse_decimal(account['balance'])
            hold = self._parse_decimal(account['hold'])
            available = self._parse_decimal(account['available'])
            
            if balance > 0:
                balances[currency] = Balance(
                    currency=currency,
                    free=available,
                    locked=hold,
                    total=balance
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
        cb_symbol = self._convert_symbol(symbol)
        
        order_data = {
            'product_id': cb_symbol,
            'side': side.lower(),
            'type': self._map_order_type(order_type)
        }
        
        # Para órdenes market, usar funds o size
        if order_type.lower() == 'market':
            if side.lower() == 'buy':
                # Para compras market, usar funds (cantidad en quote currency)
                order_data['funds'] = str(quantity * price if price else quantity)
            else:
                # Para ventas market, usar size
                order_data['size'] = str(quantity)
        else:
            # Para órdenes limit
            if not price:
                raise ExchangeError("Precio requerido para órdenes limit")
            order_data['price'] = str(price)
            order_data['size'] = str(quantity)
            
        response = await self._request(
            'POST',
            f"{self.rest_url}/orders",
            data=order_data,
            signed=True
        )
        
        return self._parse_order(response)
        
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancelar una orden"""
        try:
            await self._request(
                'DELETE',
                f"{self.rest_url}/orders/{order_id}",
                signed=True
            )
            return True
        except ExchangeError:
            return False
            
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Obtener información de una orden"""
        response = await self._request(
            'GET',
            f"{self.rest_url}/orders/{order_id}",
            signed=True
        )
        
        return self._parse_order(response)
        
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Obtener órdenes abiertas"""
        params = {'status': 'open'}
        if symbol:
            params['product_id'] = self._convert_symbol(symbol)
            
        response = await self._request(
            'GET',
            f"{self.rest_url}/orders",
            params=params,
            signed=True
        )
        
        return [self._parse_order(order) for order in response]
        
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Obtener historial de trades"""
        cb_symbol = self._convert_symbol(symbol)
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/fills",
            params={
                'product_id': cb_symbol,
                'limit': limit
            },
            signed=True
        )
        
        trades = []
        for fill in response:
            trades.append(Trade(
                id=str(fill['trade_id']),
                order_id=fill['order_id'],
                symbol=self._convert_from_cb_symbol(fill['product_id']),
                side=fill['side'],
                price=self._parse_decimal(fill['price']),
                quantity=self._parse_decimal(fill['size']),
                fee=self._parse_decimal(fill['fee']),
                fee_currency=fill['fee_currency'] if 'fee_currency' in fill else 'USD',
                timestamp=datetime.fromisoformat(fill['created_at'].replace('Z', '+00:00'))
            ))
            
        return trades
        
    async def subscribe_ticker(self, symbol: str, callback):
        """Suscribirse a actualizaciones del ticker"""
        cb_symbol = self._convert_symbol(symbol)
        
        if not self.ws_public:
            self.ws_public = await websockets.connect(self.ws_url)
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_public, callback, 'ticker'))
            )
            
        # Suscribirse al canal ticker
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [cb_symbol],
            "channels": ["ticker"]
        }
        
        await self.ws_public.send(json.dumps(subscribe_msg))
        
    async def subscribe_order_book(self, symbol: str, callback):
        """Suscribirse a actualizaciones del libro de órdenes"""
        cb_symbol = self._convert_symbol(symbol)
        
        if not self.ws_public:
            self.ws_public = await websockets.connect(self.ws_url)
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_public, callback, 'orderbook'))
            )
            
        # Suscribirse al canal level2
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [cb_symbol],
            "channels": ["level2"]
        }
        
        await self.ws_public.send(json.dumps(subscribe_msg))
        
    async def subscribe_trades(self, callback):
        """Suscribirse a actualizaciones de trades propios"""
        if not self.ws_private:
            # Generar firma para WebSocket privado
            timestamp = str(time.time())
            message = timestamp + 'GET' + '/users/self/verify'
            signature = self._generate_cb_signature(message, timestamp)
            
            self.ws_private = await websockets.connect(self.ws_url)
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_private, callback, 'trades'))
            )
            
        # Suscribirse al canal user (autenticado)
        subscribe_msg = {
            "type": "subscribe",
            "channels": ["user"],
            "signature": signature,
            "key": self.api_key,
            "passphrase": self.passphrase,
            "timestamp": timestamp
        }
        
        await self.ws_private.send(json.dumps(subscribe_msg))
        
    # Métodos privados
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convertir símbolo al formato de Coinbase Pro"""
        # Coinbase Pro usa guiones: BTC-USD
        return symbol.replace('/', '-')
        
    def _convert_from_cb_symbol(self, cb_symbol: str) -> str:
        """Convertir símbolo de Coinbase Pro al formato estándar"""
        return cb_symbol.replace('-', '/')
        
    def _map_order_type(self, order_type: str) -> str:
        """Mapear tipo de orden al formato de Coinbase Pro"""
        mapping = {
            'limit': 'limit',
            'market': 'market',
            'stop': 'stop',
            'stop_limit': 'stop_limit'
        }
        return mapping.get(order_type.lower(), order_type.lower())
        
    def _parse_order(self, data: Dict) -> Order:
        """Parsear respuesta de orden"""
        return Order(
            id=data['id'],
            symbol=self._convert_from_cb_symbol(data['product_id']),
            side=data['side'],
            type=data['type'],
            price=self._parse_decimal(data.get('price')) if data.get('price') else None,
            quantity=self._parse_decimal(data.get('size', data.get('specified_funds', 0))),
            status=self._map_order_status(data['status']),
            timestamp=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            filled=self._parse_decimal(data.get('filled_size', 0)),
            remaining=self._parse_decimal(data.get('size', 0)) - self._parse_decimal(data.get('filled_size', 0))
        )
        
    def _map_order_status(self, status: str) -> str:
        """Mapear estado de orden"""
        mapping = {
            'pending': 'open',
            'open': 'open',
            'active': 'open',
            'done': 'filled',
            'cancelled': 'cancelled',
            'rejected': 'cancelled'
        }
        return mapping.get(status, status.lower())
        
    def _generate_cb_signature(self, message: str, timestamp: str) -> str:
        """Generar firma para Coinbase Pro"""
        key = base64.b64decode(self.api_secret)
        signature = hmac.new(key, message.encode(), hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()
        
    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """Realizar petición HTTP con firma de Coinbase Pro"""
        if signed:
            timestamp = str(time.time())
            path = url.replace(self.rest_url, '')
            
            # Crear mensaje para firma
            body = json.dumps(data) if data else ''
            message = timestamp + method + path + body
            
            # Generar firma
            signature = self._generate_cb_signature(message, timestamp)
            
            headers = headers or {}
            headers.update({
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': signature,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            })
            
        return await super()._request(method, url, headers, params, data, signed=False)
        
    async def _handle_ws_messages(self, ws, callback, msg_type: str):
        """Manejar mensajes WebSocket"""
        try:
            async for message in ws:
                data = json.loads(message)
                
                # Ignorar mensajes de confirmación
                if data.get('type') in ['subscriptions', 'heartbeat']:
                    continue
                    
                if msg_type == 'ticker' and data.get('type') == 'ticker':
                    ticker = Ticker(
                        symbol=self._convert_from_cb_symbol(data['product_id']),
                        bid=self._parse_decimal(data['best_bid']),
                        ask=self._parse_decimal(data['best_ask']),
                        last=self._parse_decimal(data['price']),
                        volume=self._parse_decimal(data['volume_24h']),
                        timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                    )
                    await callback(ticker)
                    
                elif msg_type == 'orderbook':
                    if data.get('type') in ['snapshot', 'l2update']:
                        await callback(data)
                        
                elif msg_type == 'trades':
                    if data.get('type') in ['received', 'open', 'done', 'match']:
                        await callback(data)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket de Coinbase Pro cerrado")
        except Exception as e:
            logger.error(f"Error en WebSocket de Coinbase Pro: {e}")