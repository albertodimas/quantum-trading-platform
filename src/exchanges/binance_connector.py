"""
Conector para Binance Exchange
Implementa la API REST y WebSocket de Binance
"""

import asyncio
import json
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

class BinanceConnector(ExchangeBase):
    """Conector para Binance Exchange"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        
        # URLs base
        if testnet:
            self.rest_url = "https://testnet.binance.vision/api/v3"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.rest_url = "https://api.binance.com/api/v3"
            self.ws_url = "wss://stream.binance.com:9443/ws"
            
        self.listen_key = None
        self.ws_public = None
        self.ws_private = None
        self._ws_tasks = []
        
    async def connect(self):
        """Establecer conexión con Binance"""
        logger.info("Conectando con Binance...")
        
        # Verificar conectividad
        await self._request('GET', f"{self.rest_url}/ping")
        
        # Obtener listen key para user data stream
        response = await self._request(
            'POST',
            f"{self.rest_url}/userDataStream",
            headers={'X-MBX-APIKEY': self.api_key}
        )
        self.listen_key = response['listenKey']
        
        # Iniciar keep-alive para listen key
        self._ws_tasks.append(
            asyncio.create_task(self._keep_alive_listen_key())
        )
        
        logger.info("Conectado con Binance exitosamente")
        
    async def disconnect(self):
        """Cerrar conexión con Binance"""
        logger.info("Desconectando de Binance...")
        
        # Cancelar tareas
        for task in self._ws_tasks:
            task.cancel()
            
        # Cerrar websockets
        if self.ws_public:
            await self.ws_public.close()
        if self.ws_private:
            await self.ws_private.close()
            
        # Cerrar listen key
        if self.listen_key:
            await self._request(
                'DELETE',
                f"{self.rest_url}/userDataStream",
                headers={'X-MBX-APIKEY': self.api_key},
                params={'listenKey': self.listen_key}
            )
            
        # Cerrar sesión HTTP
        if self.session:
            await self.session.close()
            
        logger.info("Desconectado de Binance")
        
    async def get_ticker(self, symbol: str) -> Ticker:
        """Obtener ticker de un símbolo"""
        symbol = symbol.replace('/', '')
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/ticker/24hr",
            params={'symbol': symbol}
        )
        
        return Ticker(
            symbol=symbol,
            bid=self._parse_decimal(response['bidPrice']),
            ask=self._parse_decimal(response['askPrice']),
            last=self._parse_decimal(response['lastPrice']),
            volume=self._parse_decimal(response['volume']),
            timestamp=datetime.fromtimestamp(response['closeTime'] / 1000)
        )
        
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Obtener libro de órdenes"""
        symbol = symbol.replace('/', '')
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/depth",
            params={'symbol': symbol, 'limit': limit}
        )
        
        bids = [
            (self._parse_decimal(price), self._parse_decimal(qty))
            for price, qty in response['bids']
        ]
        
        asks = [
            (self._parse_decimal(price), self._parse_decimal(qty))
            for price, qty in response['asks']
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
            f"{self.rest_url}/account",
            headers={'X-MBX-APIKEY': self.api_key},
            signed=True
        )
        
        balances = {}
        for balance in response['balances']:
            free = self._parse_decimal(balance['free'])
            locked = self._parse_decimal(balance['locked'])
            
            if free > 0 or locked > 0:
                balances[balance['asset']] = Balance(
                    currency=balance['asset'],
                    free=free,
                    locked=locked,
                    total=free + locked
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
        symbol = symbol.replace('/', '')
        
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': self._map_order_type(order_type),
            'quantity': str(quantity)
        }
        
        # Añadir precio para órdenes limit
        if order_type.lower() == 'limit':
            if not price:
                raise ExchangeError("Precio requerido para órdenes limit")
            params['price'] = str(price)
            params['timeInForce'] = 'GTC'  # Good Till Cancel
            
        response = await self._request(
            'POST',
            f"{self.rest_url}/order",
            headers={'X-MBX-APIKEY': self.api_key},
            params=params,
            signed=True
        )
        
        return self._parse_order(response)
        
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancelar una orden"""
        symbol = symbol.replace('/', '')
        
        try:
            await self._request(
                'DELETE',
                f"{self.rest_url}/order",
                headers={'X-MBX-APIKEY': self.api_key},
                params={
                    'symbol': symbol,
                    'orderId': order_id
                },
                signed=True
            )
            return True
        except ExchangeError:
            return False
            
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Obtener información de una orden"""
        symbol = symbol.replace('/', '')
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/order",
            headers={'X-MBX-APIKEY': self.api_key},
            params={
                'symbol': symbol,
                'orderId': order_id
            },
            signed=True
        )
        
        return self._parse_order(response)
        
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Obtener órdenes abiertas"""
        params = {}
        if symbol:
            params['symbol'] = symbol.replace('/', '')
            
        response = await self._request(
            'GET',
            f"{self.rest_url}/openOrders",
            headers={'X-MBX-APIKEY': self.api_key},
            params=params,
            signed=True
        )
        
        return [self._parse_order(order) for order in response]
        
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Obtener historial de trades"""
        symbol = symbol.replace('/', '')
        
        response = await self._request(
            'GET',
            f"{self.rest_url}/myTrades",
            headers={'X-MBX-APIKEY': self.api_key},
            params={
                'symbol': symbol,
                'limit': limit
            },
            signed=True
        )
        
        trades = []
        for trade in response:
            trades.append(Trade(
                id=str(trade['id']),
                order_id=str(trade['orderId']),
                symbol=trade['symbol'],
                side='buy' if trade['isBuyer'] else 'sell',
                price=self._parse_decimal(trade['price']),
                quantity=self._parse_decimal(trade['qty']),
                fee=self._parse_decimal(trade['commission']),
                fee_currency=trade['commissionAsset'],
                timestamp=datetime.fromtimestamp(trade['time'] / 1000)
            ))
            
        return trades
        
    async def subscribe_ticker(self, symbol: str, callback):
        """Suscribirse a actualizaciones del ticker"""
        symbol = symbol.replace('/', '').lower()
        stream = f"{symbol}@ticker"
        
        if not self.ws_public:
            self.ws_public = await websockets.connect(f"{self.ws_url}/{stream}")
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_public, callback, 'ticker'))
            )
        else:
            # Suscribirse a stream adicional
            await self.ws_public.send(json.dumps({
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": 1
            }))
            
    async def subscribe_order_book(self, symbol: str, callback):
        """Suscribirse a actualizaciones del libro de órdenes"""
        symbol = symbol.replace('/', '').lower()
        stream = f"{symbol}@depth"
        
        if not self.ws_public:
            self.ws_public = await websockets.connect(f"{self.ws_url}/{stream}")
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_public, callback, 'orderbook'))
            )
        else:
            await self.ws_public.send(json.dumps({
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": 2
            }))
            
    async def subscribe_trades(self, callback):
        """Suscribirse a actualizaciones de trades propios"""
        if not self.ws_private:
            self.ws_private = await websockets.connect(f"{self.ws_url}/{self.listen_key}")
            self._ws_tasks.append(
                asyncio.create_task(self._handle_ws_messages(self.ws_private, callback, 'trades'))
            )
            
    # Métodos privados
    
    def _map_order_type(self, order_type: str) -> str:
        """Mapear tipo de orden al formato de Binance"""
        mapping = {
            'limit': 'LIMIT',
            'market': 'MARKET',
            'stop': 'STOP_LOSS',
            'stop_limit': 'STOP_LOSS_LIMIT'
        }
        return mapping.get(order_type.lower(), order_type.upper())
        
    def _parse_order(self, data: Dict) -> Order:
        """Parsear respuesta de orden"""
        return Order(
            id=str(data['orderId']),
            symbol=data['symbol'],
            side=data['side'].lower(),
            type=data['type'].lower(),
            price=self._parse_decimal(data.get('price', 0)) if data.get('price') else None,
            quantity=self._parse_decimal(data['origQty']),
            status=self._map_order_status(data['status']),
            timestamp=datetime.fromtimestamp(data['time'] / 1000),
            filled=self._parse_decimal(data.get('executedQty', 0)),
            remaining=self._parse_decimal(data['origQty']) - self._parse_decimal(data.get('executedQty', 0))
        )
        
    def _map_order_status(self, status: str) -> str:
        """Mapear estado de orden"""
        mapping = {
            'NEW': 'open',
            'PARTIALLY_FILLED': 'open',
            'FILLED': 'filled',
            'CANCELED': 'cancelled',
            'REJECTED': 'cancelled',
            'EXPIRED': 'cancelled'
        }
        return mapping.get(status, status.lower())
        
    async def _handle_ws_messages(self, ws, callback, msg_type: str):
        """Manejar mensajes WebSocket"""
        try:
            async for message in ws:
                data = json.loads(message)
                
                if msg_type == 'ticker':
                    ticker = Ticker(
                        symbol=data['s'],
                        bid=self._parse_decimal(data['b']),
                        ask=self._parse_decimal(data['a']),
                        last=self._parse_decimal(data['c']),
                        volume=self._parse_decimal(data['v']),
                        timestamp=datetime.fromtimestamp(data['E'] / 1000)
                    )
                    await callback(ticker)
                    
                elif msg_type == 'orderbook':
                    # Actualización incremental del libro de órdenes
                    await callback(data)
                    
                elif msg_type == 'trades':
                    if data.get('e') == 'executionReport':
                        # Trade ejecutado
                        await callback(data)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket cerrado")
        except Exception as e:
            logger.error(f"Error en WebSocket: {e}")
            
    async def _keep_alive_listen_key(self):
        """Mantener vivo el listen key"""
        while True:
            try:
                await asyncio.sleep(30 * 60)  # 30 minutos
                
                await self._request(
                    'PUT',
                    f"{self.rest_url}/userDataStream",
                    headers={'X-MBX-APIKEY': self.api_key},
                    params={'listenKey': self.listen_key}
                )
                
                logger.debug("Listen key renovado")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error renovando listen key: {e}")