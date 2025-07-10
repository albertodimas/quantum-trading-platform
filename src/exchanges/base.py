"""
Clase base para conectores de exchanges
Define la interfaz común para todos los exchanges
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
import logging
from decimal import Decimal
import hmac
import hashlib
import time
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class OrderBook:
    """Libro de órdenes"""
    bids: List[Tuple[Decimal, Decimal]]  # [(precio, cantidad), ...]
    asks: List[Tuple[Decimal, Decimal]]
    timestamp: datetime
    
@dataclass
class Ticker:
    """Datos del ticker"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime

@dataclass
class Balance:
    """Balance de una moneda"""
    currency: str
    free: Decimal
    locked: Decimal
    total: Decimal

@dataclass
class Order:
    """Orden de trading"""
    id: str
    symbol: str
    side: str  # 'buy' o 'sell'
    type: str  # 'limit', 'market', etc.
    price: Optional[Decimal]
    quantity: Decimal
    status: str  # 'open', 'filled', 'cancelled'
    timestamp: datetime
    filled: Decimal = Decimal('0')
    remaining: Decimal = Decimal('0')

@dataclass
class Trade:
    """Trade ejecutado"""
    id: str
    order_id: str
    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime

class ExchangeError(Exception):
    """Error base para exchanges"""
    pass

class ExchangeBase(ABC):
    """Clase base abstracta para conectores de exchanges"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections = {}
        self.rate_limits = {}
        
    @abstractmethod
    async def connect(self):
        """Establecer conexión con el exchange"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Cerrar conexión con el exchange"""
        pass
        
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Obtener ticker de un símbolo"""
        pass
        
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Obtener libro de órdenes"""
        pass
        
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """Obtener balances de la cuenta"""
        pass
        
    @abstractmethod
    async def create_order(
        self, 
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Order:
        """Crear una nueva orden"""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancelar una orden"""
        pass
        
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Obtener información de una orden"""
        pass
        
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Obtener órdenes abiertas"""
        pass
        
    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Obtener historial de trades"""
        pass
        
    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback):
        """Suscribirse a actualizaciones del ticker"""
        pass
        
    @abstractmethod
    async def subscribe_order_book(self, symbol: str, callback):
        """Suscribirse a actualizaciones del libro de órdenes"""
        pass
        
    @abstractmethod
    async def subscribe_trades(self, callback):
        """Suscribirse a actualizaciones de trades propios"""
        pass
        
    # Métodos utilitarios comunes
    
    def _generate_signature(self, query_string: str) -> str:
        """Generar firma HMAC para autenticación"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def _get_timestamp(self) -> int:
        """Obtener timestamp en milisegundos"""
        return int(time.time() * 1000)
        
    async def _rate_limit_check(self, endpoint: str):
        """Verificar límites de rate"""
        if endpoint in self.rate_limits:
            last_call = self.rate_limits[endpoint]
            time_passed = time.time() - last_call
            if time_passed < 1:  # 1 llamada por segundo por defecto
                await asyncio.sleep(1 - time_passed)
        self.rate_limits[endpoint] = time.time()
        
    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """Realizar petición HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        # Rate limiting
        await self._rate_limit_check(url)
        
        # Firma si es necesario
        if signed:
            timestamp = self._get_timestamp()
            if params:
                params['timestamp'] = timestamp
            else:
                params = {'timestamp': timestamp}
                
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
        try:
            async with self.session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=data
            ) as response:
                if response.status >= 400:
                    error_data = await response.text()
                    raise ExchangeError(f"HTTP {response.status}: {error_data}")
                    
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise ExchangeError(f"Error de conexión: {str(e)}")
            
    def _parse_decimal(self, value: Any) -> Decimal:
        """Convertir valor a Decimal de forma segura"""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
        
    def _format_symbol(self, base: str, quote: str) -> str:
        """Formatear símbolo según el exchange"""
        # Implementación por defecto, cada exchange puede sobrescribir
        return f"{base}{quote}"
        
    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parsear símbolo en base y quote"""
        # Implementación por defecto para símbolos como BTCUSDT
        # Cada exchange puede sobrescribir según su formato
        if '/' in symbol:
            return symbol.split('/')
        # Asumiendo que los últimos 3-4 caracteres son la moneda quote
        if symbol.endswith('USDT'):
            return symbol[:-4], 'USDT'
        elif symbol.endswith('BTC'):
            return symbol[:-3], 'BTC'
        elif symbol.endswith('ETH'):
            return symbol[:-3], 'ETH'
        else:
            # Por defecto, asume USD
            return symbol[:-3], 'USD'