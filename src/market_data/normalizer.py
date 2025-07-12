"""
Data Normalizer for Market Data

Normalizes market data from different exchanges into a unified format.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from decimal import Decimal

from ..core.observability import get_logger

logger = get_logger(__name__)

@dataclass
class NormalizedMarketData:
    """Base class for normalized market data"""
    exchange: str
    symbol: str
    timestamp: datetime
    received_at: datetime

@dataclass
class NormalizedTicker(NormalizedMarketData):
    """Normalized ticker data"""
    bid: Optional[float] = None
    ask: Optional[float] = None
    price: Optional[float] = None
    mid_price: Optional[float] = None
    volume_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_pct_24h: Optional[float] = None

@dataclass
class NormalizedTrade(NormalizedMarketData):
    """Normalized trade data"""
    trade_id: str
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    taker_order_id: Optional[str] = None
    maker_order_id: Optional[str] = None@dataclass
class NormalizedOrderbook(NormalizedMarketData):
    """Normalized orderbook data"""
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    sequence: Optional[int] = None
    
class DataNormalizer:
    """
    Normalizes market data from various exchange formats
    """
    
    def __init__(self):
        self.logger = logger
        
        # Exchange-specific normalizers
        self.normalizers = {
            'binance': self._normalize_binance,
            'coinbase': self._normalize_coinbase,
            'kraken': self._normalize_kraken,
            'bitfinex': self._normalize_bitfinex,
            'okx': self._normalize_okx,
            'bybit': self._normalize_bybit,
        }
        
    async def normalize(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Union[NormalizedTicker, NormalizedTrade, NormalizedOrderbook, List[Any]]:
        """Normalize market data based on exchange and type"""
        normalizer = self.normalizers.get(exchange.lower())
        
        if not normalizer:
            self.logger.warning(f"No normalizer for exchange: {exchange}")
            return self._generic_normalize(exchange, symbol, data_type, data, timestamp)
            
        return await normalizer(symbol, data_type, data, timestamp)        
    async def _normalize_binance(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Normalize Binance market data"""
        if data_type == 'ticker':
            return NormalizedTicker(
                exchange='binance',
                symbol=symbol,
                timestamp=timestamp,
                received_at=datetime.now(),
                bid=float(data.get('bidPrice', 0)) if data.get('bidPrice') else None,
                ask=float(data.get('askPrice', 0)) if data.get('askPrice') else None,
                price=float(data.get('lastPrice', 0)) if data.get('lastPrice') else None,
                volume_24h=float(data.get('volume', 0)) if data.get('volume') else None,
                high_24h=float(data.get('highPrice', 0)) if data.get('highPrice') else None,
                low_24h=float(data.get('lowPrice', 0)) if data.get('lowPrice') else None,
                change_24h=float(data.get('priceChange', 0)) if data.get('priceChange') else None,
                change_pct_24h=float(data.get('priceChangePercent', 0)) if data.get('priceChangePercent') else None
            )
            
        elif data_type == 'trades':
            if isinstance(data, list):
                trades = []
                for trade in data:
                    trades.append(NormalizedTrade(
                        exchange='binance',
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(trade['time'] / 1000),
                        received_at=datetime.now(),
                        trade_id=str(trade['id']),
                        price=float(trade['price']),
                        quantity=float(trade['qty']),
                        side='sell' if trade['isBuyerMaker'] else 'buy'
                    ))
                return trades            else:
                return NormalizedTrade(
                    exchange='binance',
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(data['time'] / 1000),
                    received_at=datetime.now(),
                    trade_id=str(data['id']),
                    price=float(data['price']),
                    quantity=float(data['qty']),
                    side='sell' if data['isBuyerMaker'] else 'buy'
                )
                
        elif data_type == 'orderbook':
            return NormalizedOrderbook(
                exchange='binance',
                symbol=symbol,
                timestamp=timestamp,
                received_at=datetime.now(),
                bids=[[float(p), float(q)] for p, q in data.get('bids', [])],
                asks=[[float(p), float(q)] for p, q in data.get('asks', [])],
                sequence=data.get('lastUpdateId')
            )
            
        return data        
    async def _normalize_coinbase(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Normalize Coinbase market data"""
        if data_type == 'ticker':
            return NormalizedTicker(
                exchange='coinbase',
                symbol=symbol,
                timestamp=timestamp,
                received_at=datetime.now(),
                bid=float(data.get('best_bid', 0)) if data.get('best_bid') else None,
                ask=float(data.get('best_ask', 0)) if data.get('best_ask') else None,
                price=float(data.get('price', 0)) if data.get('price') else None,
                volume_24h=float(data.get('volume_24h', 0)) if data.get('volume_24h') else None,
                high_24h=float(data.get('high_24h', 0)) if data.get('high_24h') else None,
                low_24h=float(data.get('low_24h', 0)) if data.get('low_24h') else None
            )
            
        elif data_type == 'trades':
            return NormalizedTrade(
                exchange='coinbase',
                symbol=symbol,
                timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
                received_at=datetime.now(),
                trade_id=str(data['trade_id']),
                price=float(data['price']),
                quantity=float(data['size']),
                side=data['side']
            )            
        elif data_type == 'orderbook':
            return NormalizedOrderbook(
                exchange='coinbase',
                symbol=symbol,
                timestamp=timestamp,
                received_at=datetime.now(),
                bids=[[float(p), float(q), 0] for p, q, _ in data.get('bids', [])],
                asks=[[float(p), float(q), 0] for p, q, _ in data.get('asks', [])],
                sequence=data.get('sequence')
            )
            
        return data
        
    async def _normalize_kraken(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Normalize Kraken market data"""
        # Kraken-specific normalization
        return self._generic_normalize('kraken', symbol, data_type, data, timestamp)
        
    async def _normalize_bitfinex(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Normalize Bitfinex market data"""
        # Bitfinex-specific normalization
        return self._generic_normalize('bitfinex', symbol, data_type, data, timestamp)        
    async def _normalize_okx(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Normalize OKX market data"""
        # OKX-specific normalization
        return self._generic_normalize('okx', symbol, data_type, data, timestamp)
        
    async def _normalize_bybit(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Normalize Bybit market data"""
        # Bybit-specific normalization
        return self._generic_normalize('bybit', symbol, data_type, data, timestamp)
        
    def _generic_normalize(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ) -> Any:
        """Generic normalization for unknown exchanges"""
        base_data = {
            'exchange': exchange,
            'symbol': symbol,
            'timestamp': timestamp,
            'received_at': datetime.now()
        }        
        if data_type == 'ticker':
            # Try to extract common fields
            ticker_data = {**base_data}
            
            # Common field mappings
            field_mappings = {
                'bid': ['bid', 'bidPrice', 'best_bid', 'bid_price'],
                'ask': ['ask', 'askPrice', 'best_ask', 'ask_price'],
                'price': ['price', 'last', 'lastPrice', 'last_price'],
                'volume': ['volume', 'vol', 'volume_24h', 'baseVolume']
            }
            
            for target_field, possible_fields in field_mappings.items():
                for field in possible_fields:
                    if field in data and data[field]:
                        try:
                            ticker_data[target_field] = float(data[field])
                            break
                        except (ValueError, TypeError):
                            continue
                            
            return NormalizedTicker(**ticker_data)
            
        elif data_type == 'trades':
            # Generic trade normalization
            return NormalizedTrade(
                **base_data,
                trade_id=str(data.get('id', data.get('trade_id', ''))),
                price=float(data.get('price', 0)),
                quantity=float(data.get('amount', data.get('quantity', data.get('size', 0)))),
                side=data.get('side', 'unknown')
            )
            
        elif data_type == 'orderbook':
            # Generic orderbook normalization
            return NormalizedOrderbook(
                **base_data,
                bids=data.get('bids', []),
                asks=data.get('asks', [])
            )
            
        return data