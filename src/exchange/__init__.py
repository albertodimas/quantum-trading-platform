"""
Exchange Integration Module

Provides connectivity to various cryptocurrency exchanges with a unified interface.
"""

from .exchange_interface import (
    ExchangeInterface,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Order,
    Trade,
    Position,
    MarketData,
    OrderBook,
    Ticker,
    Balance
)
from .binance_exchange import BinanceExchange
from .exchange_manager import ExchangeManager

__all__ = [
    'ExchangeInterface',
    'BinanceExchange',
    'ExchangeManager',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'TimeInForce',
    'Order',
    'Trade',
    'Position',
    'MarketData',
    'OrderBook',
    'Ticker',
    'Balance'
]