"""
Módulo de Conectores de Exchanges
Gestiona las conexiones con exchanges de criptomonedas reales
"""

from .base import ExchangeBase, ExchangeError
from .binance_connector import BinanceConnector
from .kraken_connector import KrakenConnector
from .coinbase_connector import CoinbaseConnector
from .factory import ExchangeFactory

__all__ = [
    'ExchangeBase',
    'ExchangeError',
    'BinanceConnector',
    'KrakenConnector',
    'CoinbaseConnector',
    'ExchangeFactory'
]