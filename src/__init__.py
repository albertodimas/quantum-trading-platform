"""
Quantum Trading Platform

Professional AI-powered algorithmic trading platform.
"""

__version__ = "0.1.0"
__author__ = "Quantum Trading Team"
__email__ = "team@quantumtrading.io"

# Public API
from src.core.config import settings
from src.trading.bot import TradingBot
from src.trading.strategy import Strategy

__all__ = ["settings", "TradingBot", "Strategy"]