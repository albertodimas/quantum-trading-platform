"""
Módulo de Backtesting
Sistema completo para pruebas históricas de estrategias de trading
"""

from .engine import BacktestEngine
from .data_handler import HistoricalDataHandler
from .portfolio import Portfolio
from .metrics import PerformanceMetrics
from .reports import ReportGenerator
from .strategy_runner import StrategyRunner

__all__ = [
    'BacktestEngine',
    'HistoricalDataHandler',
    'Portfolio',
    'PerformanceMetrics',
    'ReportGenerator',
    'StrategyRunner'
]