"""
Backtesting Framework for Quantum Trading Platform

This module provides a comprehensive backtesting engine for trading strategies,
allowing historical simulation and performance analysis.
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .data_provider import (
    DataProvider,
    HistoricalDataProvider,
    CSVDataProvider,
    DatabaseDataProvider
)
from .event_simulator import EventSimulator, MarketEvent, TimeEvent
from .portfolio_simulator import PortfolioSimulator, SimulatedPortfolio
from .execution_simulator import ExecutionSimulator, FillModel, SimulatedFill
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'DataProvider',
    'HistoricalDataProvider',
    'CSVDataProvider',
    'DatabaseDataProvider',
    'EventSimulator',
    'MarketEvent',
    'TimeEvent',
    'PortfolioSimulator',
    'SimulatedPortfolio',
    'ExecutionSimulator',
    'FillModel',
    'SimulatedFill',
    'PerformanceAnalyzer',
    'PerformanceMetrics',
]