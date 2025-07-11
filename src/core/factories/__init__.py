"""
Enterprise Factory Patterns for Quantum Trading Platform.

This module provides advanced factory patterns for:
- Exchange connectors with configuration management
- Trading strategies with parameter optimization
- Risk managers with adaptive configurations
- Market data providers with failover support
- Plugin architectures with dynamic loading
"""

from .exchange_factory import ExchangeFactory, ExchangeBuilder
from .strategy_factory import StrategyFactory, StrategyBuilder
from .risk_factory import RiskManagerFactory, RiskManagerBuilder
from .data_factory import MarketDataFactory, DataProviderBuilder
from .plugin_factory import PluginFactory, PluginLoader
from .registry import FactoryRegistry, get_factory_registry
from .builders import (
    BaseBuilder,
    ConfigurableBuilder,
    ParameterizedBuilder,
    ChainableBuilder
)

__all__ = [
    # Exchange Factories
    "ExchangeFactory",
    "ExchangeBuilder",
    
    # Strategy Factories
    "StrategyFactory", 
    "StrategyBuilder",
    
    # Risk Management Factories
    "RiskManagerFactory",
    "RiskManagerBuilder",
    
    # Data Provider Factories
    "MarketDataFactory",
    "DataProviderBuilder",
    
    # Plugin System
    "PluginFactory",
    "PluginLoader",
    
    # Registry
    "FactoryRegistry",
    "get_factory_registry",
    
    # Builders
    "BaseBuilder",
    "ConfigurableBuilder", 
    "ParameterizedBuilder",
    "ChainableBuilder",
]