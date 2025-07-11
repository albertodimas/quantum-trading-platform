"""
🏗️ Core Architecture Module
Patrones arquitectónicos enterprise para el Quantum Trading Platform
"""

from .base_repository import BaseRepository
from .unit_of_work import UnitOfWork
from .dependency_injection import DIContainer, injectable, inject
from .event_bus import EventBus, DomainEvent, EventHandler
from .factory_registry import FactoryRegistry
from .circuit_breaker import CircuitBreaker
from .rate_limiter import RateLimiter, RateLimitConfig, RateLimiterRegistry
from .cache_manager import CacheManager
from .configuration_manager import ConfigurationManager
from .metrics_collector import MetricsCollector

__all__ = [
    'BaseRepository',
    'UnitOfWork', 
    'DIContainer',
    'injectable',
    'inject',
    'EventBus',
    'DomainEvent',
    'EventHandler',
    'FactoryRegistry',
    'CircuitBreaker',
    'RateLimiter',
    'RateLimitConfig',
    'RateLimiterRegistry',
    'CacheManager',
    'ConfigurationManager',
    'MetricsCollector'
]