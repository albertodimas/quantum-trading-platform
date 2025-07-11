"""
Enterprise Redis Cache System for Quantum Trading Platform.

This module provides advanced caching capabilities including:
- Distributed Redis cache with clustering support
- Intelligent cache invalidation strategies
- Multi-tier caching (L1 memory + L2 Redis)
- Cache warming and preloading
- Performance analytics and monitoring
- Serialization optimization for trading data
"""

from .cache_manager import CacheManager, get_cache_manager
from .redis_backend import RedisBackend, RedisClusterBackend
from .memory_backend import MemoryBackend
from .serializers import (
    JsonSerializer,
    PickleSerializer, 
    CompressionSerializer,
    TradingDataSerializer
)
from .invalidation import (
    CacheInvalidator,
    InvalidationStrategy,
    TimeBasedInvalidation,
    EventBasedInvalidation,
    DependencyBasedInvalidation
)
from .warming import CacheWarmer, WarmingStrategy
from .decorators import cache, cache_result, invalidate_cache
from .analytics import CacheAnalytics, CacheMetrics

__all__ = [
    # Core
    "CacheManager",
    "get_cache_manager",
    
    # Backends
    "RedisBackend",
    "RedisClusterBackend", 
    "MemoryBackend",
    
    # Serialization
    "JsonSerializer",
    "PickleSerializer",
    "CompressionSerializer", 
    "TradingDataSerializer",
    
    # Invalidation
    "CacheInvalidator",
    "InvalidationStrategy",
    "TimeBasedInvalidation",
    "EventBasedInvalidation", 
    "DependencyBasedInvalidation",
    
    # Warming
    "CacheWarmer",
    "WarmingStrategy",
    
    # Decorators
    "cache",
    "cache_result",
    "invalidate_cache",
    
    # Analytics
    "CacheAnalytics",
    "CacheMetrics",
]