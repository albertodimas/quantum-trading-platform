"""
Advanced caching decorators for seamless function and method caching.

Provides intelligent caching decorators with:
- Automatic key generation from function arguments
- TTL and invalidation support
- Cache warming and preloading
- Trading-specific optimizations
- Performance monitoring integration
"""

import asyncio
import functools
import hashlib
import inspect
import json
from typing import Any, Callable, Optional, Union, Dict, List
from datetime import datetime, timedelta

from .cache_manager import get_cache_manager, CacheLevel


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict, 
                       namespace: Optional[str] = None) -> str:
    """
    Generate a deterministic cache key from function and arguments.
    
    Args:
        func: Function being cached
        args: Positional arguments
        kwargs: Keyword arguments  
        namespace: Optional namespace prefix
        
    Returns:
        Cache key string
    """
    # Create base key from function
    func_name = f"{func.__module__}.{func.__qualname__}"
    
    # Serialize arguments
    try:
        # Convert args to list for JSON serialization
        serializable_args = []
        for arg in args:
            if hasattr(arg, '__dict__'):
                # For objects, use their dict representation
                serializable_args.append(arg.__dict__)
            else:
                serializable_args.append(arg)
        
        # Sort kwargs for consistent ordering
        sorted_kwargs = {k: v for k, v in sorted(kwargs.items())}
        
        # Create argument signature
        arg_signature = json.dumps({
            'args': serializable_args,
            'kwargs': sorted_kwargs
        }, sort_keys=True, default=str)
        
        # Hash the signature for consistent key length
        arg_hash = hashlib.md5(arg_signature.encode()).hexdigest()
        
    except (TypeError, ValueError):
        # Fallback to string representation
        arg_str = f"{args}:{kwargs}"
        arg_hash = hashlib.md5(arg_str.encode()).hexdigest()
    
    # Combine components
    if namespace:
        return f"{namespace}:{func_name}:{arg_hash}"
    else:
        return f"{func_name}:{arg_hash}"


def cache(ttl: Optional[int] = None,
         namespace: Optional[str] = None,
         levels: Optional[List[CacheLevel]] = None,
         key_func: Optional[Callable] = None,
         condition: Optional[Callable] = None) -> Callable:
    """
    Cache decorator for functions and methods.
    
    Args:
        ttl: Time to live in seconds
        namespace: Cache namespace
        levels: Cache levels to use
        key_func: Custom key generation function
        condition: Function to determine if result should be cached
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(func, args, kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs, namespace)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, levels)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Check caching condition
            if condition and not condition(result):
                return result
            
            # Cache the result
            await cache_manager.set(cache_key, result, ttl, levels)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(func, args, kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs, namespace)
            
            # Try to get from cache (using asyncio for sync function)
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            cached_result = loop.run_until_complete(
                cache_manager.get(cache_key, levels)
            )
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check caching condition
            if condition and not condition(result):
                return result
            
            # Cache the result
            loop.run_until_complete(
                cache_manager.set(cache_key, result, ttl, levels)
            )
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_result(ttl: int = 3600, namespace: str = "results") -> Callable:
    """
    Simple result caching decorator.
    
    Args:
        ttl: Time to live in seconds
        namespace: Cache namespace
        
    Returns:
        Decorated function
    """
    return cache(ttl=ttl, namespace=namespace)


def invalidate_cache(namespace: Optional[str] = None,
                    key_pattern: Optional[str] = None) -> Callable:
    """
    Decorator to invalidate cache entries after function execution.
    
    Args:
        namespace: Cache namespace to invalidate
        key_pattern: Specific key pattern to invalidate
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function first
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache_manager = get_cache_manager()
            
            if key_pattern:
                # Invalidate specific pattern
                keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(key_pattern)
                for key in keys:
                    await cache_manager.delete(key)
            elif namespace:
                # Invalidate namespace
                pattern = f"{namespace}:*"
                keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(pattern)
                for key in keys:
                    await cache_manager.delete(key)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Execute function first
            result = func(*args, **kwargs)
            
            # Invalidate cache
            cache_manager = get_cache_manager()
            
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def invalidate():
                if key_pattern:
                    keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(key_pattern)
                    for key in keys:
                        await cache_manager.delete(key)
                elif namespace:
                    pattern = f"{namespace}:*"
                    keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(pattern)
                    for key in keys:
                        await cache_manager.delete(key)
            
            loop.run_until_complete(invalidate())
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Trading-specific decorators
def cache_market_data(symbol: str, ttl: int = 60) -> Callable:
    """
    Cache decorator specifically for market data.
    
    Args:
        symbol: Trading symbol
        ttl: Time to live in seconds (default 60s for market data)
        
    Returns:
        Decorated function
    """
    def key_generator(func, args, kwargs):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
        return f"market_data:{symbol}:{func.__name__}:{timestamp}"
    
    return cache(
        ttl=ttl,
        namespace="market_data",
        key_func=key_generator,
        levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
    )


def cache_strategy_result(strategy_name: str, ttl: int = 300) -> Callable:
    """
    Cache decorator for strategy calculations.
    
    Args:
        strategy_name: Strategy identifier
        ttl: Time to live in seconds (default 5 minutes)
        
    Returns:
        Decorated function
    """
    def key_generator(func, args, kwargs):
        # Include strategy parameters in key
        params_hash = hashlib.md5(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        return f"strategy:{strategy_name}:{func.__name__}:{params_hash}"
    
    def cache_condition(result):
        # Only cache successful strategy results
        return result is not None and (
            not isinstance(result, dict) or 
            result.get('status') == 'success'
        )
    
    return cache(
        ttl=ttl,
        namespace="strategies",
        key_func=key_generator,
        condition=cache_condition,
        levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
    )


def cache_risk_calculation(ttl: int = 120) -> Callable:
    """
    Cache decorator for risk calculations.
    
    Args:
        ttl: Time to live in seconds (default 2 minutes)
        
    Returns:
        Decorated function
    """
    def key_generator(func, args, kwargs):
        # Include portfolio state in key
        portfolio_hash = "default"
        if 'portfolio' in kwargs:
            portfolio_data = str(kwargs['portfolio'])
            portfolio_hash = hashlib.md5(portfolio_data.encode()).hexdigest()[:8]
        
        return f"risk:{func.__name__}:{portfolio_hash}"
    
    return cache(
        ttl=ttl,
        namespace="risk",
        key_func=key_generator,
        levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
    )


def cache_exchange_data(exchange_name: str, ttl: int = 30) -> Callable:
    """
    Cache decorator for exchange API data.
    
    Args:
        exchange_name: Exchange identifier
        ttl: Time to live in seconds (default 30s)
        
    Returns:
        Decorated function
    """
    def key_generator(func, args, kwargs):
        endpoint = func.__name__
        params_hash = hashlib.md5(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        return f"exchange:{exchange_name}:{endpoint}:{params_hash}"
    
    return cache(
        ttl=ttl,
        namespace="exchanges",
        key_func=key_generator,
        levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
    )


# Utility functions for manual cache management
async def warm_market_data_cache(symbols: List[str], data_types: List[str]):
    """
    Warm cache with market data for specified symbols.
    
    Args:
        symbols: List of trading symbols
        data_types: List of data types to cache
    """
    cache_manager = get_cache_manager()
    
    # This would typically fetch data from market data provider
    # and populate cache proactively
    for symbol in symbols:
        for data_type in data_types:
            cache_key = f"market_data:{symbol}:{data_type}"
            # Simulate fetching fresh data
            mock_data = {
                "symbol": symbol,
                "type": data_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"price": 100.0, "volume": 1000}
            }
            await cache_manager.set(cache_key, mock_data, ttl=60)


async def invalidate_symbol_cache(symbol: str):
    """
    Invalidate all cache entries for a specific symbol.
    
    Args:
        symbol: Trading symbol to invalidate
    """
    cache_manager = get_cache_manager()
    
    # Invalidate market data
    pattern = f"market_data:{symbol}:*"
    if CacheLevel.L2_REDIS in cache_manager.backends:
        keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(pattern)
        for key in keys:
            await cache_manager.delete(key)
    
    # Invalidate strategy data
    pattern = f"strategy:*:{symbol}:*"
    if CacheLevel.L2_REDIS in cache_manager.backends:
        keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(pattern)
        for key in keys:
            await cache_manager.delete(key)


async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get comprehensive cache performance statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    cache_manager = get_cache_manager()
    stats = cache_manager.get_stats()
    
    # Add additional trading-specific metrics
    trading_stats = {
        "market_data_cache_size": 0,
        "strategy_cache_size": 0,
        "risk_cache_size": 0,
    }
    
    # Count entries by namespace
    if CacheLevel.L2_REDIS in cache_manager.backends:
        for namespace in ["market_data", "strategies", "risk"]:
            pattern = f"{namespace}:*"
            keys = await cache_manager.backends[CacheLevel.L2_REDIS].keys(pattern)
            trading_stats[f"{namespace}_cache_size"] = len(keys)
    
    stats.update(trading_stats)
    return stats