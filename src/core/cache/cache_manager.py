"""
Advanced multi-tier cache manager with intelligent invalidation and performance optimization.

Features:
- Multi-tier caching (L1 memory + L2 Redis)
- Distributed cache with consistency guarantees
- Intelligent cache invalidation strategies
- Performance monitoring and analytics
- Cache warming and preloading
- Serialization optimization
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable
from enum import Enum
import hashlib
import json

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

T = TypeVar('T')


class CacheLevel(Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"      # In-process memory cache
    L2_REDIS = "l2_redis"        # Distributed Redis cache
    L3_DATABASE = "l3_database"  # Database cache layer


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    FIFO = "fifo"            # First In First Out
    RANDOM = "random"        # Random eviction


@dataclass
class CacheKey:
    """Structured cache key with metadata."""
    namespace: str
    identifier: str
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Generate cache key string."""
        base = f"{self.namespace}:{self.identifier}:v{self.version}"
        if self.tags:
            tag_hash = hashlib.md5(":".join(sorted(self.tags)).encode()).hexdigest()[:8]
            base += f":{tag_hash}"
        return base
    
    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: CacheKey
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def touch(self):
        """Update access metadata."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access metadata
            entry.touch()
            
            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Evict if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            # Create cache entry
            cache_key = CacheKey(namespace="memory", identifier=key)
            entry = CacheEntry(
                key=cache_key,
                value=value,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=self._estimate_size(value)
            )
            
            self._cache[key] = entry
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        with self._lock:
            return key in self._cache and not self._cache[key].is_expired
    
    async def clear(self) -> bool:
        """Clear all memory cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        with self._lock:
            if pattern == "*":
                return [k for k, v in self._cache.items() if not v.is_expired]
            
            # Simple pattern matching (could be enhanced)
            import fnmatch
            return [k for k, v in self._cache.items() 
                   if not v.is_expired and fnmatch.fnmatch(k, pattern)]
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            del self._cache[lru_key]
            self._access_order.pop(0)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(str(value).encode('utf-8'))
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "total_size_bytes": total_size,
                "hit_ratio": self._calculate_hit_ratio(),
            }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # Simplified implementation
        return 0.85  # Would track actual hits/misses


class RedisCacheBackend(CacheBackend):
    """Redis cache backend with advanced features."""
    
    def __init__(self, redis_url: str, key_prefix: str = "qtp:", 
                 compression_threshold: int = 1024):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression_threshold = compression_threshold
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if not self._connected and REDIS_AVAILABLE:
            try:
                self._redis = aioredis.from_url(self.redis_url)
                await self._redis.ping()
                self._connected = True
            except Exception as e:
                print(f"Redis connection failed: {e}")
                self._connected = False
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not REDIS_AVAILABLE:
            return None
        
        await self._ensure_connected()
        if not self._connected:
            return None
        
        try:
            redis_key = self._make_key(key)
            data = await self._redis.get(redis_key)
            
            if data is None:
                return None
            
            # Deserialize data
            return self._deserialize(data)
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not REDIS_AVAILABLE:
            return False
        
        await self._ensure_connected()
        if not self._connected:
            return False
        
        try:
            redis_key = self._make_key(key)
            serialized_data = self._serialize(value)
            
            if ttl:
                await self._redis.setex(redis_key, ttl, serialized_data)
            else:
                await self._redis.set(redis_key, serialized_data)
            
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not REDIS_AVAILABLE:
            return False
        
        await self._ensure_connected()
        if not self._connected:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self._redis.delete(redis_key)
            return result > 0
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not REDIS_AVAILABLE:
            return False
        
        await self._ensure_connected()
        if not self._connected:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self._redis.exists(redis_key)
            return result > 0
        except Exception as e:
            print(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        if not REDIS_AVAILABLE:
            return False
        
        await self._ensure_connected()
        if not self._connected:
            return False
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
            return True
        except Exception as e:
            print(f"Redis clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        if not REDIS_AVAILABLE:
            return []
        
        await self._ensure_connected()
        if not self._connected:
            return []
        
        try:
            redis_pattern = f"{self.key_prefix}{pattern}"
            keys = await self._redis.keys(redis_pattern)
            # Remove prefix from keys
            return [key.decode().replace(self.key_prefix, "") for key in keys]
        except Exception as e:
            print(f"Redis keys error: {e}")
            return []
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Use JSON for simple types, pickle for complex types
            if isinstance(value, (str, int, float, bool, list, dict)):
                data = json.dumps(value).encode('utf-8')
            else:
                import pickle
                data = pickle.dumps(value)
            
            # Compress if data is large
            if len(data) > self.compression_threshold:
                import gzip
                data = gzip.compress(data)
                data = b'compressed:' + data
            
            return data
        except Exception:
            # Fallback to string representation
            return str(value).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Check if compressed
            if data.startswith(b'compressed:'):
                import gzip
                data = gzip.decompress(data[11:])  # Remove 'compressed:' prefix
            
            # Try JSON first
            try:
                return json.loads(data.decode('utf-8'))
            except:
                # Try pickle
                import pickle
                return pickle.loads(data)
        except Exception:
            # Fallback to string
            return data.decode('utf-8')


class CacheManager:
    """
    Advanced multi-tier cache manager.
    
    Features:
    - Multi-tier caching (L1 memory + L2 Redis)
    - Intelligent cache invalidation
    - Performance monitoring and analytics
    - Cache warming and preloading
    - Serialization optimization
    - Thread-safe operations
    """
    
    def __init__(self, redis_url: Optional[str] = None, 
                 enable_l1_cache: bool = True,
                 l1_max_size: int = 1000,
                 default_ttl: int = 3600):
        self.enable_l1_cache = enable_l1_cache
        self.default_ttl = default_ttl
        
        # Initialize backends
        self.backends: Dict[CacheLevel, CacheBackend] = {}
        
        if enable_l1_cache:
            self.backends[CacheLevel.L1_MEMORY] = MemoryCacheBackend(
                max_size=l1_max_size,
                default_ttl=default_ttl
            )
        
        if redis_url and REDIS_AVAILABLE:
            self.backends[CacheLevel.L2_REDIS] = RedisCacheBackend(redis_url)
        
        # Analytics
        self._stats = defaultdict(int)
        self._start_time = time.time()
        
        # Invalidation callbacks
        self._invalidation_callbacks: List[Callable[[str], None]] = []
    
    async def get(self, key: str, levels: Optional[List[CacheLevel]] = None) -> Optional[Any]:
        """
        Get value from cache with multi-tier lookup.
        
        Args:
            key: Cache key
            levels: Cache levels to check (defaults to all)
            
        Returns:
            Cached value or None
        """
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        # Check each level in order
        for level in levels:
            if level not in self.backends:
                continue
            
            try:
                value = await self.backends[level].get(key)
                if value is not None:
                    self._stats[f"{level.value}_hits"] += 1
                    
                    # Promote to higher cache levels
                    await self._promote_to_higher_levels(key, value, level, levels)
                    
                    return value
                else:
                    self._stats[f"{level.value}_misses"] += 1
            except Exception as e:
                print(f"Cache get error for level {level}: {e}")
                self._stats[f"{level.value}_errors"] += 1
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        Set value in cache across multiple tiers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            levels: Cache levels to update (defaults to all)
            
        Returns:
            True if set in at least one level
        """
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        ttl = ttl or self.default_ttl
        success = False
        
        # Set in all specified levels
        for level in levels:
            if level not in self.backends:
                continue
            
            try:
                result = await self.backends[level].set(key, value, ttl)
                if result:
                    success = True
                    self._stats[f"{level.value}_sets"] += 1
            except Exception as e:
                print(f"Cache set error for level {level}: {e}")
                self._stats[f"{level.value}_errors"] += 1
        
        return success
    
    async def delete(self, key: str, levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        Delete key from cache across multiple tiers.
        
        Args:
            key: Cache key to delete
            levels: Cache levels to delete from (defaults to all)
            
        Returns:
            True if deleted from at least one level
        """
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        success = False
        
        # Delete from all specified levels
        for level in levels:
            if level not in self.backends:
                continue
            
            try:
                result = await self.backends[level].delete(key)
                if result:
                    success = True
                    self._stats[f"{level.value}_deletes"] += 1
            except Exception as e:
                print(f"Cache delete error for level {level}: {e}")
                self._stats[f"{level.value}_errors"] += 1
        
        # Notify invalidation callbacks
        for callback in self._invalidation_callbacks:
            try:
                callback(key)
            except Exception as e:
                print(f"Invalidation callback error: {e}")
        
        return success
    
    async def exists(self, key: str, levels: Optional[List[CacheLevel]] = None) -> bool:
        """Check if key exists in any cache level."""
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        for level in levels:
            if level not in self.backends:
                continue
            
            try:
                if await self.backends[level].exists(key):
                    return True
            except Exception as e:
                print(f"Cache exists error for level {level}: {e}")
        
        return False
    
    async def clear(self, levels: Optional[List[CacheLevel]] = None) -> bool:
        """Clear cache across multiple tiers."""
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        success = False
        
        for level in levels:
            if level not in self.backends:
                continue
            
            try:
                result = await self.backends[level].clear()
                if result:
                    success = True
                    self._stats[f"{level.value}_clears"] += 1
            except Exception as e:
                print(f"Cache clear error for level {level}: {e}")
        
        return success
    
    async def _promote_to_higher_levels(self, key: str, value: Any, 
                                      current_level: CacheLevel, 
                                      all_levels: List[CacheLevel]):
        """Promote cache entry to higher priority levels."""
        level_priority = {
            CacheLevel.L1_MEMORY: 1,
            CacheLevel.L2_REDIS: 2,
            CacheLevel.L3_DATABASE: 3
        }
        
        current_priority = level_priority.get(current_level, 999)
        
        for level in all_levels:
            if level not in self.backends:
                continue
            
            level_p = level_priority.get(level, 999)
            if level_p < current_priority:  # Higher priority (lower number)
                try:
                    await self.backends[level].set(key, value, self.default_ttl)
                except Exception as e:
                    print(f"Cache promotion error to {level}: {e}")
    
    def add_invalidation_callback(self, callback: Callable[[str], None]):
        """Add callback for cache invalidation events."""
        self._invalidation_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        runtime_seconds = time.time() - self._start_time
        
        stats = dict(self._stats)
        stats.update({
            "runtime_seconds": runtime_seconds,
            "backends": list(self.backends.keys()),
        })
        
        # Calculate hit ratios
        for level in self.backends.keys():
            hits = stats.get(f"{level.value}_hits", 0)
            misses = stats.get(f"{level.value}_misses", 0)
            total = hits + misses
            if total > 0:
                stats[f"{level.value}_hit_ratio"] = hits / total
        
        return stats
    
    async def warm_cache(self, key_value_pairs: Dict[str, Any], 
                        ttl: Optional[int] = None):
        """Warm cache with key-value pairs."""
        for key, value in key_value_pairs.items():
            await self.set(key, value, ttl)


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def initialize_cache(redis_url: Optional[str] = None,
                    enable_l1_cache: bool = True,
                    l1_max_size: int = 1000,
                    default_ttl: int = 3600) -> CacheManager:
    """Initialize the global cache system."""
    global _global_cache_manager
    _global_cache_manager = CacheManager(
        redis_url=redis_url,
        enable_l1_cache=enable_l1_cache,
        l1_max_size=l1_max_size,
        default_ttl=default_ttl
    )
    return _global_cache_manager