"""
Enhanced caching strategy for app/utils/cache.py
This implementation provides more efficient caching with TTL management
and result serialization for better performance.
"""

import functools
import hashlib
import json
import logging
import time
import pickle
import threading
from typing import Any, Callable, Dict, Optional, TypeVar, cast, List, Set, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Define a type variable for the function return type
T = TypeVar('T')

class CacheManager:
    """
    Enhanced cache manager for expensive operations.
    Features:
    - Result serialization for any type of object
    - TTL management with background cleanup
    - Thread-safe operations
    - Detailed statistics and monitoring
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance of the cache manager."""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the cache manager with performance optimizations.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Default time to live in seconds
        """
        if self._initialized:
            return
            
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = ttl
        self._lock = threading.RLock()  # Reentrant lock for better performance
        
        # Add performance metrics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "additions": 0,
            "expired": 0,
            "size": 0
        }
        
        # Last access timestamps for LRU eviction
        self._last_access = {}
        
        # Setup background cleanup thread
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        self._initialized = True
        logger.info(f"Enhanced cache manager initialized with max_size={max_size}, ttl={ttl}")
    
    def _cleanup_worker(self):
        """Background worker to clean up expired cache entries."""
        logger.info("Cache cleanup worker started")
        cleanup_interval = 60  # Check every minute
        
        while not self._stop_cleanup.wait(cleanup_interval):
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
        
        logger.info("Cache cleanup worker stopped")
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired_keys = []
        
        with self._lock:
            # Find expired keys
            for key, entry in self._cache.items():
                if now > entry.get("expires_at", 0):
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
                if key in self._last_access:
                    del self._last_access[key]
            
            # Update stats
            self._stats["expired"] += len(expired_keys)
            self._stats["size"] = len(self._cache)
        
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired cache entries")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from function arguments with improved serialization.
        
        Args:
            prefix: Prefix for the key (usually function name)
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            str: MD5 hash of the arguments
        """
        # Create a consistent representation of the arguments
        key_parts = [prefix]
        
        # Handle args
        for arg in args:
            try:
                # Try to use the object's __repr__ for better uniqueness
                if hasattr(arg, '__dict__'):
                    # For objects, use their attributes
                    key_parts.append(str(sorted(arg.__dict__.items())))
                else:
                    key_parts.append(repr(arg))
            except:
                # Fallback for non-serializable objects
                key_parts.append(str(id(arg)))
        
        # Handle kwargs
        sorted_kwargs = sorted(kwargs.items())
        for k, v in sorted_kwargs:
            try:
                if hasattr(v, '__dict__'):
                    key_parts.append(f"{k}:{sorted(v.__dict__.items())}")
                else:
                    key_parts.append(f"{k}:{repr(v)}")
            except:
                key_parts.append(f"{k}:{str(id(v))}")
        
        # Create a string and hash it
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache with TTL checking.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            cache_entry = self._cache[key]
            
            # Check if entry is expired
            if time.time() > cache_entry["expires_at"]:
                del self._cache[key]
                if key in self._last_access:
                    del self._last_access[key]
                self._stats["misses"] += 1
                self._stats["expired"] += 1
                self._stats["size"] = len(self._cache)
                return None
            
            # Update last access time for LRU
            self._last_access[key] = time.time()
            
            # Update hits counter
            self._stats["hits"] += 1
            
            # Return deserialized value
            try:
                value = cache_entry["value"]
                if cache_entry.get("serialized", False):
                    value = pickle.loads(value)
                return value
            except Exception as e:
                logger.error(f"Error deserializing cached value: {e}")
                # Count as miss on deserialization error
                self._stats["misses"] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache with optimized serialization.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom time to live in seconds
        """
        # Make room if cache is full
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_entries(1)
            
            # Calculate expiration time
            expires_at = time.time() + (ttl or self._default_ttl)
            
            # Serialize complex objects to ensure cache integrity
            serialized = False
            try:
                # Try JSON serialization first as it's faster
                json.dumps(value)
                # Value is JSON serializable, store as is
                serialized_value = value
            except (TypeError, ValueError):
                # For complex objects, use pickle
                try:
                    serialized_value = pickle.dumps(value)
                    serialized = True
                except Exception as e:
                    logger.error(f"Error serializing value for cache: {e}")
                    # Fall back to storing the original value
                    serialized_value = value
                    serialized = False
            
            # Add entry to cache
            self._cache[key] = {
                "value": serialized_value,
                "serialized": serialized,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            
            # Update last access time for LRU
            self._last_access[key] = time.time()
            
            # Update stats
            self._stats["additions"] += 1
            self._stats["size"] = len(self._cache)
    
    def _evict_entries(self, count: int = 1) -> None:
        """
        Evict entries using LRU strategy.
        
        Args:
            count: Number of entries to evict
        """
        if not self._last_access:
            return
        
        # Sort keys by last access time (oldest first)
        sorted_keys = sorted(self._last_access.items(), key=lambda x: x[1])
        
        # Evict the specified number of entries
        for i in range(min(count, len(sorted_keys))):
            key = sorted_keys[i][0]
            if key in self._cache:
                del self._cache[key]
            if key in self._last_access:
                del self._last_access[key]
            self._stats["evictions"] += 1
        
        self._stats["size"] = len(self._cache)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._last_access.clear()
            self._stats["size"] = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        with self._lock:
            hit_rate = 0.0
            total_requests = self._stats["hits"] + self._stats["misses"]
            if total_requests > 0:
                hit_rate = self._stats["hits"] / total_requests
            
            return {
                "size": self._stats["size"],
                "max_size": self._max_size,
                "ttl": self._default_ttl,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "additions": self._stats["additions"],
                "expired": self._stats["expired"],
                "memory_usage_estimate": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the cache in bytes.
        
        Returns:
            int: Estimated memory usage in bytes
        """
        total_size = 0
        sample_size = min(10, len(self._cache))
        
        if sample_size == 0:
            return 0
        
        # Sample a few entries to estimate average size
        sampled_keys = list(self._cache.keys())[:sample_size]
        for key in sampled_keys:
            entry = self._cache[key]
            try:
                if entry.get("serialized", False):
                    entry_size = len(entry["value"])
                else:
                    # For non-serialized values, use pickle to estimate size
                    entry_size = len(pickle.dumps(entry["value"]))
                total_size += entry_size
            except:
                # Fallback estimation
                total_size += 1024  # Assume 1KB per entry
        
        # Extrapolate to full cache
        avg_entry_size = total_size / sample_size
        return int(avg_entry_size * len(self._cache))
    
    def cached(self, ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Optimized decorator for caching function results.
        
        Args:
            ttl: Custom time to live in seconds
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Generate cache key
                key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Check if result is in cache
                cached_result = self.get(key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Call original function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Only cache results for functions that take significant time
                # This avoids caching overhead for very fast functions
                if execution_time > 0.001:  # Only cache if execution took more than 1ms
                    self.set(key, result, ttl)
                    logger.debug(f"Cache miss for {func.__name__} (execution time: {execution_time:.3f}s)")
                
                return result
            return cast(Callable[..., T], wrapper)
        return decorator
    
    def async_cached(self, ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Optimized decorator for caching async function results.
        
        Args:
            ttl: Custom time to live in seconds
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # Generate cache key
                key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Check if result is in cache
                cached_result = self.get(key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for async {func.__name__}")
                    return cached_result
                
                # Call original function
                import asyncio
                start_time = asyncio.get_event_loop().time()
                result = await func(*args, **kwargs)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Only cache results for functions that take significant time
                if execution_time > 0.005:  # Only cache if execution took more than 5ms
                    self.set(key, result, ttl)
                    logger.debug(f"Cache miss for async {func.__name__} (execution time: {execution_time:.3f}s)")
                
                return result
            return cast(Callable[..., T], wrapper)
        return decorator
    
    def stop(self):
        """Stop the cache manager and cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("Cache manager stopped")

# Create a global instance for application-wide use
cache_manager = CacheManager()

def setup_caching(max_size: int = 5000, ttl: int = 7200) -> None:
    """
    Initialize the cache manager with improved defaults.
    
    Args:
        max_size: Maximum number of items in the cache (increased from 1000 to 5000)
        ttl: Time to live in seconds (increased from 3600 to 7200)
    """
    global cache_manager
    # Stop existing cache manager if initialized
    if hasattr(cache_manager, 'stop') and callable(cache_manager.stop):
        cache_manager.stop()
    
    cache_manager = CacheManager(max_size=max_size, ttl=ttl)
    logger.info(f"Enhanced cache initialized with max_size={max_size}, ttl={ttl}")

def get_cache_stats() -> Dict[str, Any]:
    """
    Get detailed cache statistics.
    
    Returns:
        Dict with cache statistics
    """
    return cache_manager.get_stats()

def invalidate_key(key_prefix: str) -> int:
    """
    Invalidate cache entries by key prefix.
    
    Args:
        key_prefix: Prefix to match for cache keys
        
    Returns:
        int: Number of invalidated entries
    """
    count = 0
    with cache_manager._lock:
        keys_to_remove = [k for k in cache_manager._cache.keys() if k.startswith(key_prefix)]
        for k in keys_to_remove:
            del cache_manager._cache[k]
            if k in cache_manager._last_access:
                del cache_manager._last_access[k]
            count += 1
        cache_manager._stats["size"] = len(cache_manager._cache)
    
    logger.info(f"Invalidated {count} cache entries with prefix '{key_prefix}'")
    return count