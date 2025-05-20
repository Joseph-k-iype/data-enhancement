"""
Caching utilities to improve performance of expensive operations.
"""

import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast, List, Set, Tuple
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Define a type variable for the function return type
T = TypeVar('T')

class CacheManager:
    """
    Cache manager for expensive operations.
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
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds
        """
        if self._initialized:
            return
            
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "additions": 0
        }
        self._initialized = True
        logger.info(f"Cache manager initialized with max_size={max_size}, ttl={ttl}")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from function arguments.
        
        Args:
            prefix: Prefix for the key (usually function name)
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            str: MD5 hash of the arguments
        """
        # Convert args and kwargs to a string
        key_dict = {
            "prefix": prefix,
            "args": args,
            "kwargs": kwargs
        }
        
        # Handle non-serializable objects
        try:
            key_str = json.dumps(key_dict, sort_keys=True)
        except (TypeError, ValueError):
            # If serialization fails, use a simpler approach
            key_str = f"{prefix}:{str(args)}:{str(kwargs)}"
        
        # Hash the string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found or expired
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        cache_entry = self._cache[key]
        
        # Check if entry is expired
        if time.time() > cache_entry["expires_at"]:
            del self._cache[key]
            self._stats["misses"] += 1
            self._stats["evictions"] += 1
            return None
        
        self._stats["hits"] += 1
        return cache_entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom time to live in seconds
        """
        # Check if cache is full
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(
                self._cache.keys(), 
                key=lambda k: self._cache[k]["expires_at"]
            )
            del self._cache[oldest_key]
            self._stats["evictions"] += 1
        
        # Calculate expiration time
        expires_at = time.time() + (ttl or self._ttl)
        
        # Add entry to cache
        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        self._stats["additions"] += 1
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        hit_rate = 0.0
        total_requests = self._stats["hits"] + self._stats["misses"]
        if total_requests > 0:
            hit_rate = self._stats["hits"] / total_requests
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl": self._ttl,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "additions": self._stats["additions"]
        }
    
    def cached(self, ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator for caching function results.
        
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
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}")
                
                return result
            return cast(Callable[..., T], wrapper)
        return decorator
    
    def async_cached(self, ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator for caching async function results.
        
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
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Call original function
                result = await func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}")
                
                return result
            return cast(Callable[..., T], wrapper)
        return decorator

# Create a global instance for application-wide use
cache_manager = CacheManager()

def setup_caching(max_size: int = 1000, ttl: int = 3600) -> None:
    """
    Initialize the cache manager with custom settings.
    
    Args:
        max_size: Maximum number of items in the cache
        ttl: Time to live in seconds
    """
    global cache_manager
    cache_manager = CacheManager(max_size=max_size, ttl=ttl)
    logger.info(f"Cache initialized with max_size={max_size}, ttl={ttl}")

def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dict with cache statistics
    """
    return cache_manager.get_stats()