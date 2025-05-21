"""
Optimized token management for Azure authentication.
Replace app/utils/auth_helper.py with this implementation.
"""

import os
import logging
import threading
import time
import asyncio
import requests
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)

class TokenCache:
    """Enhanced class to manage Azure tokens in memory with efficient refresh strategy."""
    
    _instance = None
    _lock = threading.RLock()  # Using RLock for better performance
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the token cache."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenCache, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the token cache with performance optimizations."""
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                self._tokens = {}
                self._refresh_thread = None
                self._stop_refresh = threading.Event()
                self._token_callbacks = []  # Callbacks for token refresh events
                self._session = requests.Session()  # Persistent HTTP session for better performance
                self._refresh_in_progress = {}  # Track which tokens are currently being refreshed
                self._refresh_interval = 600  # Default refresh interval in seconds
                
                # Connection pooling settings for better network performance
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=3
                )
                self._session.mount('https://', adapter)
                
                self._initialized = True
                logger.info("Enhanced token cache initialized")
    
    def get_token(self, tenant_id: str, client_id: str, client_secret: str, 
                  scope: str = "https://cognitiveservices.azure.com/.default",
                  force_refresh: bool = False) -> Optional[str]:
        """
        Get a token from the cache with optimized refresh strategy.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            client_secret: Azure client secret
            scope: Token scope
            force_refresh: Whether to force a token refresh
            
        Returns:
            str: The token, or None if retrieval failed
        """
        cache_key = f"{tenant_id}:{client_id}:{scope}"
        
        # First check if token refresh is already in progress
        with self._lock:
            if cache_key in self._refresh_in_progress and self._refresh_in_progress[cache_key]:
                # If the token is being refreshed, wait a short time and recheck the cache
                logger.debug(f"Token refresh in progress for {client_id}, waiting...")
        
        with self._lock:
            # Check if we have a cached token
            if not force_refresh and cache_key in self._tokens:
                token_data = self._tokens[cache_key]
                
                # Use an adaptive buffer based on token lifetime
                # - For tokens with long lifetime (>= 30 min), use 5 min buffer
                # - For tokens with short lifetime (< 30 min), use 15% of lifetime
                now = datetime.now()
                expiry = token_data.get("expiry")
                
                if expiry:
                    token_lifetime = (expiry - now).total_seconds()
                    if token_lifetime >= 1800:  # 30 minutes
                        buffer_seconds = 300  # 5 minutes
                    else:
                        buffer_seconds = max(10, int(token_lifetime * 0.15))  # 15% of lifetime, min 10 seconds
                    
                    buffer = timedelta(seconds=buffer_seconds)
                    
                    if now < expiry - buffer:
                        return token_data.get("access_token")
        
        # Token not found, expired, or forced refresh - start refresh process
        with self._lock:
            # Mark that refresh is in progress for this token
            self._refresh_in_progress[cache_key] = True
        
        try:
            logger.info(f"Getting new token for {client_id} in tenant {tenant_id}")
            token_data = self._fetch_new_token(tenant_id, client_id, client_secret, scope)
            
            if token_data and "access_token" in token_data:
                # Cache the token with expiry time
                expires_in = token_data.get("expires_in", 3600)
                token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
                
                with self._lock:
                    self._tokens[cache_key] = token_data
                    # Mark refresh as complete
                    self._refresh_in_progress[cache_key] = False
                
                logger.info(f"Token cached for {client_id} (expires in {expires_in}s)")
                
                # Notify callbacks about the token refresh
                for callback in self._token_callbacks:
                    try:
                        callback(cache_key, token_data.get("access_token"))
                    except Exception as e:
                        logger.error(f"Error in token refresh callback: {e}")
                
                return token_data.get("access_token")
            
            logger.error(f"Failed to get token for {client_id}")
            with self._lock:
                # Mark refresh as complete even if it failed
                self._refresh_in_progress[cache_key] = False
            return None
            
        except Exception as e:
            logger.error(f"Exception getting token for {client_id}: {e}")
            with self._lock:
                # Mark refresh as complete
                self._refresh_in_progress[cache_key] = False
            return None
    
    async def get_token_async(self, tenant_id: str, client_id: str, client_secret: str, 
                            scope: str = "https://cognitiveservices.azure.com/.default",
                            force_refresh: bool = False) -> Optional[str]:
        """
        Asynchronous version of get_token with connection pooling.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            client_secret: Azure client secret
            scope: Token scope
            force_refresh: Whether to force a token refresh
            
        Returns:
            str: The token, or None if retrieval failed
        """
        cache_key = f"{tenant_id}:{client_id}:{scope}"
        
        # First check cache without forcing a refresh
        if not force_refresh:
            with self._lock:
                if cache_key in self._tokens:
                    token_data = self._tokens[cache_key]
                    
                    now = datetime.now()
                    expiry = token_data.get("expiry")
                    
                    if expiry:
                        token_lifetime = (expiry - now).total_seconds()
                        if token_lifetime >= 1800:  # 30 minutes
                            buffer_seconds = 300  # 5 minutes
                        else:
                            buffer_seconds = max(10, int(token_lifetime * 0.15))
                        
                        buffer = timedelta(seconds=buffer_seconds)
                        
                        if now < expiry - buffer:
                            return token_data.get("access_token")
        
        # Token not found, expired, or forced refresh - fetch asynchronously
        try:
            with self._lock:
                # Mark that refresh is in progress
                self._refresh_in_progress[cache_key] = True
            
            logger.info(f"Getting new token async for {client_id} in tenant {tenant_id}")
            token_data = await self._fetch_new_token_async(tenant_id, client_id, client_secret, scope)
            
            if token_data and "access_token" in token_data:
                # Cache the token with expiry time
                expires_in = token_data.get("expires_in", 3600)
                token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
                
                with self._lock:
                    self._tokens[cache_key] = token_data
                    # Mark refresh as complete
                    self._refresh_in_progress[cache_key] = False
                
                logger.info(f"Token cached async for {client_id} (expires in {expires_in}s)")
                
                # Schedule callbacks to run in the background
                asyncio.create_task(self._run_callbacks_async(cache_key, token_data.get("access_token")))
                
                return token_data.get("access_token")
            
            logger.error(f"Failed to get token async for {client_id}")
            with self._lock:
                self._refresh_in_progress[cache_key] = False
            return None
            
        except Exception as e:
            logger.error(f"Exception getting token async for {client_id}: {e}")
            with self._lock:
                self._refresh_in_progress[cache_key] = False
            return None
    
    async def _run_callbacks_async(self, cache_key: str, token: str):
        """Run token callbacks asynchronously."""
        for callback in self._token_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(cache_key, token)
                else:
                    # Run synchronous callbacks in executor
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: callback(cache_key, token))
            except Exception as e:
                logger.error(f"Error in async token refresh callback: {e}")
    
    def _fetch_new_token(self, tenant_id: str, client_id: str, client_secret: str, 
                         scope: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a new token from Azure AD with optimized connection handling.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            client_secret: Azure client secret
            scope: Token scope
            
        Returns:
            Dict: Token data, or None if retrieval failed
        """
        try:
            url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope,
                "grant_type": "client_credentials"
            }
            
            # Use connection pooling with the persistent session
            response = self._session.post(
                url, 
                data=data, 
                timeout=30,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting token: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception getting token: {e}")
            return None
    
    async def _fetch_new_token_async(self, tenant_id: str, client_id: str, client_secret: str, 
                                   scope: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a new token from Azure AD asynchronously.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            client_secret: Azure client secret
            scope: Token scope
            
        Returns:
            Dict: Token data, or None if retrieval failed
        """
        try:
            url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope,
                "grant_type": "client_credentials"
            }
            
            # Use aiohttp for async requests with TCP connection reuse
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(
                    url, 
                    data=data, 
                    timeout=30,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Error getting token async: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Exception getting token async: {e}")
            return None
    
    def register_token_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Register a callback to be notified when a token is refreshed.
        
        Args:
            callback: Function to call with (cache_key, token) when a token is refreshed
        """
        with self._lock:
            self._token_callbacks.append(callback)
    
    def start_refresh_service(self, refresh_interval: int = 600):
        """
        Start a background thread to proactively refresh tokens before expiry.
        
        Args:
            refresh_interval: How often to check for tokens to refresh (seconds)
        """
        self._refresh_interval = refresh_interval
        
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            logger.info("Token refresh service already running")
            return
            
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_service_worker,
            args=(refresh_interval,),
            daemon=True
        )
        self._refresh_thread.start()
        logger.info(f"Token refresh service started with interval {refresh_interval}s")
    
    def stop_refresh_service(self):
        """Stop the token refresh service."""
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            self._stop_refresh.set()
            self._refresh_thread.join(timeout=10)
            logger.info("Token refresh service stopped")
    
    def _refresh_service_worker(self, refresh_interval: int):
        """
        Worker function for the token refresh service with smart prioritization.
        
        Args:
            refresh_interval: How often to check for tokens to refresh (seconds)
        """
        logger.info(f"Token refresh service worker started (interval: {refresh_interval}s)")
        
        while not self._stop_refresh.wait(min(60, refresh_interval / 6)):  # Check more frequently
            try:
                now = datetime.now()
                tokens_to_refresh = []
                
                # Identify tokens to refresh under lock
                with self._lock:
                    for cache_key, token_data in self._tokens.items():
                        # Skip tokens already being refreshed
                        if cache_key in self._refresh_in_progress and self._refresh_in_progress[cache_key]:
                            continue
                            
                        expiry = token_data.get("expiry")
                        
                        # Proactively refresh tokens that will expire soon
                        # Use a larger buffer for auto-refresh:
                        # - For tokens with < 30min remaining, refresh at 50% of remaining time
                        # - For tokens with >= 30min remaining, refresh at 10min before expiry
                        if expiry:
                            time_remaining = (expiry - now).total_seconds()
                            
                            if time_remaining < 1800:  # Less than 30 minutes
                                buffer = timedelta(seconds=time_remaining * 0.5)  # 50% of remaining time
                            else:
                                buffer = timedelta(minutes=10)  # 10 minutes for long-lived tokens
                            
                            if now > expiry - buffer:
                                # Token is approaching expiration, extract credentials
                                tenant_id, client_id, scope = cache_key.split(":", 2)
                                client_secret = os.environ.get(f"AZURE_CLIENT_SECRET_{client_id}", 
                                                            os.environ.get("AZURE_CLIENT_SECRET", ""))
                                
                                if client_secret:
                                    tokens_to_refresh.append((
                                        tenant_id, client_id, client_secret, scope, cache_key,
                                        (expiry - now).total_seconds()  # Time remaining
                                    ))
                
                # Sort tokens by expiry time (refresh most urgent ones first)
                tokens_to_refresh.sort(key=lambda x: x[5])
                
                # Process urgent tokens synchronously (expiring within 2 minutes)
                urgent_tokens = [(t[0], t[1], t[2], t[3], t[4]) for t in tokens_to_refresh if t[5] < 120]
                regular_tokens = [(t[0], t[1], t[2], t[3], t[4]) for t in tokens_to_refresh if t[5] >= 120]
                
                # Process urgent tokens
                for tenant_id, client_id, client_secret, scope, cache_key in urgent_tokens:
                    try:
                        logger.info(f"URGENT auto-refreshing token for {client_id} (expires in {int(tokens_to_refresh[0][5])}s)")
                        with self._lock:
                            self._refresh_in_progress[cache_key] = True
                            
                        token_data = self._fetch_new_token(tenant_id, client_id, client_secret, scope)
                        
                        if token_data and "access_token" in token_data:
                            expires_in = token_data.get("expires_in", 3600)
                            token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
                            
                            with self._lock:
                                self._tokens[cache_key] = token_data
                                self._refresh_in_progress[cache_key] = False
                            
                            # Notify callbacks
                            for callback in self._token_callbacks:
                                try:
                                    callback(cache_key, token_data.get("access_token"))
                                except Exception as e:
                                    logger.error(f"Error in token refresh callback: {e}")
                        else:
                            with self._lock:
                                self._refresh_in_progress[cache_key] = False
                            logger.error(f"Failed to refresh urgent token for {client_id}")
                    except Exception as e:
                        with self._lock:
                            self._refresh_in_progress[cache_key] = False
                        logger.error(f"Error refreshing urgent token for {client_id}: {e}")
                
                # Process regular tokens in parallel (limit to 3 at a time)
                for i in range(0, len(regular_tokens), 3):
                    batch = regular_tokens[i:i+3]
                    threads = []
                    
                    for tenant_id, client_id, client_secret, scope, cache_key in batch:
                        with self._lock:
                            self._refresh_in_progress[cache_key] = True
                        
                        thread = threading.Thread(
                            target=self._refresh_token_thread,
                            args=(tenant_id, client_id, client_secret, scope, cache_key),
                            daemon=True
                        )
                        thread.start()
                        threads.append(thread)
                    
                    # Wait for all threads in this batch to complete
                    for thread in threads:
                        thread.join(timeout=30)
                
            except Exception as e:
                logger.error(f"Error in token refresh service: {e}")
    
    def _refresh_token_thread(self, tenant_id: str, client_id: str, client_secret: str, 
                             scope: str, cache_key: str):
        """Worker thread to refresh a token in the background."""
        try:
            logger.info(f"Auto-refreshing token for {client_id} in tenant {tenant_id}")
            token_data = self._fetch_new_token(tenant_id, client_id, client_secret, scope)
            
            if token_data and "access_token" in token_data:
                # Update token in cache
                expires_in = token_data.get("expires_in", 3600)
                token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
                
                with self._lock:
                    self._tokens[cache_key] = token_data
                    self._refresh_in_progress[cache_key] = False
                
                logger.info(f"Token refreshed for {client_id} (expires in {expires_in}s)")
                
                # Notify callbacks
                for callback in self._token_callbacks:
                    try:
                        callback(cache_key, token_data.get("access_token"))
                    except Exception as e:
                        logger.error(f"Error in token refresh callback: {e}")
            else:
                with self._lock:
                    self._refresh_in_progress[cache_key] = False
                logger.error(f"Failed to refresh token for {client_id}")
        except Exception as e:
            with self._lock:
                self._refresh_in_progress[cache_key] = False
            logger.error(f"Error in token refresh thread for {client_id}: {e}")
    
    def get_token_expiry(self, tenant_id: str, client_id: str,
                       scope: str = "https://cognitiveservices.azure.com/.default") -> Optional[datetime]:
        """
        Get the expiry time of a cached token.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            scope: Token scope
            
        Returns:
            datetime: Token expiry time, or None if not found
        """
        cache_key = f"{tenant_id}:{client_id}:{scope}"
        
        with self._lock:
            if cache_key in self._tokens:
                return self._tokens[cache_key].get("expiry")
        
        return None
    
    def get_token_status(self) -> Dict[str, Any]:
        """
        Get status of all cached tokens.
        
        Returns:
            Dict with token status information
        """
        now = datetime.now()
        status = {}
        
        with self._lock:
            for cache_key, token_data in self._tokens.items():
                parts = cache_key.split(":", 2)
                if len(parts) == 3:
                    tenant_id, client_id, scope = parts
                    
                    expiry = token_data.get("expiry")
                    if expiry:
                        seconds_remaining = (expiry - now).total_seconds()
                        status[cache_key] = {
                            "tenant_id": tenant_id,
                            "client_id": client_id,
                            "scope": scope,
                            "expires_at": expiry.isoformat(),
                            "seconds_remaining": max(0, seconds_remaining),
                            "is_active": seconds_remaining > 0,
                            "refresh_in_progress": self._refresh_in_progress.get(cache_key, False)
                        }
        
        return status

# Global token cache instance
token_cache = TokenCache()

def get_azure_token(tenant_id: str, client_id: str, client_secret: str, 
                   scope: str = "https://cognitiveservices.azure.com/.default",
                   force_refresh: bool = False) -> Optional[str]:
    """
    Get an Azure token, using the optimized token cache.
    
    Args:
        tenant_id: Azure tenant ID
        client_id: Azure client ID
        client_secret: Azure client secret
        scope: Token scope
        force_refresh: Whether to force a token refresh
        
    Returns:
        str: The token, or None if retrieval failed
    """
    return token_cache.get_token(tenant_id, client_id, client_secret, scope, force_refresh)

async def get_azure_token_async(tenant_id: str, client_id: str, client_secret: str, 
                              scope: str = "https://cognitiveservices.azure.com/.default",
                              force_refresh: bool = False) -> Optional[str]:
    """
    Get an Azure token asynchronously, using the optimized token cache.
    
    Args:
        tenant_id: Azure tenant ID
        client_id: Azure client ID
        client_secret: Azure client secret
        scope: Token scope
        force_refresh: Whether to force a token refresh
        
    Returns:
        str: The token, or None if retrieval failed
    """
    return await token_cache.get_token_async(tenant_id, client_id, client_secret, scope, force_refresh)

def get_azure_token_cached(tenant_id: str, client_id: str, client_secret: str, 
                          scope: str = "https://cognitiveservices.azure.com/.default") -> Optional[str]:
    """Alias for get_azure_token for backward compatibility."""
    return get_azure_token(tenant_id, client_id, client_secret, scope)

def initialize_token_caching(start_refresh_service: bool = True, refresh_interval: int = 600):
    """
    Initialize the token caching system with optimized settings.
    
    Args:
        start_refresh_service: Whether to start the background refresh service
        refresh_interval: How often to check for tokens to refresh (seconds)
    """
    global token_cache
    
    # Force initialization of the singleton
    token_cache = TokenCache()
    
    # Pre-fetch token if credentials are available
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    
    if tenant_id and client_id and client_secret:
        logger.info("Pre-fetching token with environment credentials")
        try:
            # Fetch token in a separate thread to avoid blocking startup
            threading.Thread(
                target=lambda: token_cache.get_token(tenant_id, client_id, client_secret),
                daemon=True
            ).start()
        except Exception as e:
            logger.error(f"Error pre-fetching token: {e}")
    
    if start_refresh_service:
        token_cache.start_refresh_service(refresh_interval)

def get_token_status() -> Dict[str, Any]:
    """
    Get status of all cached tokens.
    
    Returns:
        Dict with token status information
    """
    return token_cache.get_token_status()