import os
import logging
import threading
import time
import asyncio
import requests
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TokenCache:
    """Class to manage Azure tokens in memory with auto-refresh."""
    
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
        """Initialize the token cache."""
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                self._tokens = {}
                self._refresh_thread = None
                self._stop_refresh = threading.Event()
                self._token_callbacks = []  # Callbacks for token refresh events
                self._initialized = True
                logger.info("Token cache initialized")
    
    def get_token(self, tenant_id: str, client_id: str, client_secret: str, 
                  scope: str = "https://cognitiveservices.azure.com/.default",
                  force_refresh: bool = False) -> Optional[str]:
        """
        Get a token from the cache, with optimized refresh strategy.
        
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
        
        with self._lock:
            # Check if we have a cached token
            if not force_refresh and cache_key in self._tokens:
                token_data = self._tokens[cache_key]
                
                # Use a smaller buffer for better token utilization
                # Only refresh when < 5 minutes remaining
                now = datetime.now()
                expiry = token_data.get("expiry")
                buffer = timedelta(minutes=5)
                
                if expiry and now < expiry - buffer:
                    return token_data.get("access_token")
        
        # Token not found or expired, get a new one
        logger.info(f"Getting new token for {client_id} in tenant {tenant_id}")
        token_data = self._fetch_new_token(tenant_id, client_id, client_secret, scope)
        
        if token_data and "access_token" in token_data:
            # Cache the token with expiry time
            expires_in = token_data.get("expires_in", 3600)
            token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
            
            with self._lock:
                self._tokens[cache_key] = token_data
            
            logger.info(f"Token cached (expires in {expires_in}s)")
            
            # Notify callbacks about the token refresh
            for callback in self._token_callbacks:
                try:
                    callback(cache_key, token_data.get("access_token"))
                except Exception as e:
                    logger.error(f"Error in token refresh callback: {e}")
            
            return token_data.get("access_token")
        
        logger.error("Failed to get token")
        return None
    
    async def get_token_async(self, tenant_id: str, client_id: str, client_secret: str, 
                            scope: str = "https://cognitiveservices.azure.com/.default",
                            force_refresh: bool = False) -> Optional[str]:
        """
        Asynchronous version of get_token.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            client_secret: Azure client secret
            scope: Token scope
            force_refresh: Whether to force a token refresh
            
        Returns:
            str: The token, or None if retrieval failed
        """
        # Try to get from cache first
        cache_key = f"{tenant_id}:{client_id}:{scope}"
        
        if not force_refresh:
            with self._lock:
                if cache_key in self._tokens:
                    token_data = self._tokens[cache_key]
                    
                    now = datetime.now()
                    expiry = token_data.get("expiry")
                    buffer = timedelta(minutes=5)
                    
                    if expiry and now < expiry - buffer:
                        return token_data.get("access_token")
        
        # Run the token fetch in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        
        # Use run_in_executor to run the synchronous method in a thread pool
        token = await loop.run_in_executor(
            None,
            lambda: self.get_token(tenant_id, client_id, client_secret, scope, force_refresh)
        )
        
        return token
    
    def _fetch_new_token(self, tenant_id: str, client_id: str, client_secret: str, 
                         scope: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a new token from Azure AD.
        
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
            
            # Use connection pooling for better performance
            session = requests.Session()
            
            response = session.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting token: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception getting token: {e}")
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
        Start a background thread to periodically refresh tokens.
        
        Args:
            refresh_interval: How often to check for tokens to refresh (seconds)
        """
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
        Worker function for the token refresh service.
        
        Args:
            refresh_interval: How often to check for tokens to refresh (seconds)
        """
        logger.info("Token refresh service worker started")
        
        while not self._stop_refresh.is_set():
            try:
                now = datetime.now()
                tokens_to_refresh = []
                
                # Identify tokens to refresh under lock
                with self._lock:
                    for cache_key, token_data in self._tokens.items():
                        expiry = token_data.get("expiry")
                        
                        # Use a larger buffer for auto-refresh (15 minutes)
                        buffer = timedelta(minutes=15)
                        
                        if expiry and now > expiry - buffer:
                            # Token is approaching expiration, extract credentials
                            tenant_id, client_id, scope = cache_key.split(":", 2)
                            client_secret = os.environ.get(f"AZURE_CLIENT_SECRET_{client_id}", 
                                                          os.environ.get("AZURE_CLIENT_SECRET", ""))
                            
                            if client_secret:
                                tokens_to_refresh.append((tenant_id, client_id, client_secret, scope, cache_key))
                
                # Refresh tokens outside of lock
                for tenant_id, client_id, client_secret, scope, cache_key in tokens_to_refresh:
                    logger.info(f"Auto-refreshing token for {client_id} in tenant {tenant_id}")
                    token_data = self._fetch_new_token(tenant_id, client_id, client_secret, scope)
                    
                    if token_data and "access_token" in token_data:
                        with self._lock:
                            # Update token in cache
                            expires_in = token_data.get("expires_in", 3600)
                            token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
                            self._tokens[cache_key] = token_data
                            
                        logger.info(f"Token refreshed (expires in {expires_in}s)")
                        
                        # Notify callbacks about the token refresh
                        for callback in self._token_callbacks:
                            try:
                                callback(cache_key, token_data.get("access_token"))
                            except Exception as e:
                                logger.error(f"Error in token refresh callback: {e}")
                    else:
                        logger.error(f"Failed to refresh token for {client_id}")
            
            except Exception as e:
                logger.error(f"Error in token refresh service: {e}")
            
            # Sleep until next check, but wake up every minute to check if we should stop
            for _ in range(refresh_interval // 60):
                if self._stop_refresh.wait(60):
                    break
        
        logger.info("Token refresh service worker stopped")

# Global token cache instance
token_cache = TokenCache()

def get_azure_token(tenant_id: str, client_id: str, client_secret: str, 
                   scope: str = "https://cognitiveservices.azure.com/.default",
                   force_refresh: bool = False) -> Optional[str]:
    """
    Get an Azure token, using the token cache.
    
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
    Get an Azure token asynchronously, using the token cache.
    
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
    Initialize the token caching system.
    
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
        token_cache.get_token(tenant_id, client_id, client_secret)
    
    if start_refresh_service:
        token_cache.start_refresh_service(refresh_interval)