import os
import logging
import threading
import time
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TokenCache:
    """Class to manage Azure tokens in memory with auto-refresh."""
    
    _instance = None
    _lock = threading.Lock()
    
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
                self._initialized = True
                logger.info("Token cache initialized")
    
    def get_token(self, tenant_id: str, client_id: str, client_secret: str, 
                  scope: str = "https://cognitiveservices.azure.com/.default",
                  force_refresh: bool = False) -> Optional[str]:
        """
        Get a token from the cache, refreshing if necessary.
        
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
                
                # Check if the token is still valid with a buffer
                # Buffer is 10% of the expiration time or at least 5 minutes
                now = datetime.now()
                expiry = token_data.get("expiry")
                buffer = max(timedelta(minutes=5), timedelta(seconds=token_data.get("expires_in", 3600) * 0.1))
                
                if expiry and now < expiry - buffer:
                    logger.debug(f"Using cached token (expires in {(expiry - now).total_seconds():.0f}s)")
                    return token_data.get("access_token")
            
            # Token not found or expired, get a new one
            logger.info(f"Getting new token for {client_id} in tenant {tenant_id}")
            token_data = self._fetch_new_token(tenant_id, client_id, client_secret, scope)
            
            if token_data and "access_token" in token_data:
                # Cache the token with expiry time
                expires_in = token_data.get("expires_in", 3600)
                token_data["expiry"] = datetime.now() + timedelta(seconds=expires_in)
                self._tokens[cache_key] = token_data
                logger.info(f"Token cached (expires in {expires_in}s)")
                return token_data.get("access_token")
            
            logger.error("Failed to get token")
            return None
    
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
            
            response = requests.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting token: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception getting token: {e}")
            return None
    
    def start_refresh_service(self, refresh_interval: int = 300):
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
                        
                        # Use a larger buffer for auto-refresh (20% of expiration time)
                        buffer = max(timedelta(minutes=10), 
                                    timedelta(seconds=token_data.get("expires_in", 3600) * 0.2))
                        
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
                    else:
                        logger.error(f"Failed to refresh token for {client_id}")
            
            except Exception as e:
                logger.error(f"Error in token refresh service: {e}")
            
            # Sleep until next check
            self._stop_refresh.wait(refresh_interval)
        
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

def get_azure_token_cached(tenant_id: str, client_id: str, client_secret: str, 
                          scope: str = "https://cognitiveservices.azure.com/.default") -> Optional[str]:
    """Alias for get_azure_token for backward compatibility."""
    return get_azure_token(tenant_id, client_id, client_secret, scope)

def get_azure_token_manual(tenant_id: str, client_id: str, client_secret: str, 
                          scope: str = "https://cognitiveservices.azure.com/.default") -> Optional[str]:
    """Alias for get_azure_token for backward compatibility."""
    return get_azure_token(tenant_id, client_id, client_secret, scope)

def initialize_token_caching(start_refresh_service: bool = True, refresh_interval: int = 300):
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