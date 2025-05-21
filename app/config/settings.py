"""
Application settings for the Data Enhancement Service.
"""

# import os
# import logging
# from typing import Optional, Union, Dict, Any, List, Set
# from pydantic import BaseModel, Field, validator
# import langchain
# from langchain.cache import InMemoryCache
# from app.config.environment import get_os_env
# from langchain_openai import AzureChatOpenAI
# from app.utils.auth_helper import get_azure_token

"""
Update the LangChain caching configuration in app/config/settings.py
"""

# Update this section at the top of settings.py
import os
import logging
import threading
from typing import Optional, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field, validator, model_validator

# Update these imports for modern LangChain
from langchain_core.caches import BaseCache, InMemoryCache  # Updated import path
import langchain_core as langchain  # Updated namespace

from app.config.environment import get_os_env
from langchain_openai import AzureChatOpenAI
from app.utils.auth_helper import get_azure_token

logger = logging.getLogger(__name__)

# Initialize LangChain cache for token savings
langchain.globals.set_llm_cache(InMemoryCache())  # Updated method for setting cache

# Rest of the settings.py file remains the same


logger = logging.getLogger(__name__)

# Initialize LangChain cache
langchain.llm_cache = InMemoryCache()

def get_llm(proxy_enabled: Optional[bool] = True) -> AzureChatOpenAI:
    """
    Get a Language Model instance with optimized token management.
    
    Args:
        proxy_enabled: Whether to enable proxy settings
        
    Returns:
        AzureChatOpenAI: The language model
    """
    env = get_os_env(proxy_enabled=proxy_enabled)
    
    # Updated API version for better performance
    api_version = "2024-02-01"  # Update from "2023-05-15" to latest version
    
    try:
        # Get token using the token caching mechanism
        token = get_azure_token(
            tenant_id=env.get("AZURE_TENANT_ID", ""),
            client_id=env.get("AZURE_CLIENT_ID", ""),
            client_secret=env.get("AZURE_CLIENT_SECRET", ""),
            scope="https://cognitiveservices.azure.com/.default"
        )
        
        if token:
            logger.info("Token retrieved successfully.")
            token_provider = lambda: token
        else:
            logger.error("Failed to retrieve token.")
            raise ValueError("Failed to retrieve token.")
    except Exception as e:
        logger.error(f"Error retrieving token: {e}")
        raise ValueError("Failed to retrieve token.")
    
    model_name = env.get("MODEL_NAME", "gpt-4o-mini")  # Use faster model as default
    max_tokens = env.get("MAX_TOKENS", 2000)
    temperature = env.get("TEMPERATURE", 0.3)
    azure_endpoint = env.get("AZURE_ENDPOINT", "")
    
    # Add performance optimizations
    return AzureChatOpenAI(
        model_name=model_name,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        streaming=True,  # Enable streaming for faster perceived response
        request_timeout=60.0,  # Increase timeout for more reliable responses
        max_retries=3,  # Increase retries for reliability
        cache=True,  # Enable LangChain's caching for repeat queries
    )
    
def get_app_settings() -> Dict[str, Any]:
    """
    Get application settings.
    
    Returns:
        Dict: Application settings
    """
    env = get_os_env()
    
    model_settings = {
        "model": env.get("MODEL_NAME", "gpt-4o-mini"),
        "max_tokens": int(env.get("MAX_TOKENS", "2000")),
        "temperature": float(env.get("TEMPERATURE", "0.3")),
        "api_version": env.get("API_VERSION", "2024-02-01"),  # Updated to latest
        "azure_endpoint": env.get("AZURE_ENDPOINT", ""),
    }
    
    proxy_settings = {
        "enabled": env.get("PROXY_ENABLED", "False").lower() in ('true', 't', 'yes', 'y', '1'),
        "domain": env.get("HTTPS_PROXY_DOMAIN", "")
    }
    
    token_settings = {
        "refresh_interval": int(env.get("TOKEN_REFRESH_INTERVAL", "600")),  # Increased from 300
        "validation_threshold": int(env.get("TOKEN_VALIDATION_THRESHOLD", "600"))
    }
    
    monitoring_interval = int(env.get("MONITORING_INTERVAL", "300"))
    
    return {
        "model": model_settings,
        "proxy": proxy_settings,
        "token": token_settings,
        "monitoring_interval": monitoring_interval
    }