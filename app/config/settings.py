"""
Application settings for the Data Enhancement Service.
"""

import os
import logging
from typing import Optional, Union, Dict, Any, List, Set
from pydantic import BaseModel, Field, validator
from app.config.environment import get_os_env
from langchain_openai import AzureChatOpenAI
from app.utils.auth_helper import get_azure_token

logger = logging.getLogger(__name__)

def get_llm(proxy_enabled: Optional[bool] = True) -> AzureChatOpenAI:
    """
    Get a Language Model instance with token management.
    
    Args:
        proxy_enabled: Whether to enable proxy settings
        
    Returns:
        AzureChatOpenAI: The language model
    """
    env = get_os_env(proxy_enabled=proxy_enabled)
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
            # Create a function that returns the token
            token_provider = lambda: token
        else:
            logger.error("Failed to retrieve token.")
            raise ValueError("Failed to retrieve token.")
    except Exception as e:
        logger.error(f"Error retrieving token: {e}")
        raise ValueError("Failed to retrieve token.")
    
    model_name = env.get("MODEL_NAME", "gpt-4o-mini")
    max_tokens = env.get("MAX_TOKENS", 2000)
    temperature = env.get("TEMPERATURE", 0.3)
    api_version = env.get("API_VERSION", "2023-05-15")
    azure_endpoint = env.get("AZURE_ENDPOINT", "")
    
    return AzureChatOpenAI(
        model_name=model_name,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider)
    
def get_app_settings() -> Dict[str, Any]:
    """
    Get application settings.
    
    Returns:
        Dict: Application settings
    """
    env = get_os_env()
    pg_settings = {
        "host": env.get("PG_HOST", ""),
        "port": env.get("PG_PORT", ""),
        "user": env.get("PG_USER", ""),
        "database": env.get("PG_DB", ""),
        "schema": env.get("PG_SCHEMA", "ai_stitching_platform"),
        "min_connections": int(env.get("PG_MIN_CONNECTIONS", "1")),
        "max_connections": int(env.get("PG_MAX_CONNECTIONS", "10"))
    }
    
    model_settings = {
        "model": env.get("MODEL_NAME", "gpt-4o-mini"),
        "max_tokens": int(env.get("MAX_TOKENS", "2000")),
        "temperature": float(env.get("TEMPERATURE", "0.3")),
        "api_version": env.get("API_VERSION", "2023-05-15"),
        "azure_endpoint": env.get("AZURE_ENDPOINT", ""),
    }
    
    proxy_settings = {
        "enabled": env.get("PROXY_ENABLED", "False").lower() in ('true', 't', 'yes', 'y', '1'),
        "domain": env.get("HTTPS_PROXY_DOMAIN", "")
    }
    
    monitoring_interval = int(env.get("MONITORING_INTERVAL", "300"))
    
    return {
        "database": pg_settings,
        "model": model_settings,
        "proxy": proxy_settings,
        "monitoring_interval": monitoring_interval
    }