"""
Settings API - Routes for application settings and status information.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.config.environment import get_os_env, str_to_bool
from app.config.settings import get_llm, get_app_settings
from app.core.in_memory_job_store import job_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

class ProxySettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable proxy settings")
    domain: Optional[str] = Field(None, description="Proxy domain")

class ModelSettings(BaseModel):
    model: str = Field(..., description="Model name")
    max_tokens: int = Field(..., description="Maximum tokens")
    temperature: float = Field(..., description="Temperature for model responses")
    api_version: str = Field(..., description="API version")
    azure_endpoint: str = Field(..., description="Azure endpoint")

class TokenSettings(BaseModel):
    refresh_interval: int = Field(..., description="Token refresh interval in seconds")
    validation_threshold: int = Field(..., description="Token validation threshold in seconds")

class AppSettings(BaseModel):
    model: ModelSettings = Field(..., description="Model settings")
    proxy: ProxySettings = Field(..., description="Proxy settings")
    token: TokenSettings = Field(..., description="Token settings")
    monitoring_interval: int = Field(..., description="Monitoring interval in seconds")

@router.get("/proxy", response_model=ProxySettings)
async def get_proxy_settings():
    try:
        env = get_os_env()
        proxy_enabled = str_to_bool(env.get("PROXY_ENABLED", "False"))
        return ProxySettings(
            enabled=proxy_enabled,
            domain=env.get("HTTPS_PROXY_DOMAIN", "")
        )
    except Exception as e:
        logger.error(f"Error getting proxy settings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/proxy", response_model=ProxySettings)
async def update_proxy_settings(proxy_settings: ProxySettings):
    try:
        env = get_os_env(proxy_enabled=proxy_settings.enabled)
        # Reinitialize LLM with new proxy settings
        _ = get_llm(proxy_enabled=proxy_settings.enabled)
        return ProxySettings(
            enabled=proxy_settings.enabled,
            domain=env.get("HTTPS_PROXY_DOMAIN", "")
        )
    except Exception as e:
        logger.error(f"Error updating proxy settings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/storage/health", response_model=Dict[str, Any])
async def get_storage_health():
    try:
        health = job_store.health_check()
        return {"status": "healthy", "details": health}
    except Exception as e:
        logger.error(f"Error getting storage health: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    try:
        env = get_os_env()
        app_settings = get_app_settings()
        storage_health = job_store.health_check()
        
        return {
            "app_settings": app_settings,
            "storage": storage_health,
            "environment": {
                "proxy_enabled": str_to_bool(env.get("PROXY_ENABLED", "False")),
                "azure_endpoint": env.get("AZURE_ENDPOINT", ""),
                "model": env.get("MODEL_NAME", "gpt-4o-mini")
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/stats", response_model=List[Dict[str, Any]])
async def get_system_stats(limit: int = Query(10, description="Number of stats to return")):
    try:
        stats = job_store.get_system_stats(limit=limit)
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/model", response_model=ModelSettings)
async def get_model_settings():
    try:
        env = get_os_env()
        return ModelSettings(
            model=env.get("MODEL_NAME", "gpt-4o-mini"),
            max_tokens=int(env.get("MAX_TOKENS", "2000")),
            temperature=float(env.get("TEMPERATURE", "0.3")),
            api_version=env.get("API_VERSION", "2023-05-15"),
            azure_endpoint=env.get("AZURE_ENDPOINT", "")
        )
    except Exception as e:
        logger.error(f"Error getting model settings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/model", response_model=ModelSettings)
async def update_model_settings(model_settings: ModelSettings):
    try:
        env = get_os_env()
        
        # Update environment variables
        os.environ["MODEL_NAME"] = model_settings.model
        os.environ["MAX_TOKENS"] = str(model_settings.max_tokens)
        os.environ["TEMPERATURE"] = str(model_settings.temperature)
        os.environ["API_VERSION"] = model_settings.api_version
        os.environ["AZURE_ENDPOINT"] = model_settings.azure_endpoint
        
        # Reinitialize LLM with new settings
        _ = get_llm()
        
        return model_settings
    except Exception as e:
        logger.error(f"Error updating model settings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")