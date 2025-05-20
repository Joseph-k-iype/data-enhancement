import logging
import os
from typing import Dict, Any, List, Optional
from fasapi import APIRouter, Depends, HTTPException, Query
from app.config.environment import get_os_env, str_to_bool
from app.config.settings import get_llm, get_app_settings
from app.core.db_manager import DBManager
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

class ProxySettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable proxy settings")
    domain: Optional[str] = Field(None, description="Proxy domain")
    
class DatabaseSettings(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    user: str = Field(..., description="Database user")
    database: str = Field(..., description="Database name")
    schema: str = Field(..., description="Database schema")
    min_connections: int = Field(..., description="Minimum number of connections")
    max_connections: int = Field(..., description="Maximum number of connections")

class AppSettings(BaseModel):
    database: DatabaseSettings = Field(..., description="Database settings")
    model: Dict[str, Any] = Field(..., description="Model settings")
    proxy: ProxySettings = Field(..., description="Proxy settings")
    monitoring_interval: int = Field(..., description="Monitoring interval in seconds")
    similarity_threshold: float = Field(..., description="Similarity threshold for vector search")

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
        _ = get_llm(proxy_enabled=proxy_settings.enabled)
        return ProxySettings(
            enabled=proxy_settings.enabled,
            domain=env.get("HTTPS_PROXY_DOMAIN", "")
        )
    except Exception as e:
        logger.error(f"Error updating proxy settings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/database", response_model=DatabaseSettings)
async def get_database_settings():
    try:
        env = get_os_env()
        return DatabaseSettings(
            host=env.get("DATABASE_HOST", ""),
            port=int(env.get("DATABASE_PORT", "5432")),
            user=env.get("DATABASE_USER", ""),
            database=env.get("DATABASE_NAME", ""),
            schema=env.get("DATABASE_SCHEMA", ""),
            min_connections=int(env.get("DATABASE_MIN_CONNECTIONS", "1")),
            max_connections=int(env.get("DATABASE_MAX_CONNECTIONS", "10"))
        )
    except Exception as e:
        logger.error(f"Error getting database settings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/database/health", response_model=Dict[str, Any])
async def get_database_health(db: DBManager = Depends(lambda: DBManager())):
    try:
        health = db.health_check()
        return {"status": "healthy", "details": health}
    except Exception as e:
        logger.error(f"Error getting database health: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/status", response_model=Dict[str, Any])
async def get_system_status(db: DBManager = Depends(lambda: DBManager())):
    try:
        env = get_os_env()
        return get_app_settings()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
@router.get("/stats", response_model=List[Dict[str, Any]])
async def get_system_stats(limit: int = Query(10, description="Number of stats to return"), db: DBManager = Depends(lambda: DBManager())):
    try:
        stats = db.get_stats(limit=limit)
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")