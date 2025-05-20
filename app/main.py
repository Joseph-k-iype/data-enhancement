"""
Main Application - Entry point for the Data Element Enhancement API.

This module initializes and configures the FastAPI application, sets up routes,
middleware, and monitoring, and handles command line arguments for configuration.
"""

import argparse
import logging
import os
import sys
from typing import Optional
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure parent directory is in path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api.routes.enhancement import router as enhancement_router
from app.api.routes.settings import router as settings_router
from app.config.environment import get_os_env, str_to_bool
from app.core.system_monitor import start_monitoring, stop_monitoring
from app.core.db_manager import DBManager
from app.utils.auth_helper import initialize_token_caching

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to stdout
    ]
)
logger = logging.getLogger(__name__)

# Make sure psutil is installed (for system monitoring)
try:
    import psutil
except ImportError:
    logger.warning("psutil not installed. System monitoring features might be limited. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        logger.info("psutil installed successfully.")
    except Exception as e:
        logger.error(f"Failed to install psutil: {e}. Monitoring might not work correctly.")


def create_application(
    proxy_enabled: bool = True, 
    monitoring_interval: int = 300,
    token_refresh_interval: int = 300
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        proxy_enabled: Whether to enable proxy settings
        monitoring_interval: Interval for system monitoring in seconds
        token_refresh_interval: Interval for token refreshing in seconds
        
    Returns:
        FastAPI: The FastAPI application
    """
    logger.info("Creating FastAPI application...")
    # Initialize environment settings (proxy, credentials, etc.)
    env = get_os_env(proxy_enabled=proxy_enabled)
    
    # Initialize token caching system - this will cache an initial token
    # and start a background thread to refresh it
    initialize_token_caching(start_refresh_service=True, refresh_interval=token_refresh_interval)
    logger.info(f"Token caching system initialized with refresh interval: {token_refresh_interval}s")
    
    logger.info(f"Proxy enabled: {env.get('PROXY_ENABLED')}")
    logger.info(f"Model name: {env.get('MODEL_NAME', 'gpt-4o-mini')}")

    # Initialize PostgreSQL DBManager (for jobs, non-vector data)
    db_manager = DBManager()
    db_health = db_manager.health_check()
    if db_health["status"] == "healthy":
        pg_version_info = db_health.get('version', 'unknown version')
        pg_version = pg_version_info.split(' ')[1] if len(pg_version_info.split(' ')) > 1 else pg_version_info
        logger.info(f"PostgreSQL connection successful: Version {pg_version}")
    else:
        logger.error(f"PostgreSQL connection failed: {db_health.get('error', 'Unknown error')}")
    
    app = FastAPI(
        title="Data Element Enhancement API",
        description="API for enhancing data elements to meet ISO/IEC 11179 standards.",
        version="2.0.0", # Updated version
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files for dashboard
    static_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    if not os.path.exists(static_dir_path):
        os.makedirs(static_dir_path)
        logger.info(f"Created static directory: {static_dir_path}")
    app.mount("/static", StaticFiles(directory=static_dir_path), name="static")
    
    # Include API routers
    app.include_router(enhancement_router)
    app.include_router(settings_router)
    
    if monitoring_interval > 0:
        logger.info(f"Starting system monitoring. Interval: {monitoring_interval}s")
        start_monitoring(interval=monitoring_interval)
    else:
        logger.info("System monitoring is disabled.")

    @app.on_event("shutdown")
    def shutdown_event():
        logger.info("Application shutting down...")
        if monitoring_interval > 0:
            stop_monitoring()
        # Add any other cleanup tasks here (e.g., closing DB connections if not pooled)
        logger.info("Shutdown complete.")

    @app.get("/health", tags=["System"])
    async def health_check_endpoint():
        """Provides a health check for the API and its dependencies."""
        current_env = get_os_env() # Get current state of env vars
        db_status = db_manager.health_check()

        # Add token caching status to health check
        token_caching_status = {
            "enabled": str_to_bool(current_env.get("TOKEN_CACHING_ENABLED", "True")),
            "refresh_interval": int(current_env.get("TOKEN_REFRESH_INTERVAL", "300")),
            "validation_threshold": int(current_env.get("TOKEN_VALIDATION_THRESHOLD", "600"))
        }

        return {
            "status": "healthy",
            "version": app.version,
            "proxy_enabled": str_to_bool(current_env.get("PROXY_ENABLED", "False")),
            "azure_endpoint": current_env.get("AZURE_ENDPOINT", "Not Set"),
            "active_model": current_env.get("MODEL_NAME", "gpt-4o-mini"),
            "postgresql_status": db_status.get("status", "unknown"),
            "token_caching": token_caching_status
        }

    @app.get("/", tags=["System"])
    async def root_endpoint():
        """Root endpoint providing basic API information."""
        current_env = get_os_env()
        return {
            "application_name": app.title,
            "version": app.version,
            "status": "API is operational",
            "documentation_url": "/docs",
            "proxy_enabled": str_to_bool(current_env.get("PROXY_ENABLED", "False")),
            "token_caching_enabled": str_to_bool(current_env.get("TOKEN_CACHING_ENABLED", "True"))
        }
        
    logger.info("FastAPI application created successfully.")
    return app

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Element Enhancement API")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (for development)")
    
    # Proxy settings
    proxy_env = str_to_bool(os.getenv("PROXY_ENABLED", "True"))
    parser.add_argument("--proxy", dest="proxy_enabled", action="store_true", default=proxy_env, help="Enable proxy")
    parser.add_argument("--no-proxy", dest="proxy_enabled", action="store_false", help="Disable proxy")
    
    # Monitoring
    monitoring_env = int(os.getenv("MONITORING_INTERVAL", "300"))
    parser.add_argument("--monitoring-interval", type=int, default=monitoring_env, help="System monitoring interval in seconds (0 to disable)")

    # Token caching settings
    parser.add_argument("--token-caching", dest="token_caching_enabled", action="store_true", default=True, help="Enable token caching")
    parser.add_argument("--no-token-caching", dest="token_caching_enabled", action="store_false", help="Disable token caching")
    parser.add_argument("--token-refresh-interval", type=int, default=int(os.getenv("TOKEN_REFRESH_INTERVAL", "300")), help="Token refresh interval in seconds")
    parser.add_argument("--token-validation-threshold", type=int, default=int(os.getenv("TOKEN_VALIDATION_THRESHOLD", "600")), help="Minimum token validity time in seconds")

    # Model settings
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="Azure OpenAI model to use")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "2000")), help="Maximum tokens for model responses")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.3")), help="Temperature for model responses")

    # Config file paths (can be overridden by environment variables)
    parser.add_argument("--config-file", type=str, default=os.getenv("ENV_CONFIG_PATH", "env/config.env"), help="Path to config.env file")
    parser.add_argument("--creds-file", type=str, default=os.getenv("ENV_CREDS_PATH", "env/credentials.env"), help="Path to credentials.env file")
    parser.add_argument("--cert-file", type=str, default=os.getenv("ENV_CERT_PATH", "env/cacert.pem"), help="Path to cacert.pem file")

    return parser.parse_args()

# Application instance for Uvicorn if run directly
# This ensures settings from args are applied if __name__ == "__main__"
# Otherwise, a default app instance is created for imports.
if os.getenv("_FASTAPI_RELOAD") == "true" or __name__ != "__main__":
    # When reloading or importing, create a default app.
    # Uvicorn reload will handle re-running __main__ block.
    # Set environment variables that would normally be set by args for consistency
    os.environ.setdefault("TOKEN_CACHING_ENABLED", "True")
    os.environ.setdefault("TOKEN_REFRESH_INTERVAL", "300")
    os.environ.setdefault("TOKEN_VALIDATION_THRESHOLD", "600")
    app = create_application(
        proxy_enabled=str_to_bool(os.getenv("PROXY_ENABLED", "True")),
        monitoring_interval=int(os.getenv("MONITORING_INTERVAL", "300")),
        token_refresh_interval=int(os.getenv("TOKEN_REFRESH_INTERVAL", "300"))
    )

if __name__ == "__main__":
    args = parse_arguments()

    # Set environment variables from arguments to be available for get_os_env()
    # These will be picked up by OSEnv when get_os_env() is called within create_application()
    os.environ["ENV_CONFIG_PATH"] = args.config_file
    os.environ["ENV_CREDS_PATH"] = args.creds_file
    os.environ["ENV_CERT_PATH"] = args.cert_file
    os.environ["PROXY_ENABLED"] = str(args.proxy_enabled) # OSEnv expects string "True" or "False"
    os.environ["MONITORING_INTERVAL"] = str(args.monitoring_interval)
    
    # Model settings
    os.environ["MODEL_NAME"] = args.model
    os.environ["MAX_TOKENS"] = str(args.max_tokens)
    os.environ["TEMPERATURE"] = str(args.temperature)
    
    # Token caching settings
    os.environ["TOKEN_CACHING_ENABLED"] = str(args.token_caching_enabled)
    os.environ["TOKEN_REFRESH_INTERVAL"] = str(args.token_refresh_interval)
    os.environ["TOKEN_VALIDATION_THRESHOLD"] = str(args.token_validation_threshold)

    # Log effective settings
    logger.info(f"--- Effective Runtime Configuration ---")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Proxy Enabled: {args.proxy_enabled}")
    logger.info(f"Monitoring Interval: {args.monitoring_interval}s")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max Tokens: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Token Caching Enabled: {args.token_caching_enabled}")
    logger.info(f"Token Refresh Interval: {args.token_refresh_interval}s")
    logger.info(f"Token Validation Threshold: {args.token_validation_threshold}s")
    logger.info(f"Config File: {args.config_file}")
    logger.info(f"Credentials File: {args.creds_file}")
    logger.info(f"Certificate File: {args.cert_file}")
    logger.info(f"------------------------------------")

    # Create the application instance with parsed arguments
    # This call to create_application will use the environment variables set just above
    app_instance = create_application(
        proxy_enabled=args.proxy_enabled,
        monitoring_interval=args.monitoring_interval,
        token_refresh_interval=args.token_refresh_interval
    )
    
    # For Uvicorn reload to work correctly when __main__ is executed.
    # It needs to find the 'app' variable in the module it's running.
    if args.reload:
        os.environ["_FASTAPI_RELOAD"] = "true" # Signal that we are in reload mode
        # When reloading, uvicorn expects the app as a string "module:app_variable_name"
        uvicorn.run("main:app_instance", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app_instance, host=args.host, port=args.port)