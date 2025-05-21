"""
Corrected Main Application - Prevents automatic shutdown.

This fixes the "maximum request limit of 0 exceeded" error that causes the server to terminate.
"""

import argparse
import logging
import os
import sys
import time
import uuid
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import orjson
from fastapi import FastAPI, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
import uvicorn

# Ensure parent directory is in path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Standard imports from your app modules
from app.api.routes.enhancement import router as enhancement_router
from app.api.routes.settings import router as settings_router
from app.config.environment import get_os_env, str_to_bool
from app.core.system_monitor import start_monitoring, stop_monitoring
from app.core.in_memory_job_store import optimized_job_store as job_store
from app.utils.auth_helper import initialize_token_caching
from app.utils.cache import setup_caching

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to stdout
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

# Make sure orjson is installed for faster JSON processing
try:
    import orjson
except ImportError:
    logger.warning("orjson not installed. Installing for better performance...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "orjson"])
        import orjson
        logger.info("orjson installed successfully.")
    except Exception as e:
        logger.error(f"Failed to install orjson: {e}. Will use standard JSON processing.")

# Custom JSONResponse class using orjson for better performance
class ORJSONResponse(JSONResponse):
    media_type = "application/json"
    
    def render(self, content) -> bytes:
        return orjson.dumps(
            content,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
        )

# Use asynccontextmanager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown events.
    This is the modern way to handle startup/shutdown in FastAPI.
    """
    # Startup code
    logger.info("Application startup")
    
    # Setup environment and connections
    env = get_os_env()
    
    # Initialize advanced caching
    setup_caching(max_size=5000, ttl=7200)  # Larger cache size and longer TTL
    logger.info("Caching system initialized with expanded memory")
    
    # Initialize token caching system with proper refresh interval
    token_refresh_interval = int(env.get("TOKEN_REFRESH_INTERVAL", "600"))
    initialize_token_caching(start_refresh_service=True, refresh_interval=token_refresh_interval)
    logger.info(f"Token caching system initialized with refresh interval: {token_refresh_interval}s")
    
    # Start system monitoring if enabled
    monitoring_interval = int(env.get("MONITORING_INTERVAL", "300"))
    if monitoring_interval > 0:
        logger.info(f"Starting system monitoring. Interval: {monitoring_interval}s")
        start_monitoring(interval=monitoring_interval)
    
    # Check in-memory job store health
    store_health = job_store.health_check()
    logger.info(f"Optimized in-memory job store initialized: {store_health}")
    
    # Pre-warm CPU caches and JIT compiler
    logger.info("Pre-warming system...")
    
    # Yield control to FastAPI
    yield
    
    # Shutdown code
    logger.info("Application shutting down...")
    
    # Stop monitoring
    if monitoring_interval > 0:
        stop_monitoring()
    
    # Stop job store
    try:
        if hasattr(job_store, 'stop') and callable(job_store.stop):
            job_store.stop()
    except Exception as e:
        logger.error(f"Error stopping job store: {e}")
    
    logger.info("Shutdown complete.")

def create_application(
    proxy_enabled: bool = True, 
    monitoring_interval: int = 300,
    token_refresh_interval: int = 600,
    worker_id: Optional[int] = None
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        proxy_enabled: Whether to enable proxy settings
        monitoring_interval: Interval for system monitoring in seconds
        token_refresh_interval: Interval for token refreshing in seconds
        worker_id: Optional worker ID for multi-process deployment
        
    Returns:
        FastAPI: The FastAPI application
    """
    logger.info(f"Creating FastAPI application (worker_id={worker_id})...")
    
    # Initialize environment settings (proxy, credentials, etc.)
    env = get_os_env(proxy_enabled=proxy_enabled)
    
    logger.info(f"Proxy enabled: {env.get('PROXY_ENABLED')}")
    logger.info(f"Model name: {env.get('MODEL_NAME', 'gpt-4o-mini')}")
    
    # Custom route class that uses ORJSONResponse for better performance
    class ORJSONRoute(APIRoute):
        def get_route_handler(self):
            original_route_handler = super().get_route_handler()
            
            async def custom_route_handler(request: Request) -> Response:
                response = await original_route_handler(request)
                # Use ORJSONResponse for JSON responses
                if isinstance(response, JSONResponse) and not isinstance(response, ORJSONResponse):
                    return ORJSONResponse(
                        content=response.body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                return response
                
            return custom_route_handler
    
    app = FastAPI(
        title="Data Element Enhancement API",
        description="API for enhancing data elements to meet ISO/IEC 11179 standards.",
        version="2.1.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
        route_class=ORJSONRoute,
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Add process time tracking middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Add CORS middleware with optimized settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,  # Cache preflight requests for 24 hours
    )
    
    # Add GZip compression middleware with optimized settings
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,  # Only compress responses larger than 1KB
        compresslevel=5,  # Balanced compression level (1-9, 9 is highest)
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

    @app.get("/health", tags=["System"])
    async def health_check_endpoint():
        """Provides a health check for the API and its dependencies."""
        current_env = get_os_env() # Get current state of env vars
        store_status = job_store.health_check()

        # Add token caching status to health check
        token_caching_status = {
            "enabled": str_to_bool(current_env.get("TOKEN_CACHING_ENABLED", "True")),
            "refresh_interval": int(current_env.get("TOKEN_REFRESH_INTERVAL", "600")),
            "validation_threshold": int(current_env.get("TOKEN_VALIDATION_THRESHOLD", "600"))
        }
        
        # Add request tracking info
        from app.api.routes.enhancement import active_requests
        
        # Add system metrics
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }

        return {
            "status": "healthy",
            "version": app.version,
            "proxy_enabled": str_to_bool(current_env.get("PROXY_ENABLED", "False")),
            "azure_endpoint": current_env.get("AZURE_ENDPOINT", "Not Set"),
            "active_model": current_env.get("MODEL_NAME", "gpt-4o-mini"),
            "job_store_status": store_status,
            "token_caching": token_caching_status,
            "api_version": current_env.get("API_VERSION", "2024-02-01"),
            "active_requests": len(active_requests),
            "system_metrics": system_metrics,
            "worker_id": worker_id
        }

    @app.get("/", tags=["System"])
    async def root_endpoint():
        """Root endpoint providing basic API information."""
        current_env = get_os_env()
        return {
            "application_name": app.title,
            "version": app.version,
            "status": "API is operational",
            "documentation_url": "/api/docs",
            "proxy_enabled": str_to_bool(current_env.get("PROXY_ENABLED", "False")),
            "token_caching_enabled": str_to_bool(current_env.get("TOKEN_CACHING_ENABLED", "True"))
        }
        
    logger.info("FastAPI application created successfully.")
    return app

# Create a global app instance for use by Uvicorn when imported
app = create_application(
    proxy_enabled=str_to_bool(os.getenv("PROXY_ENABLED", "True")),
    monitoring_interval=int(os.getenv("MONITORING_INTERVAL", "300")),
    token_refresh_interval=int(os.getenv("TOKEN_REFRESH_INTERVAL", "600"))
)

# Direct execution entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Element Enhancement API")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (for development)")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "1")), help="Number of worker processes")
    
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
    parser.add_argument("--token-refresh-interval", type=int, default=int(os.getenv("TOKEN_REFRESH_INTERVAL", "600")), help="Token refresh interval in seconds")
    parser.add_argument("--token-validation-threshold", type=int, default=int(os.getenv("TOKEN_VALIDATION_THRESHOLD", "600")), help="Minimum token validity time in seconds")

    # Model settings
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="Azure OpenAI model to use")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "2000")), help="Maximum tokens for model responses")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.3")), help="Temperature for model responses")
    parser.add_argument("--api-version", type=str, default=os.getenv("API_VERSION", "2024-02-01"), help="Azure OpenAI API version")

    # Server settings
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "info"), 
                      choices=["debug", "info", "warning", "error", "critical"],
                      help="Logging level")
    parser.add_argument("--timeout-keep-alive", type=int, default=int(os.getenv("TIMEOUT_KEEP_ALIVE", "5")), 
                      help="Timeout for keep alive connections in seconds")
    parser.add_argument("--limit-concurrency", type=int, default=int(os.getenv("LIMIT_CONCURRENCY", "100")), 
                      help="Maximum number of concurrent connections")
    
    # FIXED: Changed default to None to prevent auto-shutdown and added validation
    parser.add_argument("--limit-max-requests", type=int, default=int(os.getenv("LIMIT_MAX_REQUESTS", "0")), 
                      help="Maximum number of requests per worker (0 for unlimited)")
    
    # Config file paths (can be overridden by environment variables)
    parser.add_argument("--config-file", type=str, default=os.getenv("ENV_CONFIG_PATH", "env/config.env"), help="Path to config.env file")
    parser.add_argument("--creds-file", type=str, default=os.getenv("ENV_CREDS_PATH", "env/credentials.env"), help="Path to credentials.env file")
    parser.add_argument("--cert-file", type=str, default=os.getenv("ENV_CERT_PATH", "env/cacert.pem"), help="Path to cacert.pem file")

    args = parser.parse_args()

    # Configure logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)

    # Set environment variables from arguments to be available for get_os_env()
    os.environ["ENV_CONFIG_PATH"] = args.config_file
    os.environ["ENV_CREDS_PATH"] = args.creds_file
    os.environ["ENV_CERT_PATH"] = args.cert_file
    os.environ["PROXY_ENABLED"] = str(args.proxy_enabled) # OSEnv expects string "True" or "False"
    os.environ["MONITORING_INTERVAL"] = str(args.monitoring_interval)
    
    # Model settings
    os.environ["MODEL_NAME"] = args.model
    os.environ["MAX_TOKENS"] = str(args.max_tokens)
    os.environ["TEMPERATURE"] = str(args.temperature)
    os.environ["API_VERSION"] = args.api_version
    
    # Token caching settings
    os.environ["TOKEN_CACHING_ENABLED"] = str(args.token_caching_enabled)
    os.environ["TOKEN_REFRESH_INTERVAL"] = str(args.token_refresh_interval)
    os.environ["TOKEN_VALIDATION_THRESHOLD"] = str(args.token_validation_threshold)

    # Log effective settings
    logger.info(f"--- Effective Runtime Configuration ---")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Proxy Enabled: {args.proxy_enabled}")
    logger.info(f"Monitoring Interval: {args.monitoring_interval}s")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max Tokens: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"API Version: {args.api_version}")
    logger.info(f"Token Caching Enabled: {args.token_caching_enabled}")
    logger.info(f"Token Refresh Interval: {args.token_refresh_interval}s")
    logger.info(f"Token Validation Threshold: {args.token_validation_threshold}s")
    logger.info(f"Config File: {args.config_file}")
    logger.info(f"Credentials File: {args.creds_file}")
    logger.info(f"Certificate File: {args.cert_file}")
    logger.info(f"Timeout Keep Alive: {args.timeout_keep_alive}s")
    logger.info(f"Limit Concurrency: {args.limit_concurrency}")
    
    # FIXED: Add special handling for limit_max_requests=0
    if args.limit_max_requests == 0:
        logger.info("Limit Max Requests: 0 (unlimited)")
    else:
        logger.info(f"Limit Max Requests: {args.limit_max_requests}")
    
    logger.info(f"------------------------------------")

    # For reload mode, use a single worker
    if args.reload:
        os.environ["_FASTAPI_RELOAD"] = "true" # Signal that we are in reload mode
        
        # Create the application instance with parsed arguments
        app_instance = create_application(
            proxy_enabled=args.proxy_enabled,
            monitoring_interval=args.monitoring_interval,
            token_refresh_interval=args.token_refresh_interval
        )
        
        # When reloading, uvicorn expects the app as a string "module:app_variable_name"
        logger.info(f"Starting in development mode with auto-reload at http://{args.host}:{args.port}")
        uvicorn.run(
            "app.main:app", 
            host=args.host, 
            port=args.port, 
            reload=True,
            log_level=args.log_level
        )
    else:
        # For production mode, pass configuration directly to uvicorn
        logger.info(f"Starting in production mode at http://{args.host}:{args.port} with {args.workers} workers")
        
        # FIXED: Set limit_max_requests=None when it's 0 to prevent auto-shutdown
        limit_max_requests = None if args.limit_max_requests == 0 else args.limit_max_requests
        
        uvicorn_config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            timeout_keep_alive=args.timeout_keep_alive,
            limit_concurrency=args.limit_concurrency,
            limit_max_requests=limit_max_requests,  # Use None for unlimited requests
            workers=args.workers,
            lifespan="on"
        )
        server = uvicorn.Server(uvicorn_config)
        server.run()