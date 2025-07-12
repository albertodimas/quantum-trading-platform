"""
FastAPI Application Factory

Creates and configures the FastAPI application with all routes and middleware.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .core.observability import setup_logging, get_logger
from .core.database import init_database, close_database
from .auth.middleware import AuthenticationMiddleware
from .auth.api import router as auth_router

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize database
    await init_database()
    
    # TODO: Initialize other services
    # - Redis cache
    # - Message queue connections
    # - Exchange connections
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Close database connections
    await close_database()
    
    # TODO: Cleanup other services


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Create app instance
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url=f"{settings.api_prefix}/docs" if not settings.is_production else None,
        redoc_url=f"{settings.api_prefix}/redoc" if not settings.is_production else None,
        openapi_url=f"{settings.api_prefix}/openapi.json" if not settings.is_production else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add authentication middleware
    app.add_middleware(AuthenticationMiddleware)
    
    # Add security headers middleware
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
    
    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "environment": settings.environment
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        # TODO: Add actual health checks
        return {
            "status": "healthy",
            "checks": {
                "database": "ok",
                "redis": "ok",
                "exchanges": "ok"
            }
        }
    
    # Include routers
    app.include_router(auth_router, prefix=f"{settings.api_prefix}/auth", tags=["auth"])
    
    # TODO: Include other routers
    # app.include_router(trading_router, prefix=f"{settings.api_prefix}/trading", tags=["trading"])
    # app.include_router(portfolio_router, prefix=f"{settings.api_prefix}/portfolio", tags=["portfolio"])
    # app.include_router(strategy_router, prefix=f"{settings.api_prefix}/strategies", tags=["strategies"])
    # app.include_router(market_data_router, prefix=f"{settings.api_prefix}/market", tags=["market"])
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )