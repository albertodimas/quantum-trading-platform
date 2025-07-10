"""
FastAPI application factory and configuration.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from src.api.routers import (
    auth,
    health,
    market,
    orders,
    portfolio,
    strategies,
    trading,
    websocket,
)
from src.core.config import settings
from src.core.logging import get_logger
from src.data.market import MarketDataStream
from src.trading.engine import TradingEngine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Quantum Trading Platform API")
    
    # Initialize Redis
    app.state.redis = await Redis.from_url(
        str(settings.redis_url),
        decode_responses=settings.redis_decode_responses,
    )
    
    # Initialize market data stream
    app.state.market_stream = MarketDataStream(app.state.redis)
    
    # Initialize trading engine
    app.state.trading_engine = TradingEngine("binance")
    await app.state.trading_engine.start()
    
    # Initialize WebSocket manager
    from src.api.websocket import WebSocketManager
    app.state.ws_manager = WebSocketManager(app.state.redis)
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    
    # Stop services
    await app.state.trading_engine.stop()
    await app.state.market_stream.stop()
    await app.state.redis.close()
    
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Professional AI-powered algorithmic trading platform",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Add middlewares
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(
        auth.router,
        prefix=f"{settings.api_prefix}/auth",
        tags=["authentication"],
    )
    app.include_router(
        market.router,
        prefix=f"{settings.api_prefix}/market",
        tags=["market"],
    )
    app.include_router(
        trading.router,
        prefix=f"{settings.api_prefix}/trading",
        tags=["trading"],
    )
    app.include_router(
        orders.router,
        prefix=f"{settings.api_prefix}/orders",
        tags=["orders"],
    )
    app.include_router(
        portfolio.router,
        prefix=f"{settings.api_prefix}/portfolio",
        tags=["portfolio"],
    )
    app.include_router(
        strategies.router,
        prefix=f"{settings.api_prefix}/strategies",
        tags=["strategies"],
    )
    app.include_router(
        websocket.router,
        prefix="/ws",
        tags=["websocket"],
    )
    
    # Mount Prometheus metrics endpoint
    if settings.enable_metrics:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    
    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return {
            "error": "Invalid value",
            "message": str(exc),
        }, 400
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return {
            "error": "Internal server error",
            "message": "An unexpected error occurred",
        }, 500
    
    return app