"""
FastAPI Application Factory

Creates and configures the FastAPI application with all routes and middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.observability import get_logger
from .core.database import init_db_pool, close_db_pool
from .auth.middleware import setup_middleware
from .auth.api import router as auth_router

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        Configured FastAPI app instance
    """
    # Create app
    app = FastAPI(
        title="Quantum Trading Platform",
        description="Enterprise-grade algorithmic trading platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        logger.info("Starting up Quantum Trading Platform...")
        
        # Initialize database pool
        await init_db_pool()
        
        logger.info("Startup complete!")
        
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("Shutting down Quantum Trading Platform...")
        
        # Close database pool
        await close_db_pool()
        
        logger.info("Shutdown complete!")
        
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return JSONResponse(
            content={
                "status": "healthy",
                "service": "quantum-trading-platform",
                "version": "1.0.0"
            }
        )
        
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Metrics endpoint for monitoring"""
        # TODO: Implement actual metrics collection
        return JSONResponse(
            content={
                "requests_total": 0,
                "requests_per_second": 0,
                "active_connections": 0,
                "error_rate": 0.0
            }
        )
        
    # Include routers
    app.include_router(auth_router)
    
    # TODO: Add more routers as they are implemented
    # app.include_router(trading_router)
    # app.include_router(portfolio_router)
    # app.include_router(market_data_router)
    # app.include_router(strategy_router)
    
    # Add exception handlers
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": "Not found"}
        )
        
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
        
    return app