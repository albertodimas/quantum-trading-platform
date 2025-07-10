"""
Main entry point for Quantum Trading Platform.
"""

import asyncio
import signal
import sys
from typing import Optional

import uvicorn

from src.api import create_app
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Global app instance
app = Optional[None]


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(
        "Starting Quantum Trading Platform",
        environment=settings.environment,
        debug=settings.debug,
    )
    
    # Create FastAPI app
    global app
    app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=False,  # We use our own logging
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        loop="uvloop",
        interface="asgi3",
    )
    
    # Run server
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()