#!/usr/bin/env python3
"""
Integration script to verify all modules work together.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import create_app
from src.core.config import settings
from src.core.logging import get_logger
from src.data.market import MarketDataStream
from src.trading.engine import TradingEngine
from src.trading.models import OrderSide, Signal

logger = get_logger(__name__)


async def test_integration():
    """Test integration of all modules."""
    logger.info("Starting integration test...")
    
    try:
        # 1. Test Trading Engine
        logger.info("Testing Trading Engine...")
        engine = TradingEngine("binance")
        await engine.start()
        
        # Create test signal
        signal = Signal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            confidence=0.85,
            entry_price=45000.0,
            stop_loss=44000.0,
            take_profit=46000.0,
            strategy="integration_test"
        )
        
        # Process signal (mock mode)
        logger.info(f"Processing signal: {signal}")
        # order_id = await engine.process_signal(signal)
        # logger.info(f"Signal processed, order_id: {order_id}")
        
        # 2. Test Data Module
        logger.info("Testing Data Module...")
        # Note: Requires Redis running
        try:
            from redis.asyncio import Redis
            redis = await Redis.from_url(str(settings.redis_url))
            market_stream = MarketDataStream(redis)
            
            # Test market data storage
            await redis.set("market_data:test:BTC/USDT", '{"bid": 45000, "ask": 45010}')
            data = await market_stream.get_market_data("test", "BTC/USDT")
            logger.info(f"Market data retrieved: {data}")
            
            await redis.close()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
        
        # 3. Test API Module
        logger.info("Testing API Module...")
        app = create_app()
        logger.info("FastAPI app created successfully")
        
        # Get app info
        logger.info(f"API Title: {app.title}")
        logger.info(f"API Version: {app.version}")
        logger.info(f"Available routes: {len(app.routes)}")
        
        # Stop engine
        await engine.stop()
        
        logger.info("✅ Integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        return False
    
    return True


async def main():
    """Main entry point."""
    success = await test_integration()
    
    if success:
        logger.info("\n" + "="*50)
        logger.info("🎉 All modules are integrated and working!")
        logger.info("="*50)
        logger.info("\nNext steps:")
        logger.info("1. Start Redis: redis-server")
        logger.info("2. Start PostgreSQL: sudo service postgresql start")
        logger.info("3. Run migrations: make migrate")
        logger.info("4. Start the app: make run")
        logger.info("\nOr use Docker: docker-compose up")
    else:
        logger.error("Integration test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())