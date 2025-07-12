#!/usr/bin/env python3
"""
Basic import checker to verify all modules can be imported
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Track results
results = {"success": [], "failed": []}


def test_import(module_path: str, items: list = None):
    """Test importing a module and optionally specific items"""
    try:
        if items:
            exec(f"from {module_path} import {', '.join(items)}")
        else:
            exec(f"import {module_path}")
        results["success"].append(module_path)
        return True
    except Exception as e:
        results["failed"].append({
            "module": module_path,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        return False


print("🔍 Checking Python module imports...\n")

# Core Architecture
print("📦 Core Architecture:")
test_import("src.core.architecture", ["DIContainer", "injectable", "inject", "Scope"])
test_import("src.core.architecture.base_repository", ["BaseRepository"])
test_import("src.core.architecture.unit_of_work", ["UnitOfWork"])
test_import("src.core.architecture.event_bus", ["EventBus"])
test_import("src.core.architecture.circuit_breaker", ["CircuitBreaker"])
test_import("src.core.architecture.rate_limiter", ["RateLimiter"])
test_import("src.core.architecture.factory_registry", ["FactoryRegistry"])

# Core Messaging
print("\n📦 Core Messaging:")
test_import("src.core.messaging", ["MessageBroker", "InMemoryBroker", "Event"])
test_import("src.core.messaging.message_broker", ["Message", "DeliveryMode"])
test_import("src.core.messaging.redis_broker", ["RedisBroker"])
test_import("src.core.messaging.rabbitmq_broker", ["RabbitMQBroker"])

# Core Observability
print("\n📦 Core Observability:")
test_import("src.core.observability.logger", ["get_logger", "StructuredLogger"])
test_import("src.core.observability.metrics", ["MetricsCollector"])
test_import("src.core.observability.tracing", ["Tracer"])
test_import("src.core.observability.health", ["HealthChecker"])

# Core Configuration
print("\n📦 Core Configuration:")
test_import("src.core.configuration.config_manager", ["ConfigManager"])
test_import("src.core.configuration.models", ["TradingConfig", "ExchangeConfig"])

# Core Cache
print("\n📦 Core Cache:")
test_import("src.core.cache.cache_manager", ["CacheManager"])
test_import("src.core.cache.decorators", ["cache", "invalidate_cache"])

# Core Factories
print("\n📦 Core Factories:")
test_import("src.core.factories.exchange_factory", ["ExchangeFactory"])
test_import("src.core.factories.strategy_factory", ["StrategyFactory"])

# Models
print("\n📦 Models:")
test_import("src.models.trading", ["Order", "Trade", "OrderType", "OrderSide"])
test_import("src.models.exchange", ["ExchangeCredentials", "ExchangeType"])
test_import("src.models.risk", ["RiskMetrics", "RiskLimits"])

# Exchange
print("\n📦 Exchange:")
test_import("src.exchange", ["ExchangeInterface", "ExchangeManager"])
test_import("src.exchange.binance_exchange", ["BinanceExchange"])

# Strategies
print("\n📦 Strategies:")
test_import("src.strategies", ["BaseStrategy", "StrategyManager"])
test_import("src.strategies.indicators", ["TechnicalIndicators"])
test_import("src.strategies.mean_reversion_strategy", ["MeanReversionStrategy"])

# Orders
print("\n📦 Orders:")
test_import("src.orders", ["OrderManager"])
test_import("src.orders.execution_algorithms", ["TWAPAlgorithm", "VWAPAlgorithm"])
test_import("src.orders.slippage_models", ["SlippageModel"])

# Positions
print("\n📦 Positions:")
test_import("src.positions", ["PositionTracker", "PortfolioManager"])

# Risk
print("\n📦 Risk:")
test_import("src.risk", ["RiskManager", "RiskModels"])
test_import("src.risk.stop_loss_manager", ["StopLossManager"])

# Backtesting
print("\n📦 Backtesting:")
test_import("src.backtesting", ["BacktestEngine", "DataProvider"])
test_import("src.backtesting.event_simulator", ["EventSimulator"])
test_import("src.backtesting.portfolio_simulator", ["PortfolioSimulator"])

# Market Data
print("\n📦 Market Data:")
test_import("src.market_data", ["MarketDataAggregator", "DataNormalizer"])
test_import("src.market_data.collector", ["MarketDataCollector"])
test_import("src.market_data.storage", ["MarketDataStorage"])

# Analytics
print("\n📦 Analytics:")
test_import("src.analytics", ["PerformanceAnalyzer", "MetricsCalculator"])
test_import("src.analytics.dashboard", ["Dashboard"])
test_import("src.analytics.report_generator", ["ReportGenerator"])

# Print results
print("\n" + "="*60)
print("📊 IMPORT TEST RESULTS:")
print("="*60)

if results["success"]:
    print(f"\n✅ Successful imports: {len(results['success'])}")
    for module in results["success"]:
        print(f"   ✓ {module}")

if results["failed"]:
    print(f"\n❌ Failed imports: {len(results['failed'])}")
    for failure in results["failed"]:
        print(f"\n   ✗ {failure['module']}")
        print(f"     Error: {failure['error']}")
        if "--verbose" in sys.argv:
            print(f"     Traceback:\n{failure['traceback']}")

print("\n" + "="*60)
if not results["failed"]:
    print("🎉 All imports successful! The project structure is valid.")
else:
    print(f"⚠️  {len(results['failed'])} import errors found. Fix these before running tests.")
    print("\nCommon fixes:")
    print("- Check for missing __init__.py files")
    print("- Verify module and class names match")
    print("- Ensure dependencies are installed")
    print("- Check for circular imports")

sys.exit(0 if not results["failed"] else 1)