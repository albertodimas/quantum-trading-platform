"""
Test module imports to verify all components can be imported without errors
"""

import pytest


@pytest.mark.unit
def test_core_architecture_imports():
    """Test core architecture imports"""
    # These imports should not raise any exceptions
    from src.core.architecture import (
        DIContainer,
        injectable,
        inject,
        Scope,
        BaseRepository,
        UnitOfWork,
        EventBus,
        CircuitBreaker,
        RateLimiter,
        FactoryRegistry
    )
    
    assert DIContainer is not None
    assert injectable is not None


@pytest.mark.unit
def test_core_messaging_imports():
    """Test messaging system imports"""
    from src.core.messaging import (
        MessageBroker,
        InMemoryBroker,
        Event,
        Message,
        DeliveryMode,
        MessagePriority
    )
    
    assert MessageBroker is not None
    assert InMemoryBroker is not None


@pytest.mark.unit
def test_core_observability_imports():
    """Test observability imports"""
    from src.core.observability import (
        StructuredLogger,
        get_logger,
        MetricsCollector,
        Tracer,
        HealthChecker
    )
    
    assert get_logger is not None
    assert MetricsCollector is not None


@pytest.mark.unit
def test_core_configuration_imports():
    """Test configuration imports"""
    from src.core.configuration import (
        ConfigManager,
        TradingConfig,
        ExchangeConfig,
        RiskConfig
    )
    
    assert ConfigManager is not None
    assert TradingConfig is not None


@pytest.mark.unit
def test_core_cache_imports():
    """Test cache imports"""
    from src.core.cache import (
        CacheManager,
        cache,
        invalidate_cache,
        cache_result
    )
    
    assert CacheManager is not None
    assert cache is not None


@pytest.mark.unit
def test_exchange_imports():
    """Test exchange imports"""
    from src.exchange import (
        ExchangeInterface,
        ExchangeManager,
        BinanceExchange,
        ExchangeCredentials,
        ExchangeType
    )
    
    assert ExchangeInterface is not None
    assert BinanceExchange is not None


@pytest.mark.unit
def test_strategy_imports():
    """Test strategy imports"""
    from src.strategies import (
        BaseStrategy,
        StrategyManager,
        TechnicalIndicators,
        MeanReversionStrategy
    )
    
    assert BaseStrategy is not None
    assert TechnicalIndicators is not None


@pytest.mark.unit
def test_order_imports():
    """Test order management imports"""
    from src.orders import (
        OrderManager,
        ExecutionAlgorithm,
        TWAPAlgorithm,
        VWAPAlgorithm,
        SlippageModel
    )
    
    assert OrderManager is not None
    assert TWAPAlgorithm is not None


@pytest.mark.unit
def test_position_imports():
    """Test position tracking imports"""
    from src.positions import (
        PositionTracker,
        PortfolioManager,
        Position,
        PositionStatus
    )
    
    assert PositionTracker is not None
    assert PortfolioManager is not None


@pytest.mark.unit
def test_risk_imports():
    """Test risk management imports"""
    from src.risk import (
        RiskManager,
        RiskModels,
        StopLossManager,
        RiskLimits,
        VaRModel
    )
    
    assert RiskManager is not None
    assert VaRModel is not None


@pytest.mark.unit
def test_backtesting_imports():
    """Test backtesting imports"""
    from src.backtesting import (
        BacktestEngine,
        DataProvider,
        EventSimulator,
        PortfolioSimulator,
        ExecutionSimulator,
        PerformanceAnalyzer
    )
    
    assert BacktestEngine is not None
    assert DataProvider is not None


@pytest.mark.unit
def test_market_data_imports():
    """Test market data imports"""
    from src.market_data import (
        MarketDataAggregator,
        DataNormalizer,
        MarketDataCollector,
        DataDistributor,
        MarketDataStorage,
        DataQualityChecker
    )
    
    assert MarketDataAggregator is not None
    assert DataNormalizer is not None


@pytest.mark.unit
def test_analytics_imports():
    """Test analytics imports"""
    from src.analytics import (
        PerformanceAnalyzer,
        MetricsCalculator,
        Dashboard,
        ReportGenerator,
        PortfolioAnalytics,
        TradeAnalytics,
        RiskAnalytics
    )
    
    assert PerformanceAnalyzer is not None
    assert Dashboard is not None


@pytest.mark.unit
def test_model_imports():
    """Test model imports"""
    from src.models import (
        Order,
        Trade,
        OrderType,
        OrderSide,
        OrderStatus,
        ExchangeCredentials,
        RiskMetrics
    )
    
    assert Order is not None
    assert Trade is not None
    assert OrderType is not None


if __name__ == "__main__":
    # Run basic import test
    print("Testing imports...")
    
    test_core_architecture_imports()
    print("✓ Core architecture imports OK")
    
    test_core_messaging_imports()
    print("✓ Core messaging imports OK")
    
    test_core_observability_imports()
    print("✓ Core observability imports OK")
    
    test_core_configuration_imports()
    print("✓ Core configuration imports OK")
    
    test_core_cache_imports()
    print("✓ Core cache imports OK")
    
    test_exchange_imports()
    print("✓ Exchange imports OK")
    
    test_strategy_imports()
    print("✓ Strategy imports OK")
    
    test_order_imports()
    print("✓ Order management imports OK")
    
    test_position_imports()
    print("✓ Position tracking imports OK")
    
    test_risk_imports()
    print("✓ Risk management imports OK")
    
    test_backtesting_imports()
    print("✓ Backtesting imports OK")
    
    test_market_data_imports()
    print("✓ Market data imports OK")
    
    test_analytics_imports()
    print("✓ Analytics imports OK")
    
    test_model_imports()
    print("✓ Model imports OK")
    
    print("\n✅ All imports successful!")