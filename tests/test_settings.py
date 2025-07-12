"""
Test settings and configuration
"""

import os
from pathlib import Path

# Test environment
TEST_ENV = os.getenv("TEST_ENV", "unit")
CI_ENVIRONMENT = os.getenv("CI", "false").lower() == "true"

# Paths
BASE_DIR = Path(__file__).parent.parent
TEST_DIR = BASE_DIR / "tests"
FIXTURES_DIR = TEST_DIR / "fixtures"
TEST_DATA_DIR = TEST_DIR / "test_data"

# Database settings for testing
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://test:test@localhost:5432/quantum_trading_test"
)

# Redis settings for testing
TEST_REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    "redis://localhost:6379/15"  # Use DB 15 for tests
)

# RabbitMQ settings for testing
TEST_RABBITMQ_URL = os.getenv(
    "TEST_RABBITMQ_URL",
    "amqp://guest:guest@localhost:5672/test"
)

# Test exchange credentials (mock)
TEST_EXCHANGE_CREDENTIALS = {
    "binance": {
        "api_key": "test_binance_api_key",
        "api_secret": "test_binance_api_secret",
        "testnet": True
    },
    "coinbase": {
        "api_key": "test_coinbase_api_key",
        "api_secret": "test_coinbase_api_secret",
        "passphrase": "test_passphrase"
    }
}

# Performance test settings
PERFORMANCE_TEST_CONFIG = {
    "load_test_users": 100,
    "load_test_duration": 60,  # seconds
    "benchmark_iterations": 1000,
    "memory_limit_mb": 512
}

# Integration test settings
INTEGRATION_TEST_CONFIG = {
    "use_real_exchanges": False,
    "use_real_database": CI_ENVIRONMENT,
    "use_real_redis": CI_ENVIRONMENT,
    "use_real_rabbitmq": False,
    "cleanup_after_test": True
}

# Test data settings
TEST_DATA_CONFIG = {
    "sample_trades_count": 1000,
    "sample_orders_count": 500,
    "historical_days": 30,
    "ticker_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "test_portfolio_value": 100000.0
}

# Timeout settings
TEST_TIMEOUTS = {
    "unit_test": 5,  # seconds
    "integration_test": 30,
    "e2e_test": 60,
    "performance_test": 300
}

# Mock settings
MOCK_SETTINGS = {
    "mock_exchange_latency": 0.1,  # seconds
    "mock_database_latency": 0.05,
    "mock_redis_latency": 0.01,
    "mock_failure_rate": 0.1  # 10% failure rate for chaos testing
}

# Coverage settings
COVERAGE_CONFIG = {
    "min_coverage": 80,
    "fail_under": 80,
    "exclude_patterns": [
        "*/tests/*",
        "*/migrations/*",
        "*/__pycache__/*",
        "*/venv/*"
    ]
}

# Logging settings for tests
TEST_LOGGING = {
    "level": "DEBUG" if not CI_ENVIRONMENT else "WARNING",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "disable_existing_loggers": False,
    "log_to_file": CI_ENVIRONMENT,
    "log_file": TEST_DIR / "test_run.log"
}

# Feature flags for tests
TEST_FEATURES = {
    "test_real_exchanges": False,
    "test_ml_models": False,
    "test_heavy_computations": not CI_ENVIRONMENT,
    "test_external_apis": False,
    "test_notifications": False
}

# Benchmark thresholds
BENCHMARK_THRESHOLDS = {
    "order_placement_ms": 100,
    "position_calculation_ms": 50,
    "risk_check_ms": 30,
    "backtest_speed_trades_per_second": 1000,
    "websocket_latency_ms": 10
}

# Test categories to run
TEST_CATEGORIES = {
    "unit": True,
    "integration": not CI_ENVIRONMENT or TEST_ENV == "integration",
    "e2e": TEST_ENV == "e2e",
    "performance": TEST_ENV == "performance",
    "security": TEST_ENV == "security"
}

# Security test settings
SECURITY_TEST_CONFIG = {
    "test_sql_injection": True,
    "test_xss": True,
    "test_rate_limiting": True,
    "test_authentication": True,
    "test_authorization": True
}

# Fixture data paths
FIXTURE_FILES = {
    "sample_ohlcv": FIXTURES_DIR / "ohlcv_data.json",
    "sample_orderbook": FIXTURES_DIR / "orderbook_data.json",
    "sample_trades": FIXTURES_DIR / "trades_data.json",
    "sample_config": FIXTURES_DIR / "test_config.yaml"
}

# Create necessary directories
for directory in [TEST_DIR, FIXTURES_DIR, TEST_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Environment-specific overrides
if CI_ENVIRONMENT:
    # Override settings for CI
    TEST_DATABASE_URL = os.getenv("DATABASE_URL", TEST_DATABASE_URL)
    TEST_REDIS_URL = os.getenv("REDIS_URL", TEST_REDIS_URL)
    TEST_RABBITMQ_URL = os.getenv("RABBITMQ_URL", TEST_RABBITMQ_URL)


def get_test_config():
    """Get complete test configuration"""
    return {
        "environment": TEST_ENV,
        "ci": CI_ENVIRONMENT,
        "database_url": TEST_DATABASE_URL,
        "redis_url": TEST_REDIS_URL,
        "rabbitmq_url": TEST_RABBITMQ_URL,
        "timeouts": TEST_TIMEOUTS,
        "features": TEST_FEATURES,
        "categories": TEST_CATEGORIES
    }