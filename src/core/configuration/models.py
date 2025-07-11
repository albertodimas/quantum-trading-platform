"""
Pydantic configuration models with validation and type safety.

Provides strongly-typed configuration models for all system components with:
- Automatic validation and type checking
- Environment variable mapping
- Default values and constraints
- Nested configuration support
- Documentation and examples
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import timedelta
from decimal import Decimal

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.env_settings import BaseSettings


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    JSON = "json"
    CONSOLE = "console"


class DatabaseEngine(str, Enum):
    """Supported database engines."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"


class CacheBackend(str, Enum):
    """Cache backend types."""
    REDIS = "redis"
    MEMORY = "memory"
    MEMCACHED = "memcached"


class ExchangeType(str, Enum):
    """Exchange types."""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    MARGIN = "margin"


# Database Configuration
class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    engine: DatabaseEngine = DatabaseEngine.POSTGRESQL
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username") 
    password: str = Field(..., description="Database password")
    
    # Connection pool settings
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=300, description="Connection recycle time")
    
    # SSL/TLS settings
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    ssl_cert: Optional[Path] = Field(default=None, description="SSL certificate path")
    ssl_key: Optional[Path] = Field(default=None, description="SSL key path")
    ssl_ca: Optional[Path] = Field(default=None, description="SSL CA path")
    
    # Query settings
    query_timeout: int = Field(default=30, ge=1, description="Query timeout in seconds")
    statement_timeout: int = Field(default=60, ge=1, description="Statement timeout in seconds")
    
    @property
    def url(self) -> str:
        """Get database URL."""
        return f"{self.engine.value}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


# Redis Configuration
class RedisConfig(BaseModel):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    database: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    # Connection settings
    max_connections: int = Field(default=50, ge=1, le=1000, description="Maximum connections")
    connection_timeout: int = Field(default=5, ge=1, description="Connection timeout in seconds")
    socket_timeout: int = Field(default=5, ge=1, description="Socket timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    
    # Cluster settings
    cluster_mode: bool = Field(default=False, description="Enable cluster mode")
    cluster_nodes: List[str] = Field(default_factory=list, description="Cluster node addresses")
    
    # Cache settings
    default_ttl: int = Field(default=3600, ge=1, description="Default TTL in seconds")
    key_prefix: str = Field(default="qtp:", description="Key prefix")
    
    # Compression
    compression_enabled: bool = Field(default=True, description="Enable compression")
    compression_min_size: int = Field(default=1024, ge=1, description="Minimum size for compression")
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


# Security Configuration
class SecurityConfig(BaseModel):
    """Security and authentication configuration."""
    
    # Encryption
    secret_key: str = Field(..., min_length=32, description="Application secret key")
    encryption_algorithm: str = Field(default="HS256", description="JWT encryption algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, ge=1, description="Refresh token expiry")
    
    # API Security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Rate limit per minute")
    max_request_size: int = Field(default=10_000_000, ge=1, description="Max request size in bytes")
    
    # CORS
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS origins")
    cors_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"], description="CORS methods")
    cors_headers: List[str] = Field(default_factory=lambda: ["*"], description="CORS headers")
    
    # Password requirements
    password_min_length: int = Field(default=8, ge=6, description="Minimum password length")
    password_require_uppercase: bool = Field(default=True, description="Require uppercase letters")
    password_require_lowercase: bool = Field(default=True, description="Require lowercase letters")
    password_require_numbers: bool = Field(default=True, description="Require numbers")
    password_require_symbols: bool = Field(default=True, description="Require symbols")
    
    # Session security
    session_cookie_secure: bool = Field(default=True, description="Secure session cookies")
    session_cookie_httponly: bool = Field(default=True, description="HTTP-only session cookies")
    session_cookie_samesite: str = Field(default="strict", description="SameSite cookie policy")


# Trading Configuration
class TradingConfig(BaseModel):
    """Trading system configuration."""
    
    # Trading parameters
    default_order_size: Decimal = Field(default=Decimal("100"), gt=0, description="Default order size")
    max_position_size: Decimal = Field(default=Decimal("10000"), gt=0, description="Maximum position size")
    max_daily_trades: int = Field(default=100, ge=1, description="Maximum daily trades")
    
    # Risk management
    max_portfolio_risk: Decimal = Field(default=Decimal("0.02"), gt=0, le=1, description="Maximum portfolio risk")
    stop_loss_percentage: Decimal = Field(default=Decimal("0.05"), gt=0, le=1, description="Stop loss percentage")
    take_profit_percentage: Decimal = Field(default=Decimal("0.1"), gt=0, description="Take profit percentage")
    
    # Execution settings
    order_timeout_seconds: int = Field(default=30, ge=1, description="Order timeout in seconds")
    price_precision: int = Field(default=8, ge=2, le=12, description="Price precision")
    quantity_precision: int = Field(default=8, ge=2, le=12, description="Quantity precision")
    
    # Market data
    market_data_timeout: int = Field(default=5, ge=1, description="Market data timeout in seconds")
    price_staleness_threshold: int = Field(default=10, ge=1, description="Price staleness threshold in seconds")
    
    # Backtesting
    backtest_start_date: Optional[str] = Field(default=None, description="Backtest start date (YYYY-MM-DD)")
    backtest_end_date: Optional[str] = Field(default=None, description="Backtest end date (YYYY-MM-DD)")
    backtest_initial_capital: Decimal = Field(default=Decimal("100000"), gt=0, description="Initial capital for backtesting")
    
    @validator('max_portfolio_risk', 'stop_loss_percentage')
    def validate_percentages(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Percentage must be between 0 and 1')
        return v


# Exchange Configuration
class ExchangeConfig(BaseModel):
    """Exchange connection configuration."""
    
    name: str = Field(..., description="Exchange name")
    exchange_type: ExchangeType = Field(default=ExchangeType.SPOT, description="Exchange type")
    api_key: str = Field(..., description="API key")
    api_secret: str = Field(..., description="API secret")
    api_passphrase: Optional[str] = Field(default=None, description="API passphrase (if required)")
    
    # Connection settings
    base_url: str = Field(..., description="Exchange base URL")
    websocket_url: Optional[str] = Field(default=None, description="WebSocket URL")
    timeout: int = Field(default=10, ge=1, description="Request timeout in seconds")
    
    # Rate limiting
    requests_per_second: int = Field(default=10, ge=1, description="Requests per second limit")
    burst_limit: int = Field(default=50, ge=1, description="Burst request limit")
    
    # Trading settings
    sandbox_mode: bool = Field(default=True, description="Use sandbox/testnet")
    supported_symbols: List[str] = Field(default_factory=list, description="Supported trading symbols")
    default_symbol: str = Field(default="BTC/USDT", description="Default trading symbol")
    
    # Fees
    maker_fee: Decimal = Field(default=Decimal("0.001"), ge=0, description="Maker fee percentage")
    taker_fee: Decimal = Field(default=Decimal("0.001"), ge=0, description="Taker fee percentage")
    
    @validator('requests_per_second', 'burst_limit')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v


# Risk Management Configuration
class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    # Position limits
    max_position_value: Decimal = Field(default=Decimal("50000"), gt=0, description="Maximum position value")
    max_positions_per_symbol: int = Field(default=3, ge=1, description="Maximum positions per symbol")
    max_total_positions: int = Field(default=20, ge=1, description="Maximum total positions")
    
    # Risk metrics
    var_confidence_level: Decimal = Field(default=Decimal("0.95"), gt=0, lt=1, description="VaR confidence level")
    var_time_horizon_days: int = Field(default=1, ge=1, description="VaR time horizon in days")
    max_drawdown_percentage: Decimal = Field(default=Decimal("0.1"), gt=0, le=1, description="Maximum drawdown")
    
    # Correlation limits
    max_correlation: Decimal = Field(default=Decimal("0.8"), ge=0, le=1, description="Maximum position correlation")
    correlation_window_days: int = Field(default=30, ge=7, description="Correlation calculation window")
    
    # Stop-loss and circuit breakers
    enable_stop_loss: bool = Field(default=True, description="Enable stop-loss orders")
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breakers")
    circuit_breaker_threshold: Decimal = Field(default=Decimal("0.05"), gt=0, description="Circuit breaker threshold")
    
    # Risk monitoring
    risk_check_interval: int = Field(default=60, ge=1, description="Risk check interval in seconds")
    alert_threshold: Decimal = Field(default=Decimal("0.8"), gt=0, le=1, description="Risk alert threshold")


# Observability Configuration
class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration."""
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    log_format: LogFormat = Field(default=LogFormat.JSON, description="Log format")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    log_rotation_size: str = Field(default="100MB", description="Log rotation size")
    log_retention_days: int = Field(default=30, ge=1, description="Log retention in days")
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8000, ge=1024, le=65535, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    
    # Tracing
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    tracing_endpoint: Optional[str] = Field(default=None, description="Tracing collector endpoint")
    trace_sample_rate: float = Field(default=0.1, ge=0, le=1, description="Trace sampling rate")
    
    # Health checks
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_port: int = Field(default=8001, ge=1024, le=65535, description="Health check port")
    health_check_interval: int = Field(default=30, ge=5, description="Health check interval in seconds")
    
    # Alerting
    enable_alerts: bool = Field(default=True, description="Enable alerting")
    alert_webhook_url: Optional[str] = Field(default=None, description="Alert webhook URL")
    alert_email: Optional[str] = Field(default=None, description="Alert email address")


# Main Application Configuration
class AppConfig(BaseSettings):
    """Main application configuration that combines all sub-configurations."""
    
    # Application settings
    app_name: str = Field(default="Quantum Trading Platform", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    
    # Configuration file paths
    config_file: Optional[Path] = Field(default=None, description="Configuration file path")
    secrets_file: Optional[Path] = Field(default=None, description="Secrets file path")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    redis: RedisConfig = Field(default_factory=RedisConfig, description="Redis configuration")
    security: SecurityConfig = Field(..., description="Security configuration")
    trading: TradingConfig = Field(default_factory=TradingConfig, description="Trading configuration")
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk management configuration")
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig, description="Observability configuration")
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict, description="Exchange configurations")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        allow_population_by_field_name = True
        validate_assignment = True
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            """Customize configuration source priority."""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )
    
    @root_validator
    def validate_environment_consistency(cls, values):
        """Validate environment-specific configurations."""
        environment = values.get('environment')
        debug = values.get('debug')
        
        if environment == Environment.PRODUCTION:
            if debug:
                raise ValueError('Debug mode cannot be enabled in production')
            
            security = values.get('security')
            if security and not security.session_cookie_secure:
                raise ValueError('Secure cookies must be enabled in production')
        
        return values
    
    @validator('port')
    def validate_port_range(cls, v):
        """Validate port is in acceptable range."""
        if not 1024 <= v <= 65535:
            raise ValueError('Port must be between 1024 and 65535')
        return v
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING


# Configuration factory functions
def create_development_config() -> AppConfig:
    """Create development configuration with sensible defaults."""
    return AppConfig(
        environment=Environment.DEVELOPMENT,
        debug=True,
        security=SecurityConfig(
            secret_key="development-secret-key-change-in-production-" + "x" * 10,
            session_cookie_secure=False,
            cors_origins=["http://localhost:3000", "http://localhost:8080"]
        ),
        observability=ObservabilityConfig(
            log_level=LogLevel.DEBUG,
            log_format=LogFormat.CONSOLE,
            enable_metrics=True,
            enable_tracing=True,
            trace_sample_rate=1.0
        )
    )


def create_production_config() -> AppConfig:
    """Create production configuration template."""
    return AppConfig(
        environment=Environment.PRODUCTION,
        debug=False,
        workers=4,
        security=SecurityConfig(
            secret_key="CHANGE_THIS_IN_PRODUCTION",
            session_cookie_secure=True,
            cors_origins=[]  # Must be explicitly configured
        ),
        observability=ObservabilityConfig(
            log_level=LogLevel.INFO,
            log_format=LogFormat.JSON,
            enable_metrics=True,
            enable_tracing=True,
            trace_sample_rate=0.1
        )
    )