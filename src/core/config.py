"""
Configuration management for Quantum Trading Platform.

Uses Pydantic Settings for type-safe configuration with
environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    app_name: str = "Quantum Trading Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    
    # API
    api_prefix: str = "/api/v1"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Security
    secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Database
    database_url: PostgresDsn
    database_pool_size: int = 20
    database_max_overflow: int = 40
    database_pool_timeout: int = 30
    
    # Redis
    redis_url: RedisDsn
    redis_pool_size: int = 10
    redis_decode_responses: bool = True
    
    # Celery
    celery_broker_url: str
    celery_result_backend: str
    celery_task_always_eager: bool = False
    
    # Exchange APIs
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    binance_testnet: bool = True
    
    coinbase_api_key: Optional[str] = None
    coinbase_api_secret: Optional[str] = None
    
    # AI/ML
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.7
    
    # Trading
    default_quote_currency: str = "USDT"
    max_position_size: float = 10000.0  # Maximum position size in quote currency
    default_risk_percentage: float = 2.0  # Risk per trade as % of portfolio
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = True
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_file: Optional[Path] = None
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_exchange_config(self, exchange: str) -> Dict[str, Any]:
        """Get configuration for a specific exchange."""
        configs = {
            "binance": {
                "apiKey": self.binance_api_key,
                "secret": self.binance_api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "testnet": self.binance_testnet,
                }
            },
            "coinbase": {
                "apiKey": self.coinbase_api_key,
                "secret": self.coinbase_api_secret,
                "enableRateLimit": True,
            }
        }
        return configs.get(exchange, {})


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()