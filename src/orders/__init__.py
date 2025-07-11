"""
Order Management System components.

This module provides comprehensive order management functionality including:
- Order lifecycle management
- Execution algorithms (TWAP, VWAP, Iceberg)
- Slippage and market impact modeling
- Smart order routing
"""

from .order_manager import (
    OrderManager,
    OrderRequest,
    OrderTracking,
    OrderExecution,
    OrderValidationError,
    InsufficientBalanceError,
    RiskLimitExceededError
)

from .execution_algorithms import (
    AlgorithmType,
    AlgorithmConfig,
    TWAPConfig,
    VWAPConfig,
    IcebergConfig,
    AlgorithmExecution,
    ExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    IcebergAlgorithm,
    AlgorithmFactory
)

from .slippage_model import (
    ImpactModel,
    MarketConditions,
    SlippageEstimate,
    SlippageModel,
    LinearImpactModel,
    SquareRootImpactModel,
    AlmgrenChrissModel,
    AdaptiveSlippageModel
)

__all__ = [
    # Order Manager
    "OrderManager",
    "OrderRequest",
    "OrderTracking",
    "OrderExecution",
    "OrderValidationError",
    "InsufficientBalanceError",
    "RiskLimitExceededError",
    
    # Execution Algorithms
    "AlgorithmType",
    "AlgorithmConfig",
    "TWAPConfig",
    "VWAPConfig",
    "IcebergConfig",
    "AlgorithmExecution",
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "IcebergAlgorithm",
    "AlgorithmFactory",
    
    # Slippage Models
    "ImpactModel",
    "MarketConditions",
    "SlippageEstimate",
    "SlippageModel",
    "LinearImpactModel",
    "SquareRootImpactModel",
    "AlmgrenChrissModel",
    "AdaptiveSlippageModel"
]