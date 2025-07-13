"""
Trading module for Quantum Trading Platform.

This module contains the core trading engine, order management,
position tracking, and risk management components.
"""

from .engine import TradingEngine
from .order_manager import OrderManager
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .execution_algorithms import ExecutionEngine, ExecutionParameters, ExecutionAlgorithm
from .order_lifecycle import OrderLifecycleManager
from .fifo_calculator import FIFOCalculator, TransactionType, PositionSummary, PnLReport
from .market_impact import MarketImpactCalculator, MarketImpactModel, ImpactParameters, ImpactForecast
from .portfolio_optimizer import PortfolioOptimizer, OptimizationObjective, OptimizationResult, AssetData
from .risk_checks import (
    FunctionalRiskChecker, RiskCheckResult, RiskContext, RiskCheckType, 
    RiskSeverity, CheckResult, PositionSizeCheck, LeverageCheck, 
    ConcentrationCheck, VolatilityCheck, LiquidityCheck
)
from .var_calculator import (
    VaRCalculator, VaRMethod, VaRHorizon, VaRConfiguration, VaRResult,
    ComponentVaRResult, PortfolioVaRReport, VaRDataProcessor
)
from .models import (
    ExecutionReport,
    Order,
    Position,
    Signal,
)

__all__ = [
    "TradingEngine",
    "OrderManager",
    "PositionManager",
    "RiskManager",
    "ExecutionEngine",
    "ExecutionParameters", 
    "ExecutionAlgorithm",
    "OrderLifecycleManager",
    "FIFOCalculator",
    "TransactionType",
    "PositionSummary",
    "PnLReport",
    "MarketImpactCalculator",
    "MarketImpactModel",
    "ImpactParameters",
    "ImpactForecast",
    "PortfolioOptimizer",
    "OptimizationObjective",
    "OptimizationResult",
    "AssetData",
    "FunctionalRiskChecker",
    "RiskCheckResult",
    "RiskContext",
    "RiskCheckType",
    "RiskSeverity",
    "CheckResult",
    "PositionSizeCheck",
    "LeverageCheck",
    "ConcentrationCheck",
    "VolatilityCheck",
    "LiquidityCheck",
    "VaRCalculator",
    "VaRMethod",
    "VaRHorizon",
    "VaRConfiguration",
    "VaRResult",
    "ComponentVaRResult",
    "PortfolioVaRReport",
    "VaRDataProcessor",
    "ExecutionReport",
    "Order",
    "Position",
    "Signal",
]