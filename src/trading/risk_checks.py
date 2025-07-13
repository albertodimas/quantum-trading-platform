"""
Functional Risk Checks System

Comprehensive risk validation framework with specialized check types,
dynamic thresholds, and real-time monitoring capabilities.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import abc
import statistics
import math

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderSide
from .models import Signal, Position, Order

logger = get_logger(__name__)


class RiskCheckType(Enum):
    """Types of risk checks"""
    PRE_TRADE = "pre_trade"
    POST_TRADE = "post_trade" 
    PORTFOLIO = "portfolio"
    MARKET_CONDITIONS = "market_conditions"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


class RiskSeverity(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CheckResult(Enum):
    """Risk check results"""
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"
    CONDITIONAL = "conditional"


@dataclass
class RiskCheckResult:
    """Risk check execution result"""
    check_name: str
    check_type: RiskCheckType
    result: CheckResult
    severity: RiskSeverity
    score: float  # 0.0 = no risk, 1.0 = maximum risk
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    overridable: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskContext:
    """Context for risk evaluation"""
    account_balance: Decimal
    current_positions: Dict[str, Position]
    portfolio_value: Decimal
    daily_pnl: Decimal
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    compliance_profile: str = "standard"
    override_permissions: List[str] = field(default_factory=list)


class RiskCheckInterface(abc.ABC):
    """Abstract base class for risk checks"""
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Check name"""
        pass
    
    @property
    @abc.abstractmethod
    def check_type(self) -> RiskCheckType:
        """Check type"""
        pass
    
    @property
    @abc.abstractmethod
    def severity(self) -> RiskSeverity:
        """Default severity level"""
        pass
    
    @abc.abstractmethod
    async def evaluate(self, signal: Signal, context: RiskContext) -> RiskCheckResult:
        """Evaluate risk for given signal and context"""
        pass
    
    @abc.abstractmethod
    async def get_threshold_config(self) -> Dict[str, Any]:
        """Get current threshold configuration"""
        pass


class PositionSizeCheck(RiskCheckInterface):
    """Position size validation check"""
    
    def __init__(self, max_position_size: Decimal = Decimal("10000")):
        self.max_position_size = max_position_size
        self.max_portfolio_percentage = Decimal("0.25")  # 25% max per position
    
    @property
    def name(self) -> str:
        return "position_size_check"
    
    @property
    def check_type(self) -> RiskCheckType:
        return RiskCheckType.PRE_TRADE
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.HIGH
    
    async def evaluate(self, signal: Signal, context: RiskContext) -> RiskCheckResult:
        """Evaluate position size constraints"""
        
        if not signal.quantity:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.WARNING,
                severity=RiskSeverity.LOW,
                score=0.1,
                message="No quantity specified in signal",
                recommendations=["Set explicit position size"]
            )
        
        # Check absolute position size
        if signal.quantity > self.max_position_size:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.CRITICAL,
                score=1.0,
                message=f"Position size {signal.quantity} exceeds maximum {self.max_position_size}",
                details={
                    "requested_size": float(signal.quantity),
                    "max_allowed": float(self.max_position_size)
                },
                overridable=False
            )
        
        # Check portfolio percentage
        position_value = signal.quantity * signal.entry_price
        portfolio_percentage = position_value / context.portfolio_value if context.portfolio_value > 0 else Decimal("0")
        
        if portfolio_percentage > self.max_portfolio_percentage:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.HIGH,
                score=float(portfolio_percentage),
                message=f"Position would be {portfolio_percentage:.1%} of portfolio (max {self.max_portfolio_percentage:.1%})",
                details={
                    "position_percentage": float(portfolio_percentage),
                    "max_percentage": float(self.max_portfolio_percentage),
                    "position_value": float(position_value)
                },
                recommendations=[
                    f"Reduce position size to {float(self.max_portfolio_percentage * context.portfolio_value / signal.entry_price):.2f} units"
                ]
            )
        
        # Calculate risk score based on size
        size_ratio = signal.quantity / self.max_position_size
        score = min(float(size_ratio), 0.5)  # Cap at 0.5 for approved positions
        
        return RiskCheckResult(
            check_name=self.name,
            check_type=self.check_type,
            result=CheckResult.APPROVED,
            severity=RiskSeverity.LOW,
            score=score,
            message="Position size within limits",
            details={
                "position_size": float(signal.quantity),
                "portfolio_percentage": float(portfolio_percentage)
            }
        )
    
    async def get_threshold_config(self) -> Dict[str, Any]:
        return {
            "max_position_size": float(self.max_position_size),
            "max_portfolio_percentage": float(self.max_portfolio_percentage)
        }


class LeverageCheck(RiskCheckInterface):
    """Leverage and margin validation check"""
    
    def __init__(self, max_leverage: float = 3.0):
        self.max_leverage = max_leverage
        self.margin_call_threshold = 0.8  # 80% margin utilization
    
    @property
    def name(self) -> str:
        return "leverage_check"
    
    @property
    def check_type(self) -> RiskCheckType:
        return RiskCheckType.PRE_TRADE
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.HIGH
    
    async def evaluate(self, signal: Signal, context: RiskContext) -> RiskCheckResult:
        """Evaluate leverage constraints"""
        
        # Calculate current portfolio leverage
        total_position_value = sum(
            abs(pos.quantity) * pos.current_price 
            for pos in context.current_positions.values()
        )
        
        # Add proposed position value
        new_position_value = signal.quantity * signal.entry_price if signal.quantity else Decimal("0")
        total_exposure = total_position_value + new_position_value
        
        current_leverage = float(total_exposure / context.account_balance) if context.account_balance > 0 else 0
        
        if current_leverage > self.max_leverage:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.CRITICAL,
                score=min(current_leverage / self.max_leverage, 1.0),
                message=f"Leverage {current_leverage:.2f}x exceeds maximum {self.max_leverage}x",
                details={
                    "current_leverage": current_leverage,
                    "max_leverage": self.max_leverage,
                    "total_exposure": float(total_exposure),
                    "account_balance": float(context.account_balance)
                },
                recommendations=[
                    f"Reduce position size to achieve max {self.max_leverage}x leverage",
                    "Close existing positions to free up margin"
                ]
            )
        
        # Check margin utilization
        margin_used = total_exposure / self.max_leverage
        margin_utilization = float(margin_used / context.account_balance) if context.account_balance > 0 else 0
        
        if margin_utilization > self.margin_call_threshold:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.WARNING,
                severity=RiskSeverity.MEDIUM,
                score=margin_utilization,
                message=f"High margin utilization: {margin_utilization:.1%}",
                details={
                    "margin_utilization": margin_utilization,
                    "margin_call_threshold": self.margin_call_threshold
                },
                recommendations=["Monitor positions closely for margin calls"]
            )
        
        return RiskCheckResult(
            check_name=self.name,
            check_type=self.check_type,
            result=CheckResult.APPROVED,
            severity=RiskSeverity.LOW,
            score=current_leverage / self.max_leverage,
            message="Leverage within acceptable limits",
            details={
                "current_leverage": current_leverage,
                "margin_utilization": margin_utilization
            }
        )
    
    async def get_threshold_config(self) -> Dict[str, Any]:
        return {
            "max_leverage": self.max_leverage,
            "margin_call_threshold": self.margin_call_threshold
        }


class ConcentrationCheck(RiskCheckInterface):
    """Portfolio concentration risk check"""
    
    def __init__(self):
        self.max_single_position = Decimal("0.20")  # 20% max per position
        self.max_sector_exposure = Decimal("0.40")   # 40% max per sector
        self.max_correlation_positions = 5  # Max highly correlated positions
    
    @property
    def name(self) -> str:
        return "concentration_check"
    
    @property
    def check_type(self) -> RiskCheckType:
        return RiskCheckType.PORTFOLIO
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.MEDIUM
    
    async def evaluate(self, signal: Signal, context: RiskContext) -> RiskCheckResult:
        """Evaluate concentration risks"""
        
        if not signal.quantity:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.APPROVED,
                severity=RiskSeverity.LOW,
                score=0.0,
                message="No quantity to evaluate"
            )
        
        # Calculate new position value
        new_position_value = signal.quantity * signal.entry_price
        
        # Check single position concentration
        current_symbol_value = Decimal("0")
        if signal.symbol in context.current_positions:
            pos = context.current_positions[signal.symbol]
            current_symbol_value = abs(pos.quantity) * pos.current_price
        
        total_symbol_value = current_symbol_value + new_position_value
        concentration = total_symbol_value / context.portfolio_value if context.portfolio_value > 0 else Decimal("0")
        
        if concentration > self.max_single_position:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.HIGH,
                score=float(concentration),
                message=f"Position concentration {concentration:.1%} exceeds limit {self.max_single_position:.1%}",
                details={
                    "symbol": signal.symbol,
                    "current_concentration": float(concentration),
                    "max_concentration": float(self.max_single_position)
                },
                recommendations=[
                    f"Reduce position size to stay under {self.max_single_position:.1%} limit"
                ]
            )
        
        # Check sector concentration (simplified - based on symbol patterns)
        sector_exposure = await self._calculate_sector_exposure(signal.symbol, context, new_position_value)
        
        if sector_exposure > self.max_sector_exposure:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.WARNING,
                severity=RiskSeverity.MEDIUM,
                score=float(sector_exposure),
                message=f"Sector exposure {sector_exposure:.1%} approaching limit {self.max_sector_exposure:.1%}",
                details={
                    "sector_exposure": float(sector_exposure),
                    "max_sector_exposure": float(self.max_sector_exposure)
                }
            )
        
        return RiskCheckResult(
            check_name=self.name,
            check_type=self.check_type,
            result=CheckResult.APPROVED,
            severity=RiskSeverity.LOW,
            score=max(float(concentration), float(sector_exposure)),
            message="Concentration within acceptable limits",
            details={
                "position_concentration": float(concentration),
                "sector_exposure": float(sector_exposure)
            }
        )
    
    async def _calculate_sector_exposure(self, symbol: str, context: RiskContext, new_value: Decimal) -> Decimal:
        """Calculate sector exposure (simplified implementation)"""
        # Simplified sector classification based on symbol patterns
        sector_map = {
            "BTC": "crypto_major",
            "ETH": "crypto_major", 
            "USD": "fiat",
            "EUR": "fiat",
            "XRP": "crypto_alt",
            "ADA": "crypto_alt"
        }
        
        # Determine symbol sector
        symbol_sector = "unknown"
        for pattern, sector in sector_map.items():
            if pattern in symbol:
                symbol_sector = sector
                break
        
        # Calculate total sector exposure
        sector_value = new_value
        for pos_symbol, position in context.current_positions.items():
            pos_sector = "unknown"
            for pattern, sector in sector_map.items():
                if pattern in pos_symbol:
                    pos_sector = sector
                    break
            
            if pos_sector == symbol_sector:
                sector_value += abs(position.quantity) * position.current_price
        
        return sector_value / context.portfolio_value if context.portfolio_value > 0 else Decimal("0")
    
    async def get_threshold_config(self) -> Dict[str, Any]:
        return {
            "max_single_position": float(self.max_single_position),
            "max_sector_exposure": float(self.max_sector_exposure),
            "max_correlation_positions": self.max_correlation_positions
        }


class VolatilityCheck(RiskCheckInterface):
    """Market volatility risk check"""
    
    def __init__(self):
        self.high_volatility_threshold = 0.05  # 5% daily volatility
        self.extreme_volatility_threshold = 0.10  # 10% daily volatility
        self.volatility_position_scaling = True
    
    @property
    def name(self) -> str:
        return "volatility_check"
    
    @property
    def check_type(self) -> RiskCheckType:
        return RiskCheckType.MARKET_CONDITIONS
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.MEDIUM
    
    async def evaluate(self, signal: Signal, context: RiskContext) -> RiskCheckResult:
        """Evaluate volatility-based risks"""
        
        # Get market volatility from context (would be real market data in production)
        symbol_volatility = context.market_conditions.get(f"{signal.symbol}_volatility", 0.02)
        market_volatility = context.market_conditions.get("market_volatility", 0.015)
        
        max_volatility = max(symbol_volatility, market_volatility)
        
        if max_volatility > self.extreme_volatility_threshold:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.CRITICAL,
                score=min(max_volatility * 10, 1.0),  # Scale to 0-1
                message=f"Extreme market volatility detected: {max_volatility:.1%}",
                details={
                    "symbol_volatility": symbol_volatility,
                    "market_volatility": market_volatility,
                    "extreme_threshold": self.extreme_volatility_threshold
                },
                recommendations=[
                    "Wait for market volatility to decrease",
                    "Use smaller position sizes if trading required"
                ],
                overridable=True
            )
        
        if max_volatility > self.high_volatility_threshold:
            # Calculate volatility-adjusted position size
            volatility_factor = self.high_volatility_threshold / max_volatility
            suggested_size = signal.quantity * Decimal(str(volatility_factor)) if signal.quantity else None
            
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.WARNING,
                severity=RiskSeverity.MEDIUM,
                score=max_volatility * 5,  # Scale to 0-1
                message=f"High volatility detected: {max_volatility:.1%}",
                details={
                    "symbol_volatility": symbol_volatility,
                    "market_volatility": market_volatility,
                    "suggested_position_reduction": float(suggested_size) if suggested_size else None
                },
                recommendations=[
                    f"Consider reducing position size by {(1-volatility_factor):.1%}",
                    "Use tighter stop losses",
                    "Monitor position more closely"
                ]
            )
        
        return RiskCheckResult(
            check_name=self.name,
            check_type=self.check_type,
            result=CheckResult.APPROVED,
            severity=RiskSeverity.LOW,
            score=max_volatility * 2,  # Scale to 0-1
            message="Market volatility within normal ranges",
            details={
                "symbol_volatility": symbol_volatility,
                "market_volatility": market_volatility
            }
        )
    
    async def get_threshold_config(self) -> Dict[str, Any]:
        return {
            "high_volatility_threshold": self.high_volatility_threshold,
            "extreme_volatility_threshold": self.extreme_volatility_threshold,
            "volatility_position_scaling": self.volatility_position_scaling
        }


class LiquidityCheck(RiskCheckInterface):
    """Market liquidity risk check"""
    
    def __init__(self):
        self.min_daily_volume = Decimal("100000")  # Minimum daily volume
        self.max_spread_percentage = 0.005  # 0.5% max bid-ask spread
        self.market_impact_threshold = 0.001  # 0.1% max market impact
    
    @property
    def name(self) -> str:
        return "liquidity_check"
    
    @property
    def check_type(self) -> RiskCheckType:
        return RiskCheckType.MARKET_CONDITIONS
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.MEDIUM
    
    async def evaluate(self, signal: Signal, context: RiskContext) -> RiskCheckResult:
        """Evaluate liquidity risks"""
        
        # Get market data from context (would be real market data in production)
        daily_volume = context.market_conditions.get(f"{signal.symbol}_volume", 1000000)
        bid_price = context.market_conditions.get(f"{signal.symbol}_bid", signal.entry_price * Decimal("0.999"))
        ask_price = context.market_conditions.get(f"{signal.symbol}_ask", signal.entry_price * Decimal("1.001"))
        
        # Check minimum volume
        if daily_volume < float(self.min_daily_volume):
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.HIGH,
                score=1.0 - (daily_volume / float(self.min_daily_volume)),
                message=f"Insufficient liquidity: daily volume ${daily_volume:,.0f}",
                details={
                    "daily_volume": daily_volume,
                    "min_required": float(self.min_daily_volume)
                },
                recommendations=[
                    "Trade only highly liquid instruments",
                    "Use smaller position sizes"
                ]
            )
        
        # Check bid-ask spread
        spread = ask_price - bid_price
        spread_percentage = float(spread / signal.entry_price) if signal.entry_price > 0 else 0
        
        if spread_percentage > self.max_spread_percentage:
            return RiskCheckResult(
                check_name=self.name,
                check_type=self.check_type,
                result=CheckResult.WARNING,
                severity=RiskSeverity.MEDIUM,
                score=spread_percentage / self.max_spread_percentage,
                message=f"Wide spread detected: {spread_percentage:.2%}",
                details={
                    "spread_percentage": spread_percentage,
                    "max_spread": self.max_spread_percentage,
                    "bid_price": float(bid_price),
                    "ask_price": float(ask_price)
                },
                recommendations=[
                    "Use limit orders instead of market orders",
                    "Consider reducing position size"
                ]
            )
        
        # Estimate market impact
        if signal.quantity:
            position_value = signal.quantity * signal.entry_price
            volume_impact = float(position_value) / daily_volume
            
            if volume_impact > self.market_impact_threshold:
                return RiskCheckResult(
                    check_name=self.name,
                    check_type=self.check_type,
                    result=CheckResult.WARNING,
                    severity=RiskSeverity.MEDIUM,
                    score=volume_impact / self.market_impact_threshold,
                    message=f"High market impact estimated: {volume_impact:.3%} of daily volume",
                    details={
                        "market_impact": volume_impact,
                        "threshold": self.market_impact_threshold,
                        "position_value": float(position_value),
                        "daily_volume": daily_volume
                    },
                    recommendations=[
                        "Split order into smaller chunks",
                        "Use TWAP or VWAP execution algorithms"
                    ]
                )
        
        return RiskCheckResult(
            check_name=self.name,
            check_type=self.check_type,
            result=CheckResult.APPROVED,
            severity=RiskSeverity.LOW,
            score=max(spread_percentage / self.max_spread_percentage, 0.1),
            message="Adequate market liquidity",
            details={
                "daily_volume": daily_volume,
                "spread_percentage": spread_percentage
            }
        )
    
    async def get_threshold_config(self) -> Dict[str, Any]:
        return {
            "min_daily_volume": float(self.min_daily_volume),
            "max_spread_percentage": self.max_spread_percentage,
            "market_impact_threshold": self.market_impact_threshold
        }


@injectable
class FunctionalRiskChecker:
    """
    Comprehensive functional risk checking system.
    
    Orchestrates multiple risk check types and provides
    unified risk assessment with detailed reporting.
    """
    
    def __init__(self):
        """Initialize risk checker with default checks"""
        
        # Initialize risk checks
        self.risk_checks: List[RiskCheckInterface] = [
            PositionSizeCheck(),
            LeverageCheck(),
            ConcentrationCheck(),
            VolatilityCheck(),
            LiquidityCheck()
        ]
        
        # Risk check registry
        self.check_registry = {check.name: check for check in self.risk_checks}
        
        # Risk check history
        self.check_history: List[RiskCheckResult] = []
        
        # Override tracking
        self.active_overrides: Dict[str, Dict] = {}
        
        logger.info(
            "Functional risk checker initialized",
            check_count=len(self.risk_checks),
            checks=[check.name for check in self.risk_checks]
        )
    
    async def run_all_checks(
        self,
        signal: Signal,
        context: RiskContext,
        check_types: Optional[List[RiskCheckType]] = None
    ) -> List[RiskCheckResult]:
        """
        Run all applicable risk checks for a signal.
        
        Args:
            signal: Trading signal to evaluate
            context: Risk evaluation context
            check_types: Optional filter for check types
            
        Returns:
            List of risk check results
        """
        
        # Filter checks by type if specified
        applicable_checks = self.risk_checks
        if check_types:
            applicable_checks = [
                check for check in self.risk_checks 
                if check.check_type in check_types
            ]
        
        # Run checks concurrently
        tasks = [
            check.evaluate(signal, context) 
            for check in applicable_checks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Risk check failed: {applicable_checks[i].name}",
                    error=str(result)
                )
            else:
                valid_results.append(result)
                self.check_history.append(result)
        
        # Keep history manageable
        if len(self.check_history) > 1000:
            self.check_history = self.check_history[-1000:]
        
        logger.info(
            "Risk checks completed",
            signal_symbol=signal.symbol,
            checks_run=len(valid_results),
            approved=sum(1 for r in valid_results if r.result == CheckResult.APPROVED),
            rejected=sum(1 for r in valid_results if r.result == CheckResult.REJECTED),
            warnings=sum(1 for r in valid_results if r.result == CheckResult.WARNING)
        )
        
        return valid_results
    
    async def evaluate_overall_risk(
        self,
        signal: Signal,
        context: RiskContext
    ) -> RiskCheckResult:
        """
        Evaluate overall risk by combining all check results.
        
        Returns:
            Aggregate risk assessment
        """
        
        # Run all checks
        results = await self.run_all_checks(signal, context)
        
        if not results:
            return RiskCheckResult(
                check_name="overall_risk",
                check_type=RiskCheckType.PORTFOLIO,
                result=CheckResult.REJECTED,
                severity=RiskSeverity.CRITICAL,
                score=1.0,
                message="No risk checks completed successfully"
            )
        
        # Determine overall result
        critical_failures = [r for r in results if r.result == CheckResult.REJECTED and r.severity == RiskSeverity.CRITICAL]
        rejections = [r for r in results if r.result == CheckResult.REJECTED]
        warnings = [r for r in results if r.result == CheckResult.WARNING]
        
        # Calculate aggregate score
        total_score = sum(r.score for r in results)
        avg_score = total_score / len(results)
        
        # Determine overall result and severity
        if critical_failures:
            overall_result = CheckResult.REJECTED
            overall_severity = RiskSeverity.CRITICAL
            message = f"Critical risk failures: {len(critical_failures)}"
        elif rejections:
            overall_result = CheckResult.REJECTED
            overall_severity = RiskSeverity.HIGH
            message = f"Risk check failures: {len(rejections)}"
        elif warnings:
            overall_result = CheckResult.WARNING
            overall_severity = RiskSeverity.MEDIUM
            message = f"Risk warnings: {len(warnings)}"
        else:
            overall_result = CheckResult.APPROVED
            overall_severity = RiskSeverity.LOW
            message = "All risk checks passed"
        
        # Compile recommendations
        all_recommendations = []
        for result in results:
            if result.result in [CheckResult.REJECTED, CheckResult.WARNING]:
                all_recommendations.extend(result.recommendations)
        
        return RiskCheckResult(
            check_name="overall_risk",
            check_type=RiskCheckType.PORTFOLIO,
            result=overall_result,
            severity=overall_severity,
            score=avg_score,
            message=message,
            details={
                "total_checks": len(results),
                "critical_failures": len(critical_failures),
                "rejections": len(rejections),
                "warnings": len(warnings),
                "check_details": [
                    {
                        "name": r.check_name,
                        "result": r.result.value,
                        "score": r.score,
                        "message": r.message
                    }
                    for r in results
                ]
            },
            recommendations=list(set(all_recommendations))  # Remove duplicates
        )
    
    async def add_check(self, check: RiskCheckInterface) -> None:
        """Add a new risk check"""
        self.risk_checks.append(check)
        self.check_registry[check.name] = check
        
        logger.info(f"Added risk check: {check.name}")
    
    async def remove_check(self, check_name: str) -> bool:
        """Remove a risk check"""
        if check_name in self.check_registry:
            check = self.check_registry[check_name]
            self.risk_checks.remove(check)
            del self.check_registry[check_name]
            
            logger.info(f"Removed risk check: {check_name}")
            return True
        
        return False
    
    async def get_check_status(self) -> Dict[str, Any]:
        """Get current status of all risk checks"""
        status = {}
        
        for check in self.risk_checks:
            status[check.name] = {
                "type": check.check_type.value,
                "severity": check.severity.value,
                "config": await check.get_threshold_config()
            }
        
        return {
            "checks": status,
            "total_checks": len(self.risk_checks),
            "history_size": len(self.check_history),
            "active_overrides": len(self.active_overrides)
        }
    
    async def add_override(
        self,
        check_name: str,
        user_id: str,
        reason: str,
        expiry: Optional[datetime] = None
    ) -> bool:
        """Add risk check override"""
        
        if check_name not in self.check_registry:
            return False
        
        override_id = f"{check_name}_{user_id}_{datetime.now(timezone.utc).timestamp()}"
        
        self.active_overrides[override_id] = {
            "check_name": check_name,
            "user_id": user_id,
            "reason": reason,
            "created_at": datetime.now(timezone.utc),
            "expiry": expiry
        }
        
        logger.warning(
            "Risk check override added",
            check_name=check_name,
            user_id=user_id,
            reason=reason,
            override_id=override_id
        )
        
        return True
    
    async def cleanup_expired_overrides(self) -> int:
        """Remove expired overrides"""
        now = datetime.now(timezone.utc)
        expired_count = 0
        
        expired_keys = []
        for override_id, override in self.active_overrides.items():
            if override.get("expiry") and override["expiry"] < now:
                expired_keys.append(override_id)
                expired_count += 1
        
        for key in expired_keys:
            del self.active_overrides[key]
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired risk overrides")
        
        return expired_count
    
    def get_risk_summary(self, lookback_minutes: int = 60) -> Dict[str, Any]:
        """Get risk summary for recent period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        
        recent_checks = [
            result for result in self.check_history
            if result.timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return {
                "period_minutes": lookback_minutes,
                "total_checks": 0,
                "avg_score": 0.0,
                "rejection_rate": 0.0,
                "warning_rate": 0.0
            }
        
        total_checks = len(recent_checks)
        rejections = sum(1 for r in recent_checks if r.result == CheckResult.REJECTED)
        warnings = sum(1 for r in recent_checks if r.result == CheckResult.WARNING)
        avg_score = sum(r.score for r in recent_checks) / total_checks
        
        return {
            "period_minutes": lookback_minutes,
            "total_checks": total_checks,
            "avg_score": avg_score,
            "rejection_rate": rejections / total_checks,
            "warning_rate": warnings / total_checks,
            "check_breakdown": {
                check_name: {
                    "count": len([r for r in recent_checks if r.check_name == check_name]),
                    "avg_score": statistics.mean([r.score for r in recent_checks if r.check_name == check_name])
                }
                for check_name in set(r.check_name for r in recent_checks)
            }
        }