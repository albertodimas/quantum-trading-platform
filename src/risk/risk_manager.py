"""
Risk Management Engine for the Quantum Trading Platform.

Features:
- Real-time risk monitoring and enforcement
- Position limits and exposure controls
- Margin and leverage management
- Stop-loss and risk limits
- Portfolio-level risk metrics
- Risk alerts and notifications
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import numpy as np
from collections import defaultdict

from ..core.observability.logger import get_logger
from ..core.observability.metrics import get_metrics_collector
from ..core.observability.tracing import trace_async
from ..core.messaging.event_bus import get_event_bus, Event, EventPriority
from ..core.architecture.circuit_breaker import CircuitBreaker
from ..core.cache.cache_manager import CacheManager
from ..positions.position_tracker import PositionTracker, Position, PositionSide
from ..positions.portfolio_manager import PortfolioManager, PortfolioMetrics
from ..orders.order_manager import OrderRequest, OrderTracking


logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """Types of risk violations."""
    POSITION_LIMIT = "position_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    LOSS_LIMIT = "loss_limit"
    CONCENTRATION = "concentration"
    MARGIN_CALL = "margin_call"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Position limits
    max_position_size: Dict[str, Decimal] = field(default_factory=dict)  # Per symbol
    max_position_value: Decimal = Decimal("100000")  # Per position
    max_total_positions: int = 50
    
    # Exposure limits
    max_gross_exposure: Decimal = Decimal("1000000")
    max_net_exposure: Decimal = Decimal("500000")
    max_sector_exposure: Dict[str, Decimal] = field(default_factory=dict)
    
    # Leverage limits
    max_leverage: Decimal = Decimal("3.0")
    max_margin_usage: Decimal = Decimal("0.8")  # 80% of available margin
    
    # Loss limits
    max_daily_loss: Decimal = Decimal("10000")
    max_weekly_loss: Decimal = Decimal("25000")
    max_monthly_loss: Decimal = Decimal("50000")
    max_drawdown: Decimal = Decimal("0.15")  # 15%
    
    # Concentration limits
    max_concentration_single: Decimal = Decimal("0.2")  # 20% in single position
    max_concentration_sector: Decimal = Decimal("0.4")  # 40% in single sector
    max_correlation_exposure: Decimal = Decimal("0.7")  # 70% correlated positions
    
    # Volatility limits
    max_portfolio_volatility: Decimal = Decimal("0.25")  # 25% annualized
    max_position_volatility: Decimal = Decimal("0.5")    # 50% annualized
    
    # Trading limits
    max_orders_per_minute: int = 100
    max_orders_per_symbol: int = 20
    max_order_value: Decimal = Decimal("50000")


@dataclass
class RiskViolation:
    """Risk violation details."""
    violation_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    symbol: Optional[str]
    current_value: Decimal
    limit_value: Decimal
    description: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    # Exposure metrics
    gross_exposure: Decimal
    net_exposure: Decimal
    long_exposure: Decimal
    short_exposure: Decimal
    
    # P&L metrics
    daily_pnl: Decimal
    weekly_pnl: Decimal
    monthly_pnl: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    
    # Risk metrics
    portfolio_var: Decimal  # Value at Risk
    portfolio_cvar: Decimal  # Conditional VaR
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    
    # Leverage and margin
    current_leverage: Decimal
    margin_used: Decimal
    margin_available: Decimal
    
    # Concentration
    largest_position_pct: Decimal
    top_5_concentration: Decimal
    sector_concentrations: Dict[str, Decimal]
    
    # Volatility
    portfolio_volatility: Decimal
    position_volatilities: Dict[str, Decimal]
    
    # Correlations
    correlation_exposure: Decimal
    correlation_matrix: Optional[np.ndarray] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """
    Risk management engine for monitoring and enforcing risk limits.
    
    Features:
    - Real-time risk monitoring
    - Pre-trade and post-trade risk checks
    - Dynamic risk limit adjustment
    - Risk alerts and notifications
    - Historical risk tracking
    """
    
    def __init__(self, position_tracker: PositionTracker,
                 portfolio_manager: PortfolioManager,
                 risk_limits: RiskLimits,
                 cache_manager: Optional[CacheManager] = None):
        self.position_tracker = position_tracker
        self.portfolio_manager = portfolio_manager
        self.risk_limits = risk_limits
        self._logger = get_logger(self.__class__.__name__)
        self._metrics = get_metrics_collector().get_collector("risk")
        self._event_bus = get_event_bus()
        self._cache = cache_manager
        
        # Circuit breakers for critical operations
        self._order_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        # Risk state
        self._active_violations: Dict[str, RiskViolation] = {}
        self._risk_metrics_history: List[RiskMetrics] = []
        self._max_history = 1000
        
        # Trading state
        self._order_counts: Dict[str, List[datetime]] = defaultdict(list)
        self._daily_pnl_cache: Dict[str, Decimal] = {}
        
        # Configuration
        self._margin_call_threshold = Decimal("0.9")  # 90% margin usage
        self._force_liquidation_threshold = Decimal("0.95")  # 95% margin usage
        
        # Background tasks
        self._monitoring_task = asyncio.create_task(self._monitor_risk())
        self._cleanup_task = asyncio.create_task(self._cleanup_old_data())
    
    @trace_async(name="pre_trade_risk_check", tags={"component": "risk_manager"})
    async def check_pre_trade_risk(self, order_request: OrderRequest) -> Tuple[bool, List[str]]:
        """
        Perform pre-trade risk checks.
        
        Args:
            order_request: Order to check
            
        Returns:
            Tuple of (is_allowed, list_of_reasons)
        """
        violations = []
        
        # Check order limits
        order_check = await self._check_order_limits(order_request)
        if order_check:
            violations.extend(order_check)
        
        # Check position limits
        position_check = await self._check_position_limits(order_request)
        if position_check:
            violations.extend(position_check)
        
        # Check exposure limits
        exposure_check = await self._check_exposure_limits(order_request)
        if exposure_check:
            violations.extend(exposure_check)
        
        # Check leverage limits
        leverage_check = await self._check_leverage_limits(order_request)
        if leverage_check:
            violations.extend(leverage_check)
        
        # Check loss limits
        loss_check = await self._check_loss_limits()
        if loss_check:
            violations.extend(loss_check)
        
        # Record metrics
        if self._metrics:
            self._metrics.record_metric("risk.pre_trade_checks", 1, tags={
                "symbol": order_request.symbol,
                "side": order_request.side.value,
                "passed": len(violations) == 0
            })
        
        # Log violations
        if violations:
            self._logger.warning("Pre-trade risk check failed",
                               order_symbol=order_request.symbol,
                               violations=violations)
        
        return len(violations) == 0, violations
    
    async def _check_order_limits(self, order_request: OrderRequest) -> List[str]:
        """Check order-specific limits."""
        violations = []
        
        # Check order value
        order_value = order_request.quantity * (order_request.price or Decimal("0"))
        if order_value > self.risk_limits.max_order_value:
            violations.append(
                f"Order value {order_value} exceeds limit {self.risk_limits.max_order_value}"
            )
        
        # Check order rate limits
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old timestamps
        self._order_counts["_total"] = [
            ts for ts in self._order_counts["_total"] if ts > minute_ago
        ]
        self._order_counts[order_request.symbol] = [
            ts for ts in self._order_counts[order_request.symbol] if ts > minute_ago
        ]
        
        # Check total order rate
        if len(self._order_counts["_total"]) >= self.risk_limits.max_orders_per_minute:
            violations.append(
                f"Order rate limit exceeded: {self.risk_limits.max_orders_per_minute}/min"
            )
        
        # Check per-symbol order rate
        if len(self._order_counts[order_request.symbol]) >= self.risk_limits.max_orders_per_symbol:
            violations.append(
                f"Symbol order rate limit exceeded: {self.risk_limits.max_orders_per_symbol}/min"
            )
        
        return violations
    
    async def _check_position_limits(self, order_request: OrderRequest) -> List[str]:
        """Check position-specific limits."""
        violations = []
        
        # Get current position
        current_position = await self.position_tracker.get_position(order_request.symbol)
        current_quantity = current_position.quantity if current_position else Decimal("0")
        
        # Calculate new position
        if order_request.side.value == "buy":
            new_quantity = current_quantity + order_request.quantity
        else:
            new_quantity = current_quantity - order_request.quantity
        
        # Check position size limit
        if order_request.symbol in self.risk_limits.max_position_size:
            limit = self.risk_limits.max_position_size[order_request.symbol]
            if abs(new_quantity) > limit:
                violations.append(
                    f"Position size {abs(new_quantity)} exceeds limit {limit} for {order_request.symbol}"
                )
        
        # Check position value limit
        price = order_request.price or current_position.current_price if current_position else Decimal("0")
        if price > 0:
            position_value = abs(new_quantity) * price
            if position_value > self.risk_limits.max_position_value:
                violations.append(
                    f"Position value {position_value} exceeds limit {self.risk_limits.max_position_value}"
                )
        
        # Check total positions limit
        all_positions = await self.position_tracker.get_all_positions()
        active_positions = sum(1 for p in all_positions.values() if p.quantity != 0)
        
        if current_quantity == 0 and new_quantity != 0:  # Opening new position
            if active_positions >= self.risk_limits.max_total_positions:
                violations.append(
                    f"Total positions {active_positions} at limit {self.risk_limits.max_total_positions}"
                )
        
        return violations
    
    async def _check_exposure_limits(self, order_request: OrderRequest) -> List[str]:
        """Check exposure limits."""
        violations = []
        
        # Calculate current metrics
        metrics = await self.calculate_risk_metrics()
        
        # Estimate new exposure after order
        price = order_request.price or Decimal("0")
        order_value = order_request.quantity * price
        
        if order_request.side.value == "buy":
            new_gross = metrics.gross_exposure + order_value
            new_net = metrics.net_exposure + order_value
        else:
            new_gross = metrics.gross_exposure + order_value
            new_net = metrics.net_exposure - order_value
        
        # Check gross exposure
        if new_gross > self.risk_limits.max_gross_exposure:
            violations.append(
                f"Gross exposure {new_gross} would exceed limit {self.risk_limits.max_gross_exposure}"
            )
        
        # Check net exposure
        if abs(new_net) > self.risk_limits.max_net_exposure:
            violations.append(
                f"Net exposure {abs(new_net)} would exceed limit {self.risk_limits.max_net_exposure}"
            )
        
        return violations
    
    async def _check_leverage_limits(self, order_request: OrderRequest) -> List[str]:
        """Check leverage and margin limits."""
        violations = []
        
        # Get current metrics
        metrics = await self.calculate_risk_metrics()
        
        # Check current leverage
        if metrics.current_leverage > self.risk_limits.max_leverage:
            violations.append(
                f"Current leverage {metrics.current_leverage} exceeds limit {self.risk_limits.max_leverage}"
            )
        
        # Check margin usage
        margin_usage = metrics.margin_used / (metrics.margin_used + metrics.margin_available) if metrics.margin_available > 0 else Decimal("1")
        
        if margin_usage > self.risk_limits.max_margin_usage:
            violations.append(
                f"Margin usage {margin_usage:.2%} exceeds limit {self.risk_limits.max_margin_usage:.2%}"
            )
        
        return violations
    
    async def _check_loss_limits(self) -> List[str]:
        """Check loss limits."""
        violations = []
        
        # Get P&L metrics
        metrics = await self.calculate_risk_metrics()
        
        # Check daily loss
        if metrics.daily_pnl < -self.risk_limits.max_daily_loss:
            violations.append(
                f"Daily loss {abs(metrics.daily_pnl)} exceeds limit {self.risk_limits.max_daily_loss}"
            )
        
        # Check weekly loss
        if metrics.weekly_pnl < -self.risk_limits.max_weekly_loss:
            violations.append(
                f"Weekly loss {abs(metrics.weekly_pnl)} exceeds limit {self.risk_limits.max_weekly_loss}"
            )
        
        # Check monthly loss
        if metrics.monthly_pnl < -self.risk_limits.max_monthly_loss:
            violations.append(
                f"Monthly loss {abs(metrics.monthly_pnl)} exceeds limit {self.risk_limits.max_monthly_loss}"
            )
        
        # Check drawdown
        if metrics.current_drawdown > self.risk_limits.max_drawdown:
            violations.append(
                f"Drawdown {metrics.current_drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown:.2%}"
            )
        
        return violations
    
    @trace_async(name="calculate_risk_metrics", tags={"component": "risk_manager"})
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics."""
        # Get all positions
        positions = await self.position_tracker.get_all_positions()
        
        # Calculate exposure metrics
        gross_exposure = Decimal("0")
        net_exposure = Decimal("0")
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        
        for position in positions.values():
            if position.quantity > 0:
                long_exposure += position.market_value
                net_exposure += position.market_value
            else:
                short_exposure += position.market_value
                net_exposure -= position.market_value
            
            gross_exposure += position.market_value
        
        # Get portfolio metrics
        try:
            portfolio_metrics = await self.portfolio_manager.calculate_portfolio_metrics()
            portfolio_var = portfolio_metrics.var_95
            portfolio_cvar = portfolio_metrics.cvar_95
            sharpe_ratio = portfolio_metrics.sharpe_ratio
            sortino_ratio = portfolio_metrics.sortino_ratio
            portfolio_volatility = portfolio_metrics.volatility
        except:
            portfolio_var = Decimal("0")
            portfolio_cvar = Decimal("0")
            sharpe_ratio = Decimal("0")
            sortino_ratio = Decimal("0")
            portfolio_volatility = Decimal("0")
        
        # Calculate P&L metrics
        daily_pnl = await self._calculate_period_pnl(timedelta(days=1))
        weekly_pnl = await self._calculate_period_pnl(timedelta(days=7))
        monthly_pnl = await self._calculate_period_pnl(timedelta(days=30))
        
        # Calculate drawdown
        max_drawdown, current_drawdown = await self._calculate_drawdown()
        
        # Calculate leverage (simplified)
        account_value = gross_exposure  # Placeholder
        current_leverage = gross_exposure / account_value if account_value > 0 else Decimal("0")
        
        # Margin calculations (simplified)
        margin_used = gross_exposure * Decimal("0.1")  # 10% margin requirement
        margin_available = account_value - margin_used
        
        # Concentration metrics
        largest_position_pct = Decimal("0")
        if gross_exposure > 0:
            largest_position = max(positions.values(), key=lambda p: p.market_value, default=None)
            if largest_position:
                largest_position_pct = largest_position.market_value / gross_exposure
        
        # Top 5 concentration
        sorted_positions = sorted(positions.values(), key=lambda p: p.market_value, reverse=True)
        top_5_value = sum(p.market_value for p in sorted_positions[:5])
        top_5_concentration = top_5_value / gross_exposure if gross_exposure > 0 else Decimal("0")
        
        # Position volatilities (placeholder)
        position_volatilities = {
            symbol: Decimal("0.2")  # 20% placeholder
            for symbol in positions.keys()
        }
        
        metrics = RiskMetrics(
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            current_leverage=current_leverage,
            margin_used=margin_used,
            margin_available=margin_available,
            largest_position_pct=largest_position_pct,
            top_5_concentration=top_5_concentration,
            sector_concentrations={},
            portfolio_volatility=portfolio_volatility,
            position_volatilities=position_volatilities,
            correlation_exposure=Decimal("0")  # Placeholder
        )
        
        # Cache metrics
        if self._cache:
            await self._cache.set_async("risk:current_metrics", metrics.__dict__, ttl=60)
        
        return metrics
    
    async def _calculate_period_pnl(self, period: timedelta) -> Decimal:
        """Calculate P&L for a specific period."""
        # Get historical snapshots
        start_time = datetime.utcnow() - period
        snapshots = await self.position_tracker.get_snapshots(start_time=start_time)
        
        if not snapshots:
            return Decimal("0")
        
        # Calculate P&L from snapshots
        start_value = snapshots[0].total_market_value
        current_positions = await self.position_tracker.get_all_positions()
        current_value = sum(p.market_value for p in current_positions.values())
        
        # Add realized P&L
        total_realized = sum(p.realized_pnl for p in current_positions.values())
        
        return current_value - start_value + total_realized
    
    async def _calculate_drawdown(self) -> Tuple[Decimal, Decimal]:
        """Calculate maximum and current drawdown."""
        # Get historical snapshots
        snapshots = await self.position_tracker.get_snapshots()
        
        if len(snapshots) < 2:
            return Decimal("0"), Decimal("0")
        
        # Calculate drawdown series
        values = [float(s.total_market_value) for s in snapshots]
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        
        max_drawdown = abs(min(drawdowns))
        current_drawdown = abs(drawdowns[-1])
        
        return Decimal(str(max_drawdown)), Decimal(str(current_drawdown))
    
    async def check_risk_violations(self) -> List[RiskViolation]:
        """Check for any risk violations."""
        violations = []
        metrics = await self.calculate_risk_metrics()
        
        # Check exposure limits
        if metrics.gross_exposure > self.risk_limits.max_gross_exposure:
            violations.append(self._create_violation(
                RiskType.EXPOSURE_LIMIT,
                RiskLevel.HIGH,
                current=metrics.gross_exposure,
                limit=self.risk_limits.max_gross_exposure,
                description="Gross exposure limit exceeded"
            ))
        
        # Check leverage limits
        if metrics.current_leverage > self.risk_limits.max_leverage:
            violations.append(self._create_violation(
                RiskType.LEVERAGE_LIMIT,
                RiskLevel.HIGH,
                current=metrics.current_leverage,
                limit=self.risk_limits.max_leverage,
                description="Leverage limit exceeded"
            ))
        
        # Check loss limits
        if metrics.daily_pnl < -self.risk_limits.max_daily_loss:
            violations.append(self._create_violation(
                RiskType.LOSS_LIMIT,
                RiskLevel.CRITICAL,
                current=abs(metrics.daily_pnl),
                limit=self.risk_limits.max_daily_loss,
                description="Daily loss limit exceeded"
            ))
        
        # Check drawdown
        if metrics.current_drawdown > self.risk_limits.max_drawdown:
            violations.append(self._create_violation(
                RiskType.LOSS_LIMIT,
                RiskLevel.HIGH,
                current=metrics.current_drawdown,
                limit=self.risk_limits.max_drawdown,
                description="Maximum drawdown exceeded"
            ))
        
        # Check concentration
        if metrics.largest_position_pct > self.risk_limits.max_concentration_single:
            violations.append(self._create_violation(
                RiskType.CONCENTRATION,
                RiskLevel.MEDIUM,
                current=metrics.largest_position_pct,
                limit=self.risk_limits.max_concentration_single,
                description="Single position concentration limit exceeded"
            ))
        
        # Check margin call
        margin_usage = metrics.margin_used / (metrics.margin_used + metrics.margin_available) if metrics.margin_available > 0 else Decimal("1")
        if margin_usage > self._margin_call_threshold:
            violations.append(self._create_violation(
                RiskType.MARGIN_CALL,
                RiskLevel.CRITICAL,
                current=margin_usage,
                limit=self._margin_call_threshold,
                description="Margin call threshold exceeded"
            ))
        
        # Check volatility
        if metrics.portfolio_volatility > self.risk_limits.max_portfolio_volatility:
            violations.append(self._create_violation(
                RiskType.VOLATILITY,
                RiskLevel.MEDIUM,
                current=metrics.portfolio_volatility,
                limit=self.risk_limits.max_portfolio_volatility,
                description="Portfolio volatility limit exceeded"
            ))
        
        return violations
    
    def _create_violation(self, risk_type: RiskType, risk_level: RiskLevel,
                         current: Decimal, limit: Decimal, description: str,
                         symbol: Optional[str] = None) -> RiskViolation:
        """Create a risk violation."""
        import uuid
        
        return RiskViolation(
            violation_id=str(uuid.uuid4()),
            risk_type=risk_type,
            risk_level=risk_level,
            symbol=symbol,
            current_value=current,
            limit_value=limit,
            description=description,
            timestamp=datetime.utcnow()
        )
    
    async def _monitor_risk(self):
        """Background task to monitor risk continuously."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check for violations
                violations = await self.check_risk_violations()
                
                # Process new violations
                for violation in violations:
                    if violation.violation_id not in self._active_violations:
                        self._active_violations[violation.violation_id] = violation
                        
                        # Publish risk event
                        await self._publish_risk_event(violation)
                        
                        # Take action based on severity
                        if violation.risk_level == RiskLevel.CRITICAL:
                            await self._handle_critical_risk(violation)
                
                # Check resolved violations
                current_violation_types = {(v.risk_type, v.symbol) for v in violations}
                
                for vid, violation in list(self._active_violations.items()):
                    if (violation.risk_type, violation.symbol) not in current_violation_types:
                        violation.resolved = True
                        violation.resolution_time = datetime.utcnow()
                        
                        # Publish resolution event
                        await self._publish_risk_resolution_event(violation)
                        
                        # Remove from active
                        del self._active_violations[vid]
                
                # Update metrics history
                metrics = await self.calculate_risk_metrics()
                self._risk_metrics_history.append(metrics)
                
                if len(self._risk_metrics_history) > self._max_history:
                    self._risk_metrics_history = self._risk_metrics_history[-self._max_history:]
                
                # Record monitoring metrics
                if self._metrics:
                    self._metrics.record_metric("risk.violations.active", len(self._active_violations))
                    self._metrics.record_metric("risk.leverage.current", float(metrics.current_leverage))
                    self._metrics.record_metric("risk.exposure.gross", float(metrics.gross_exposure))
                
            except Exception as e:
                self._logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _handle_critical_risk(self, violation: RiskViolation):
        """Handle critical risk violations."""
        self._logger.critical("Critical risk violation detected",
                            violation_type=violation.risk_type.value,
                            description=violation.description,
                            current=str(violation.current_value),
                            limit=str(violation.limit_value))
        
        # Different handling based on violation type
        if violation.risk_type == RiskType.MARGIN_CALL:
            # Initiate position reduction
            await self._initiate_margin_call_liquidation()
        
        elif violation.risk_type == RiskType.LOSS_LIMIT:
            # Halt new trading
            await self._halt_trading()
    
    async def _initiate_margin_call_liquidation(self):
        """Initiate margin call liquidation process."""
        # Get positions sorted by P&L (worst first)
        positions = await self.position_tracker.get_all_positions()
        sorted_positions = sorted(
            positions.values(),
            key=lambda p: p.unrealized_pnl + p.realized_pnl
        )
        
        # Start liquidating positions
        for position in sorted_positions:
            if position.quantity != 0:
                self._logger.warning("Initiating margin call liquidation",
                                   symbol=position.symbol,
                                   quantity=str(position.quantity))
                
                # Publish liquidation event
                event = Event(
                    type="risk.margin_call.liquidation",
                    data={
                        "symbol": position.symbol,
                        "quantity": str(position.quantity),
                        "reason": "margin_call"
                    },
                    source="risk_manager",
                    priority=EventPriority.CRITICAL
                )
                await self._event_bus.publish(event)
                
                # Check if margin restored
                metrics = await self.calculate_risk_metrics()
                margin_usage = metrics.margin_used / (metrics.margin_used + metrics.margin_available)
                
                if margin_usage < self._margin_call_threshold * Decimal("0.9"):  # 10% buffer
                    break
    
    async def _halt_trading(self):
        """Halt all new trading."""
        self._logger.critical("Halting all trading due to risk limits")
        
        # Publish halt event
        event = Event(
            type="risk.trading.halt",
            data={"reason": "risk_limit_exceeded"},
            source="risk_manager",
            priority=EventPriority.CRITICAL
        )
        await self._event_bus.publish(event)
    
    async def _publish_risk_event(self, violation: RiskViolation):
        """Publish risk violation event."""
        event = Event(
            type=f"risk.violation.{violation.risk_type.value}",
            data={
                "violation_id": violation.violation_id,
                "risk_level": violation.risk_level.value,
                "symbol": violation.symbol,
                "current_value": str(violation.current_value),
                "limit_value": str(violation.limit_value),
                "description": violation.description
            },
            source="risk_manager",
            priority=EventPriority.HIGH if violation.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else EventPriority.NORMAL
        )
        await self._event_bus.publish(event)
    
    async def _publish_risk_resolution_event(self, violation: RiskViolation):
        """Publish risk violation resolution event."""
        event = Event(
            type="risk.violation.resolved",
            data={
                "violation_id": violation.violation_id,
                "risk_type": violation.risk_type.value,
                "duration": str(violation.resolution_time - violation.timestamp)
            },
            source="risk_manager",
            priority=EventPriority.NORMAL
        )
        await self._event_bus.publish(event)
    
    async def _cleanup_old_data(self):
        """Clean up old data periodically."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Clean old order counts
                cutoff = datetime.utcnow() - timedelta(hours=1)
                for symbol in list(self._order_counts.keys()):
                    self._order_counts[symbol] = [
                        ts for ts in self._order_counts[symbol] if ts > cutoff
                    ]
                    if not self._order_counts[symbol]:
                        del self._order_counts[symbol]
                
                # Clean old P&L cache
                self._daily_pnl_cache.clear()
                
                self._logger.info("Risk data cleanup completed")
                
            except Exception as e:
                self._logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary."""
        return {
            "active_violations": len(self._active_violations),
            "violations_by_type": self._get_violations_by_type(),
            "violations_by_level": self._get_violations_by_level(),
            "critical_violations": [
                v for v in self._active_violations.values()
                if v.risk_level == RiskLevel.CRITICAL
            ]
        }
    
    def _get_violations_by_type(self) -> Dict[str, int]:
        """Get violation count by type."""
        counts = defaultdict(int)
        for violation in self._active_violations.values():
            counts[violation.risk_type.value] += 1
        return dict(counts)
    
    def _get_violations_by_level(self) -> Dict[str, int]:
        """Get violation count by level."""
        counts = defaultdict(int)
        for violation in self._active_violations.values():
            counts[violation.risk_level.value] += 1
        return dict(counts)
    
    async def get_risk_history(self, hours: int = 24) -> List[RiskMetrics]:
        """Get risk metrics history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self._risk_metrics_history if m.timestamp > cutoff]
    
    async def update_risk_limits(self, new_limits: RiskLimits):
        """Update risk limits dynamically."""
        old_limits = self.risk_limits
        self.risk_limits = new_limits
        
        self._logger.info("Risk limits updated",
                         changes=self._compare_limits(old_limits, new_limits))
        
        # Re-check violations with new limits
        await self.check_risk_violations()
    
    def _compare_limits(self, old: RiskLimits, new: RiskLimits) -> Dict[str, Any]:
        """Compare risk limits for changes."""
        changes = {}
        
        # Compare each field
        for field in ['max_gross_exposure', 'max_net_exposure', 'max_leverage',
                     'max_daily_loss', 'max_drawdown']:
            old_val = getattr(old, field)
            new_val = getattr(new, field)
            if old_val != new_val:
                changes[field] = {"old": str(old_val), "new": str(new_val)}
        
        return changes
    
    async def shutdown(self):
        """Shutdown the risk manager."""
        # Cancel monitoring tasks
        self._monitoring_task.cancel()
        self._cleanup_task.cancel()
        
        await asyncio.gather(
            self._monitoring_task,
            self._cleanup_task,
            return_exceptions=True
        )
        
        self._logger.info("Risk manager shutdown complete")