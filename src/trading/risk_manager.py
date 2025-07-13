"""
Risk Management System

Handles risk assessment, position sizing, and portfolio risk monitoring.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..core.config import settings
from .models import Signal, Position
from .risk_checks import FunctionalRiskChecker, RiskContext, CheckResult
from .var_calculator import VaRCalculator, VaRMethod, VaRHorizon

logger = get_logger(__name__)


@dataclass
class RiskCheck:
    """Risk check result."""
    approved: bool
    reason: Optional[str] = None
    risk_score: float = 0.0
    max_position_size: Optional[Decimal] = None


@dataclass
class RiskStatus:
    """Portfolio risk status."""
    healthy: bool
    reason: Optional[str] = None
    action_required: bool = False
    metrics: Optional[Dict] = None


@injectable
class RiskManager:
    """
    Risk Manager handles all risk assessment and portfolio protection.
    
    Features:
    - Position size calculation
    - Portfolio risk monitoring
    - Stop loss management
    - Drawdown protection
    - Correlation risk
    """
    
    def __init__(self, functional_checker: Optional[FunctionalRiskChecker] = None):
        """Initialize risk manager."""
        # Risk limits (would be configurable in production)
        self.max_position_size = Decimal(str(settings.max_position_size))
        self.max_portfolio_risk = Decimal("0.05")  # 5% of portfolio
        self.max_daily_loss = Decimal("0.02")  # 2% daily loss limit
        self.max_drawdown = Decimal("0.10")  # 10% max drawdown
        
        # Functional risk checker integration
        self.functional_checker = functional_checker or FunctionalRiskChecker()
        
        # VaR calculator integration
        self.var_calculator = VaRCalculator()
        
        # Risk tracking
        self._daily_pnl = Decimal("0")
        self._max_drawdown_today = Decimal("0")
        self._position_correlations: Dict[str, float] = {}
        self._current_positions: Dict[str, Position] = {}
        self._account_balance = Decimal("100000")  # Default balance
        self._portfolio_value = Decimal("100000")   # Default portfolio value
        
        # Risk breach tracking
        self._risk_breaches: List[Dict] = []
        self._last_risk_check = datetime.now(timezone.utc)
        
        logger.info(
            "Risk manager initialized",
            max_position_size=float(self.max_position_size),
            max_portfolio_risk=float(self.max_portfolio_risk),
            max_daily_loss=float(self.max_daily_loss)
        )
    
    async def check_signal(self, signal: Signal) -> RiskCheck:
        """
        Check if trading signal passes risk requirements using functional risk checker.
        
        Args:
            signal: Trading signal to evaluate
            
        Returns:
            Risk check result
        """
        logger.debug(
            "Checking signal risk with functional checker",
            symbol=signal.symbol,
            side=signal.side.value,
            confidence=signal.confidence
        )
        
        # Create risk context
        context = RiskContext(
            account_balance=self._account_balance,
            current_positions=self._current_positions,
            portfolio_value=self._portfolio_value,
            daily_pnl=self._daily_pnl,
            market_conditions={
                f"{signal.symbol}_volatility": 0.02,  # Default 2% volatility
                "market_volatility": 0.015,  # Default 1.5% market volatility
                f"{signal.symbol}_volume": 1000000,  # Default volume
                f"{signal.symbol}_bid": signal.entry_price * Decimal("0.999"),
                f"{signal.symbol}_ask": signal.entry_price * Decimal("1.001")
            }
        )
        
        # Run comprehensive functional risk checks
        overall_result = await self.functional_checker.evaluate_overall_risk(signal, context)
        
        # Convert functional risk result to legacy format for compatibility
        if overall_result.result == CheckResult.APPROVED:
            max_pos_size = await self._calculate_max_position_size(signal)
            return RiskCheck(
                approved=True,
                reason=overall_result.message,
                risk_score=overall_result.score,
                max_position_size=max_pos_size
            )
        elif overall_result.result == CheckResult.WARNING:
            max_pos_size = await self._calculate_max_position_size(signal)
            return RiskCheck(
                approved=True,  # Allow with warnings
                reason=f"WARNING: {overall_result.message}",
                risk_score=overall_result.score,
                max_position_size=max_pos_size
            )
        else:  # REJECTED
            return RiskCheck(
                approved=False,
                reason=overall_result.message,
                risk_score=overall_result.score
            )
    
    async def check_portfolio_risk(self, portfolio_status: Dict) -> RiskStatus:
        """
        Check overall portfolio risk status.
        
        Args:
            portfolio_status: Current portfolio status
            
        Returns:
            Risk status assessment
        """
        total_pnl = Decimal(str(portfolio_status.get("total_pnl", 0)))
        total_value = Decimal(str(portfolio_status.get("total_value", 0)))
        
        # Update daily tracking
        self._update_daily_tracking(total_pnl)
        
        # Check daily loss limit
        if self._daily_pnl < -self.max_daily_loss * total_value:
            return RiskStatus(
                healthy=False,
                reason=f"Daily loss limit exceeded: {self._daily_pnl}",
                action_required=True,
                metrics={
                    "daily_pnl": float(self._daily_pnl),
                    "daily_limit": float(-self.max_daily_loss * total_value)
                }
            )
        
        # Check drawdown
        if self._max_drawdown_today > self.max_drawdown:
            return RiskStatus(
                healthy=False,
                reason=f"Maximum drawdown exceeded: {self._max_drawdown_today:.2%}",
                action_required=True,
                metrics={
                    "current_drawdown": float(self._max_drawdown_today),
                    "max_drawdown": float(self.max_drawdown)
                }
            )
        
        # Check position concentration
        position_count = portfolio_status.get("position_count", 0)
        if position_count > 20:  # Max 20 positions
            return RiskStatus(
                healthy=False,
                reason=f"Too many positions: {position_count}",
                action_required=False,
                metrics={"position_count": position_count}
            )
        
        # Portfolio is healthy
        return RiskStatus(
            healthy=True,
            reason="Portfolio risk within limits",
            action_required=False,
            metrics={
                "daily_pnl": float(self._daily_pnl),
                "current_drawdown": float(self._max_drawdown_today),
                "position_count": position_count
            }
        )
    
    async def calculate_position_size(
        self,
        signal: Signal,
        account_balance: Decimal,
        risk_percentage: Optional[float] = None
    ) -> Decimal:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            signal: Trading signal
            account_balance: Available account balance
            risk_percentage: Risk percentage (uses default if None)
            
        Returns:
            Recommended position size
        """
        risk_pct = risk_percentage or settings.default_risk_percentage
        
        # Use signal's suggested quantity if provided
        if signal.quantity:
            return min(signal.quantity, self.max_position_size)
        
        # Calculate based on stop loss distance
        if signal.stop_loss:
            risk_amount = account_balance * (Decimal(str(risk_pct)) / 100)
            price_diff = abs(signal.entry_price - signal.stop_loss)
            
            if price_diff > 0:
                position_size = risk_amount / price_diff
                return min(position_size, self.max_position_size)
        
        # Default calculation - fixed percentage of balance
        default_size = (account_balance * Decimal(str(risk_pct)) / 100) / signal.entry_price
        return min(default_size, self.max_position_size)
    
    async def update_position_risk(self, positions: Dict[str, Position]) -> None:
        """
        Update risk metrics based on current positions.
        
        Args:
            positions: Current positions
        """
        # Calculate position correlations (simplified)
        symbols = list(positions.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Simplified correlation calculation
                # In production, this would use historical price data
                correlation = self._calculate_correlation(symbol1, symbol2)
                self._position_correlations[f"{symbol1}_{symbol2}"] = correlation
        
        logger.debug(
            "Position risk updated",
            position_count=len(positions),
            correlations=len(self._position_correlations)
        )
    
    def get_limits(self) -> Dict:
        """Get current risk limits."""
        return {
            "max_position_size": float(self.max_position_size),
            "max_portfolio_risk": float(self.max_portfolio_risk),
            "max_daily_loss": float(self.max_daily_loss),
            "max_drawdown": float(self.max_drawdown)
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        return {
            "daily_pnl": float(self._daily_pnl),
            "max_drawdown_today": float(self._max_drawdown_today),
            "risk_breaches_count": len(self._risk_breaches),
            "last_risk_check": self._last_risk_check.isoformat(),
            "position_correlations": len(self._position_correlations)
        }
    
    async def add_risk_override(
        self,
        symbol: str,
        override_type: str,
        value: Decimal,
        expiry: Optional[datetime] = None
    ) -> None:
        """
        Add temporary risk override for specific symbol.
        
        Args:
            symbol: Trading symbol
            override_type: Type of override ('position_size', 'risk_percentage')
            value: Override value
            expiry: Override expiry time
        """
        # Implementation for risk overrides
        # Useful for special market conditions or manual intervention
        logger.info(
            "Risk override added",
            symbol=symbol,
            type=override_type,
            value=float(value),
            expiry=expiry.isoformat() if expiry else None
        )
    
    # Private methods
    
    async def _calculate_max_position_size(self, signal: Signal) -> Decimal:
        """Calculate maximum allowed position size for signal."""
        # Base on entry price and max position value
        max_value = self.max_position_size
        max_quantity = max_value / signal.entry_price
        
        # Apply additional constraints based on signal characteristics
        confidence_factor = Decimal(str(signal.confidence))
        adjusted_max = max_quantity * confidence_factor
        
        return min(adjusted_max, self.max_position_size)
    
    def _update_daily_tracking(self, current_pnl: Decimal) -> None:
        """Update daily P&L and drawdown tracking."""
        # Reset daily tracking if new day
        now = datetime.now(timezone.utc)
        if self._last_risk_check.date() != now.date():
            self._daily_pnl = Decimal("0")
            self._max_drawdown_today = Decimal("0")
        
        # Update metrics
        self._daily_pnl = current_pnl
        
        if current_pnl < 0:
            drawdown_pct = abs(current_pnl) / (abs(current_pnl) + Decimal("100000"))  # Simplified
            self._max_drawdown_today = max(self._max_drawdown_today, drawdown_pct)
        
        self._last_risk_check = now
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Calculate correlation between two symbols.
        
        This is a simplified implementation. In production,
        this would use historical price data and proper statistical methods.
        """
        # Simplified correlation based on symbol similarity
        if symbol1 == symbol2:
            return 1.0
        
        # Check if symbols share common base or quote currency
        if any(curr in symbol1 for curr in ["BTC", "ETH", "USD"]):
            if any(curr in symbol2 for curr in ["BTC", "ETH", "USD"]):
                return 0.3  # Moderate correlation
        
        return 0.1  # Low correlation (default)
    
    async def _record_risk_breach(self, breach_type: str, details: Dict) -> None:
        """Record risk breach for analysis."""
        breach = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": breach_type,
            "details": details
        }
        
        self._risk_breaches.append(breach)
        
        # Keep only last 100 breaches
        if len(self._risk_breaches) > 100:
            self._risk_breaches = self._risk_breaches[-100:]
        
        logger.warning("Risk breach recorded", breach=breach)
    
    async def update_account_balance(self, balance: Decimal) -> None:
        """Update account balance for risk calculations"""
        self._account_balance = balance
        logger.debug(f"Account balance updated: {float(balance)}")
    
    async def update_portfolio_value(self, value: Decimal) -> None:
        """Update portfolio value for risk calculations"""
        self._portfolio_value = value
        logger.debug(f"Portfolio value updated: {float(value)}")
    
    async def update_positions(self, positions: Dict[str, Position]) -> None:
        """Update current positions for risk calculations"""
        self._current_positions = positions
        await self.update_position_risk(positions)
        logger.debug(f"Positions updated: {len(positions)} positions")
    
    async def get_functional_risk_status(self) -> Dict[str, Any]:
        """Get status from functional risk checker"""
        status = await self.functional_checker.get_check_status()
        risk_summary = self.functional_checker.get_risk_summary()
        
        return {
            "functional_checker": status,
            "risk_summary": risk_summary,
            "integration_active": True
        }
    
    async def run_specific_risk_checks(
        self, 
        signal: Signal, 
        check_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Run specific risk checks by name"""
        context = RiskContext(
            account_balance=self._account_balance,
            current_positions=self._current_positions,
            portfolio_value=self._portfolio_value,
            daily_pnl=self._daily_pnl,
            market_conditions={
                f"{signal.symbol}_volatility": 0.02,
                "market_volatility": 0.015,
                f"{signal.symbol}_volume": 1000000,
                f"{signal.symbol}_bid": signal.entry_price * Decimal("0.999"),
                f"{signal.symbol}_ask": signal.entry_price * Decimal("1.001")
            }
        )
        
        # Get specific checks
        specific_checks = [
            check for check in self.functional_checker.risk_checks
            if check.name in check_names
        ]
        
        if not specific_checks:
            return []
        
        # Run checks
        results = []
        for check in specific_checks:
            try:
                result = await check.evaluate(signal, context)
                results.append({
                    "check_name": result.check_name,
                    "result": result.result.value,
                    "severity": result.severity.value,
                    "score": result.score,
                    "message": result.message,
                    "details": result.details,
                    "recommendations": result.recommendations
                })
            except Exception as e:
                logger.error(f"Error running check {check.name}: {str(e)}")
                results.append({
                    "check_name": check.name,
                    "result": "error",
                    "message": f"Check failed: {str(e)}"
                })
        
        return results
    
    async def calculate_portfolio_var(
        self,
        confidence_level: float = 0.95,
        horizon: VaRHorizon = VaRHorizon.DAILY,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> Dict[str, Any]:
        """
        Calculate portfolio Value at Risk.
        
        Args:
            confidence_level: VaR confidence level
            horizon: Time horizon for VaR calculation
            method: VaR calculation method
            
        Returns:
            VaR calculation result
        """
        
        try:
            var_result = await self.var_calculator.calculate_portfolio_var(
                positions=self._current_positions,
                portfolio_value=self._portfolio_value,
                method=method,
                confidence_level=confidence_level,
                horizon=horizon
            )
            
            # Check if VaR exceeds limits
            var_percentage = var_result.var_percentage
            var_limit = float(self.max_portfolio_risk) * 100  # Convert to percentage
            
            risk_status = "acceptable" if var_percentage <= var_limit else "excessive"
            
            result = {
                "var_amount": float(var_result.var_amount),
                "var_percentage": var_percentage,
                "var_limit_percentage": var_limit,
                "risk_status": risk_status,
                "method": var_result.method.value,
                "confidence_level": var_result.confidence_level,
                "horizon": var_result.horizon.value,
                "calculation_timestamp": var_result.calculation_timestamp.isoformat()
            }
            
            # Add expected shortfall if available
            if var_result.expected_shortfall:
                result["expected_shortfall"] = float(var_result.expected_shortfall)
            
            # Add statistical metrics if available
            if var_result.return_mean is not None:
                result["return_statistics"] = {
                    "mean": var_result.return_mean,
                    "std": var_result.return_std,
                    "skewness": var_result.skewness,
                    "kurtosis": var_result.kurtosis
                }
            
            logger.info(
                f"Portfolio VaR calculated: {var_percentage:.2f}%",
                method=method.value,
                confidence_level=confidence_level,
                horizon=horizon.value,
                risk_status=risk_status
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio VaR: {str(e)}")
            return {
                "error": str(e),
                "var_amount": 0.0,
                "var_percentage": 0.0,
                "risk_status": "unknown"
            }
    
    async def generate_comprehensive_var_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive VaR report with multiple methods and metrics.
        
        Returns:
            Comprehensive VaR analysis report
        """
        
        try:
            var_report = await self.var_calculator.calculate_comprehensive_var_report(
                positions=self._current_positions,
                portfolio_value=self._portfolio_value
            )
            
            # Convert to serializable format
            result = {
                "portfolio_value": float(var_report.portfolio_value),
                "calculation_timestamp": var_report.calculation_timestamp.isoformat(),
                "var_results": [],
                "component_vars": [],
                "risk_metrics": var_report.risk_metrics,
                "model_validation": var_report.model_validation
            }
            
            # Process VaR results
            for var_result in var_report.var_results:
                var_data = {
                    "method": var_result.method.value,
                    "confidence_level": var_result.confidence_level,
                    "horizon": var_result.horizon.value,
                    "var_amount": float(var_result.var_amount),
                    "var_percentage": var_result.var_percentage,
                    "metadata": var_result.metadata
                }
                
                if var_result.expected_shortfall:
                    var_data["expected_shortfall"] = float(var_result.expected_shortfall)
                
                result["var_results"].append(var_data)
            
            # Process component VaRs
            for comp_var in var_report.component_vars:
                result["component_vars"].append({
                    "symbol": comp_var.symbol,
                    "component_var": float(comp_var.component_var),
                    "marginal_var": float(comp_var.marginal_var),
                    "contribution_percentage": comp_var.contribution_percentage,
                    "position_value": float(comp_var.position_value)
                })
            
            # Add risk assessment
            max_var_percentage = max(
                (vr["var_percentage"] for vr in result["var_results"]),
                default=0.0
            )
            var_limit = float(self.max_portfolio_risk) * 100
            
            result["risk_assessment"] = {
                "max_var_percentage": max_var_percentage,
                "var_limit_percentage": var_limit,
                "risk_status": "acceptable" if max_var_percentage <= var_limit else "excessive",
                "diversification_ratio": self._calculate_diversification_ratio(var_report.component_vars),
                "concentration_risk": len([cv for cv in var_report.component_vars if cv.contribution_percentage > 20])
            }
            
            logger.info(
                "Comprehensive VaR report generated",
                max_var_percentage=max_var_percentage,
                component_count=len(result["component_vars"]),
                methods_calculated=len(result["var_results"])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate VaR report: {str(e)}")
            return {
                "error": str(e),
                "portfolio_value": float(self._portfolio_value),
                "calculation_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def check_var_limits(self) -> RiskStatus:
        """
        Check if current portfolio VaR exceeds defined limits.
        
        Returns:
            Risk status based on VaR limits
        """
        
        try:
            # Calculate current VaR
            var_result = await self.calculate_portfolio_var(
                confidence_level=0.95,
                horizon=VaRHorizon.DAILY,
                method=VaRMethod.HISTORICAL
            )
            
            var_percentage = var_result.get("var_percentage", 0.0)
            var_limit = float(self.max_portfolio_risk) * 100
            
            if var_percentage > var_limit:
                return RiskStatus(
                    healthy=False,
                    reason=f"Portfolio VaR {var_percentage:.2f}% exceeds limit {var_limit:.2f}%",
                    action_required=True,
                    metrics={
                        "current_var": var_percentage,
                        "var_limit": var_limit,
                        "var_amount": var_result.get("var_amount", 0.0)
                    }
                )
            elif var_percentage > var_limit * 0.8:  # Warning at 80% of limit
                return RiskStatus(
                    healthy=True,
                    reason=f"Portfolio VaR {var_percentage:.2f}% approaching limit {var_limit:.2f}%",
                    action_required=False,
                    metrics={
                        "current_var": var_percentage,
                        "var_limit": var_limit,
                        "warning_threshold": var_limit * 0.8
                    }
                )
            else:
                return RiskStatus(
                    healthy=True,
                    reason=f"Portfolio VaR {var_percentage:.2f}% within limits",
                    action_required=False,
                    metrics={
                        "current_var": var_percentage,
                        "var_limit": var_limit
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to check VaR limits: {str(e)}")
            return RiskStatus(
                healthy=False,
                reason=f"VaR calculation failed: {str(e)}",
                action_required=True
            )
    
    def _calculate_diversification_ratio(self, component_vars) -> float:
        """Calculate portfolio diversification ratio"""
        if not component_vars:
            return 1.0
        
        # Simple diversification measure based on concentration
        contributions = [cv.contribution_percentage for cv in component_vars]
        if not contributions:
            return 1.0
        
        # Herfindahl index (lower = more diversified)
        herfindahl = sum(c**2 for c in contributions) / 10000  # Convert from percentage
        
        # Convert to diversification ratio (higher = more diversified)
        return 1.0 / herfindahl if herfindahl > 0 else 1.0
    
    async def get_var_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of VaR calculation capabilities and configuration"""
        return {
            "var_calculator": self.var_calculator.get_calculation_summary(),
            "integration_status": {
                "functional_checker_active": True,
                "var_calculator_active": True,
                "positions_count": len(self._current_positions),
                "portfolio_value": float(self._portfolio_value),
                "account_balance": float(self._account_balance)
            },
            "risk_limits": {
                "max_portfolio_risk": float(self.max_portfolio_risk),
                "max_daily_loss": float(self.max_daily_loss),
                "max_drawdown": float(self.max_drawdown)
            }
        }