"""
Risk Management Agent for portfolio and position management.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from src.agents.base import BaseAgent
from src.core.logging import get_logger

logger = get_logger(__name__)


class RiskManagementAgent(BaseAgent):
    """
    Agent specialized in risk assessment and portfolio management.
    
    Monitors positions, calculates risk metrics, and ensures
    trading decisions align with risk parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk management agent."""
        super().__init__("RiskManagementAgent", config)
        
        # Risk parameters
        self.risk_params = {
            "max_position_size": config.get("max_position_size", 0.1),  # 10% per position
            "max_portfolio_risk": config.get("max_portfolio_risk", 0.02),  # 2% total risk
            "max_correlation": config.get("max_correlation", 0.7),  # Max correlation between positions
            "max_leverage": config.get("max_leverage", 1.0),  # No leverage by default
            "max_drawdown": config.get("max_drawdown", 0.2),  # 20% max drawdown
            "risk_reward_ratio": config.get("risk_reward_ratio", 2.0),  # 2:1 minimum
            "max_daily_loss": config.get("max_daily_loss", 0.05),  # 5% daily loss limit
        }
        
        # Risk models
        self.var_confidence = config.get("var_confidence", 0.95)  # 95% VaR
        self.lookback_period = config.get("lookback_period", 30)  # Days for calculations
        
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform risk analysis on portfolio and proposed trades.
        
        Args:
            data: Portfolio data, market data, and proposed trades
            
        Returns:
            Risk analysis results
        """
        if not await self.validate_data(data):
            return {"error": "Invalid data provided"}
        
        try:
            # Extract data
            portfolio = data.get("portfolio", {})
            market_data = data.get("market_data", {})
            proposed_trade = data.get("proposed_trade", None)
            
            # Calculate current portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(portfolio, market_data)
            
            # Assess market risk
            market_risk = await self._assess_market_risk(market_data)
            
            # Evaluate position risks
            position_risks = await self._evaluate_position_risks(portfolio, market_data)
            
            # Check risk limits
            risk_limits = self._check_risk_limits(portfolio_metrics)
            
            # Analyze proposed trade if provided
            trade_analysis = None
            if proposed_trade:
                trade_analysis = await self._analyze_proposed_trade(
                    proposed_trade, portfolio, market_data
                )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                portfolio_metrics, market_risk, position_risks, risk_limits
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                portfolio_metrics, risk_limits, trade_analysis
            )
            
            # Create analysis result
            analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio_metrics": portfolio_metrics,
                "market_risk": market_risk,
                "position_risks": position_risks,
                "risk_limits": risk_limits,
                "trade_analysis": trade_analysis,
                "risk_score": risk_score,
                "recommendations": recommendations,
                "signal_strength": 1.0 - risk_score,  # Inverse of risk
            }
            
            await self._store_analysis(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {"error": str(e)}
    
    async def get_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management signals."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        risk_score = analysis.get("risk_score", 0)
        risk_limits = analysis.get("risk_limits", {})
        recommendations = analysis.get("recommendations", [])
        
        # Generate signals based on risk conditions
        for rec in recommendations:
            if rec["action"] in ["reduce_position", "close_position", "hedge"]:
                signals.append({
                    "type": "risk_management",
                    "action": rec["action"],
                    "symbol": rec.get("symbol"),
                    "reason": rec["reason"],
                    "urgency": rec.get("urgency", "medium"),
                    "confidence": 0.9,  # High confidence in risk signals
                    "risk_score": risk_score,
                })
        
        # Add stop-loss adjustment signals
        position_risks = analysis.get("position_risks", {})
        for symbol, risk_data in position_risks.items():
            if risk_data.get("stop_loss_adjustment"):
                signals.append({
                    "type": "risk_management",
                    "action": "adjust_stop_loss",
                    "symbol": symbol,
                    "new_stop_loss": risk_data["suggested_stop_loss"],
                    "reason": "Risk reduction based on volatility",
                    "confidence": 0.85,
                })
        
        return signals
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for risk analysis."""
        return ["portfolio"]
    
    async def _calculate_portfolio_metrics(self, portfolio: Dict, market_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        positions = portfolio.get("positions", [])
        balance = portfolio.get("balance", 0)
        
        if not positions:
            return {
                "total_value": balance,
                "position_count": 0,
                "exposure": 0,
                "leverage": 0,
                "concentration": 0,
                "var_95": 0,
                "sharpe_ratio": 0,
            }
        
        # Calculate position values and exposure
        total_position_value = 0
        position_values = {}
        returns_data = []
        
        for position in positions:
            symbol = position["symbol"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            
            # Get current price
            current_price = market_data.get(symbol, {}).get("price", entry_price)
            position_value = quantity * current_price
            
            total_position_value += abs(position_value)
            position_values[symbol] = position_value
            
            # Collect returns for risk calculations
            if symbol in market_data and "returns" in market_data[symbol]:
                returns_data.append(market_data[symbol]["returns"])
        
        total_value = balance + total_position_value
        
        # Calculate metrics
        metrics = {
            "total_value": total_value,
            "cash_balance": balance,
            "position_count": len(positions),
            "exposure": total_position_value,
            "leverage": total_position_value / total_value if total_value > 0 else 0,
            "cash_ratio": balance / total_value if total_value > 0 else 1,
        }
        
        # Concentration risk (Herfindahl index)
        if total_position_value > 0:
            concentrations = [(abs(v) / total_position_value) ** 2 
                            for v in position_values.values()]
            metrics["concentration"] = sum(concentrations)
        else:
            metrics["concentration"] = 0
        
        # Value at Risk (VaR)
        if returns_data:
            portfolio_returns = np.mean(returns_data, axis=0)
            metrics["var_95"] = self._calculate_var(portfolio_returns, self.var_confidence)
            metrics["expected_shortfall"] = self._calculate_expected_shortfall(
                portfolio_returns, self.var_confidence
            )
        else:
            metrics["var_95"] = 0
            metrics["expected_shortfall"] = 0
        
        # Performance metrics
        metrics.update(self._calculate_performance_metrics(portfolio, market_data))
        
        return metrics
    
    async def _assess_market_risk(self, market_data: Dict) -> Dict[str, Any]:
        """Assess overall market risk conditions."""
        # Market risk indicators
        risk_indicators = {
            "volatility": {},
            "correlation": {},
            "liquidity": {},
            "trend": {},
        }
        
        # Calculate average volatility
        volatilities = []
        for symbol, data in market_data.items():
            if "volatility" in data:
                volatilities.append(data["volatility"])
                risk_indicators["volatility"][symbol] = data["volatility"]
        
        avg_volatility = np.mean(volatilities) if volatilities else 0
        
        # Market regime
        if avg_volatility > 0.3:
            market_regime = "high_volatility"
            risk_level = "high"
        elif avg_volatility > 0.15:
            market_regime = "normal"
            risk_level = "medium"
        else:
            market_regime = "low_volatility"
            risk_level = "low"
        
        # Correlation analysis
        if len(market_data) > 1:
            correlation_matrix = self._calculate_correlation_matrix(market_data)
            avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(
                correlation_matrix, k=1)]))
            risk_indicators["correlation"]["average"] = avg_correlation
        else:
            avg_correlation = 0
        
        return {
            "market_regime": market_regime,
            "risk_level": risk_level,
            "average_volatility": avg_volatility,
            "average_correlation": avg_correlation,
            "indicators": risk_indicators,
        }
    
    async def _evaluate_position_risks(self, portfolio: Dict, market_data: Dict) -> Dict[str, Any]:
        """Evaluate risk for each position."""
        positions = portfolio.get("positions", [])
        position_risks = {}
        
        for position in positions:
            symbol = position["symbol"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            stop_loss = position.get("stop_loss")
            
            # Get market data
            symbol_data = market_data.get(symbol, {})
            current_price = symbol_data.get("price", entry_price)
            volatility = symbol_data.get("volatility", 0.2)  # Default 20% vol
            
            # Calculate position metrics
            position_value = abs(quantity * current_price)
            pnl = (current_price - entry_price) * quantity
            pnl_percentage = pnl / (entry_price * abs(quantity)) if entry_price > 0 else 0
            
            # Risk metrics
            risk_metrics = {
                "position_value": position_value,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage,
                "volatility": volatility,
                "var_95": position_value * volatility * 1.645,  # Simple VaR
            }
            
            # Stop loss analysis
            if stop_loss:
                stop_distance = abs(current_price - stop_loss) / current_price
                risk_metrics["stop_distance"] = stop_distance
                risk_metrics["stop_loss_risk"] = position_value * stop_distance
                
                # Check if stop loss needs adjustment
                min_stop_distance = volatility * 2  # 2 standard deviations
                if stop_distance < min_stop_distance:
                    risk_metrics["stop_loss_adjustment"] = True
                    risk_metrics["suggested_stop_loss"] = current_price * (
                        1 - min_stop_distance if quantity > 0 else 1 + min_stop_distance
                    )
            
            # Time-based risk (position age)
            position_age = (datetime.utcnow() - datetime.fromisoformat(
                position.get("opened_at", datetime.utcnow().isoformat())
            )).days
            
            if position_age > 30:
                risk_metrics["age_risk"] = "high"
            elif position_age > 7:
                risk_metrics["age_risk"] = "medium"
            else:
                risk_metrics["age_risk"] = "low"
            
            position_risks[symbol] = risk_metrics
        
        return position_risks
    
    def _check_risk_limits(self, portfolio_metrics: Dict) -> Dict[str, Any]:
        """Check if portfolio is within risk limits."""
        violations = []
        warnings = []
        
        # Check leverage
        leverage = portfolio_metrics.get("leverage", 0)
        if leverage > self.risk_params["max_leverage"]:
            violations.append({
                "type": "leverage",
                "current": leverage,
                "limit": self.risk_params["max_leverage"],
                "severity": "high",
            })
        
        # Check concentration
        concentration = portfolio_metrics.get("concentration", 0)
        if concentration > 0.3:  # 30% in single position
            warnings.append({
                "type": "concentration",
                "current": concentration,
                "message": "High position concentration detected",
                "severity": "medium",
            })
        
        # Check VaR
        var_95 = portfolio_metrics.get("var_95", 0)
        total_value = portfolio_metrics.get("total_value", 1)
        var_percentage = var_95 / total_value if total_value > 0 else 0
        
        if var_percentage > self.risk_params["max_portfolio_risk"]:
            warnings.append({
                "type": "var",
                "current": var_percentage,
                "limit": self.risk_params["max_portfolio_risk"],
                "severity": "medium",
            })
        
        # Check drawdown
        drawdown = portfolio_metrics.get("max_drawdown", 0)
        if abs(drawdown) > self.risk_params["max_drawdown"]:
            violations.append({
                "type": "drawdown",
                "current": drawdown,
                "limit": self.risk_params["max_drawdown"],
                "severity": "high",
            })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "risk_capacity_used": var_percentage / self.risk_params["max_portfolio_risk"],
        }
    
    async def _analyze_proposed_trade(self, trade: Dict, portfolio: Dict, 
                                    market_data: Dict) -> Dict[str, Any]:
        """Analyze risk impact of proposed trade."""
        symbol = trade["symbol"]
        side = trade["side"]
        quantity = trade["quantity"]
        price = trade.get("price", market_data.get(symbol, {}).get("price", 0))
        
        # Calculate trade value
        trade_value = abs(quantity * price)
        total_value = portfolio.get("balance", 0) + sum(
            p["quantity"] * market_data.get(p["symbol"], {}).get("price", p["entry_price"])
            for p in portfolio.get("positions", [])
        )
        
        # Position sizing check
        position_size = trade_value / total_value if total_value > 0 else 1
        size_appropriate = position_size <= self.risk_params["max_position_size"]
        
        # Risk/Reward analysis
        stop_loss = trade.get("stop_loss")
        take_profit = trade.get("take_profit")
        
        risk_reward = None
        if stop_loss and take_profit:
            risk = abs(price - stop_loss)
            reward = abs(take_profit - price)
            risk_reward = reward / risk if risk > 0 else 0
        
        risk_reward_acceptable = (risk_reward >= self.risk_params["risk_reward_ratio"] 
                                if risk_reward else False)
        
        # Correlation check
        existing_positions = [p["symbol"] for p in portfolio.get("positions", [])]
        correlation_risk = self._check_correlation_risk(symbol, existing_positions, market_data)
        
        # Kelly criterion for optimal size
        win_rate = 0.55  # Assumed win rate
        avg_win = risk_reward if risk_reward else 2
        avg_loss = 1
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        optimal_size = min(kelly_fraction * 0.25, self.risk_params["max_position_size"])  # 25% Kelly
        
        return {
            "trade_value": trade_value,
            "position_size": position_size,
            "size_appropriate": size_appropriate,
            "risk_reward": risk_reward,
            "risk_reward_acceptable": risk_reward_acceptable,
            "correlation_risk": correlation_risk,
            "optimal_size": optimal_size,
            "recommended_size": min(position_size, optimal_size),
            "approval": size_appropriate and risk_reward_acceptable and not correlation_risk["high"],
        }
    
    def _calculate_risk_score(self, portfolio_metrics: Dict, market_risk: Dict,
                            position_risks: Dict, risk_limits: Dict) -> float:
        """Calculate overall risk score (0-1)."""
        scores = []
        
        # Portfolio risk score
        leverage = portfolio_metrics.get("leverage", 0)
        leverage_score = min(leverage / self.risk_params["max_leverage"], 1.0)
        scores.append(leverage_score * 0.3)
        
        # Market risk score
        market_risk_level = {"low": 0.2, "medium": 0.5, "high": 0.8}
        market_score = market_risk_level.get(market_risk.get("risk_level", "medium"), 0.5)
        scores.append(market_score * 0.2)
        
        # Position risk score
        high_risk_positions = sum(1 for r in position_risks.values() 
                                if r.get("pnl_percentage", 0) < -0.1)
        position_score = min(high_risk_positions / 3, 1.0)  # Normalized by 3
        scores.append(position_score * 0.3)
        
        # Compliance score
        violations = len(risk_limits.get("violations", []))
        compliance_score = min(violations / 2, 1.0)  # Normalized by 2
        scores.append(compliance_score * 0.2)
        
        return sum(scores)
    
    def _generate_recommendations(self, portfolio_metrics: Dict, risk_limits: Dict,
                                trade_analysis: Optional[Dict]) -> List[Dict[str, Any]]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # Check violations
        for violation in risk_limits.get("violations", []):
            if violation["type"] == "leverage":
                recommendations.append({
                    "action": "reduce_position",
                    "reason": f"Leverage ({violation['current']:.2f}) exceeds limit",
                    "urgency": "high",
                })
            elif violation["type"] == "drawdown":
                recommendations.append({
                    "action": "reduce_risk",
                    "reason": f"Drawdown ({violation['current']:.1%}) exceeds limit",
                    "urgency": "high",
                })
        
        # Check warnings
        for warning in risk_limits.get("warnings", []):
            if warning["type"] == "concentration":
                recommendations.append({
                    "action": "diversify",
                    "reason": "High position concentration detected",
                    "urgency": "medium",
                })
            elif warning["type"] == "var":
                recommendations.append({
                    "action": "hedge",
                    "reason": "Portfolio VaR approaching limit",
                    "urgency": "medium",
                })
        
        # Trade recommendations
        if trade_analysis and not trade_analysis.get("approval", False):
            if not trade_analysis.get("size_appropriate", False):
                recommendations.append({
                    "action": "reduce_trade_size",
                    "reason": "Trade size exceeds risk limits",
                    "suggested_size": trade_analysis.get("recommended_size"),
                    "urgency": "high",
                })
            if not trade_analysis.get("risk_reward_acceptable", False):
                recommendations.append({
                    "action": "improve_risk_reward",
                    "reason": f"Risk/reward ratio below minimum ({self.risk_params['risk_reward_ratio']})",
                    "urgency": "medium",
                })
        
        return recommendations
    
    def _calculate_var(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0
        
        # Parametric VaR (assumes normal distribution)
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        var = -(mean + z_score * std)
        return max(var, 0)
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        if len(returns) == 0:
            return 0
        
        # Get VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        
        # Calculate average of returns below VaR
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) > 0:
            return -np.mean(tail_returns)
        else:
            return 0
    
    def _calculate_performance_metrics(self, portfolio: Dict, market_data: Dict) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        # Get historical performance
        history = portfolio.get("performance_history", [])
        
        if len(history) < 2:
            return {
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }
        
        # Calculate returns
        returns = []
        for i in range(1, len(history)):
            prev_value = history[i-1]["value"]
            curr_value = history[i]["value"]
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        if not returns:
            return {
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }
        
        returns = np.array(returns)
        
        # Sharpe ratio (annualized)
        risk_free_rate = 0.02 / 252  # 2% annual, daily
        excess_returns = returns - risk_free_rate
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }
    
    def _calculate_correlation_matrix(self, market_data: Dict) -> np.ndarray:
        """Calculate correlation matrix for assets."""
        symbols = list(market_data.keys())
        n = len(symbols)
        
        if n < 2:
            return np.array([[1.0]])
        
        # Extract returns data
        returns_data = []
        for symbol in symbols:
            if "returns" in market_data[symbol]:
                returns_data.append(market_data[symbol]["returns"])
        
        if len(returns_data) < 2:
            return np.eye(n)
        
        # Calculate correlation
        returns_matrix = np.array(returns_data)
        correlation = np.corrcoef(returns_matrix)
        
        return correlation
    
    def _check_correlation_risk(self, symbol: str, existing_positions: List[str],
                               market_data: Dict) -> Dict[str, Any]:
        """Check correlation risk with existing positions."""
        if not existing_positions or symbol not in market_data:
            return {"high": False, "average_correlation": 0}
        
        correlations = []
        
        for position in existing_positions:
            if position in market_data and "correlation" in market_data[symbol]:
                corr = market_data[symbol]["correlation"].get(position, 0)
                correlations.append(abs(corr))
        
        if correlations:
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            
            return {
                "high": max_correlation > self.risk_params["max_correlation"],
                "average_correlation": avg_correlation,
                "max_correlation": max_correlation,
            }
        
        return {"high": False, "average_correlation": 0}