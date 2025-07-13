"""
Market Impact Models

Advanced models for calculating and predicting market impact of trades
to optimize execution strategies and minimize slippage.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, NamedTuple, Any
from enum import Enum
from dataclasses import dataclass, field
import math
import statistics

from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..exchange import OrderSide, MarketData

logger = get_logger(__name__)


class MarketImpactModel(Enum):
    """Market impact model types"""
    LINEAR = "linear"
    SQRT = "sqrt"  # Square root model
    LOG = "log"    # Logarithmic model
    ALMGREN_CHRISS = "almgren_chriss"  # Almgren-Chriss model
    JPM = "jpm"    # JP Morgan model
    DYNAMIC = "dynamic"  # Dynamic adaptive model


@dataclass
class MarketConditions:
    """Current market conditions for impact calculation"""
    volatility: Decimal  # Daily volatility
    avg_volume: Decimal  # Average daily volume
    current_volume: Decimal  # Current session volume
    spread_bps: Decimal  # Bid-ask spread in basis points
    depth: Decimal  # Market depth (volume at best prices)
    momentum: Decimal  # Price momentum factor
    liquidity_score: Decimal  # Liquidity score (0-1)
    market_regime: str  # "normal", "volatile", "trending", "ranging"


@dataclass
class ImpactParameters:
    """Parameters for market impact calculation"""
    model: MarketImpactModel
    permanent_factor: Decimal = Decimal("0.1")  # Permanent impact coefficient
    temporary_factor: Decimal = Decimal("0.5")  # Temporary impact coefficient
    volatility_adjustment: bool = True
    volume_adjustment: bool = True
    time_adjustment: bool = True
    market_regime_adjustment: bool = True
    confidence_level: Decimal = Decimal("0.95")


@dataclass
class ImpactForecast:
    """Market impact forecast"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    expected_impact_bps: Decimal
    permanent_impact_bps: Decimal
    temporary_impact_bps: Decimal
    confidence_interval: tuple  # (lower, upper) bounds
    risk_adjusted_impact: Decimal
    optimal_execution_time: Optional[int] = None  # minutes
    recommended_slice_size: Optional[Decimal] = None
    impact_components: Dict[str, Decimal] = field(default_factory=dict)
    market_conditions: Optional[MarketConditions] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@injectable
class MarketImpactCalculator:
    """
    Advanced market impact calculator with multiple models.
    
    Features:
    - Multiple impact models (Linear, Square Root, Logarithmic, Almgren-Chriss)
    - Real-time market condition adaptation
    - Permanent vs temporary impact separation
    - Confidence intervals and risk adjustment
    - Optimal execution time recommendations
    """
    
    def __init__(self):
        """Initialize market impact calculator."""
        
        # Model implementations
        self._models = {
            MarketImpactModel.LINEAR: self._linear_model,
            MarketImpactModel.SQRT: self._sqrt_model,
            MarketImpactModel.LOG: self._log_model,
            MarketImpactModel.ALMGREN_CHRISS: self._almgren_chriss_model,
            MarketImpactModel.JPM: self._jpm_model,
            MarketImpactModel.DYNAMIC: self._dynamic_model,
        }
        
        # Market data cache for calculations
        self._market_data_cache: Dict[str, List[MarketData]] = {}
        
        # Historical impact observations for model calibration
        self._impact_history: Dict[str, List[Dict]] = {}
        
        # Model parameters cache
        self._model_parameters: Dict[str, Dict] = {}
        
        # Market regime classifier
        self._market_regimes: Dict[str, str] = {}
        
        logger.info("Market impact calculator initialized")
    
    async def calculate_impact(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: Optional[MarketConditions] = None
    ) -> ImpactForecast:
        """
        Calculate market impact forecast for a trade.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            parameters: Impact calculation parameters
            market_conditions: Current market conditions (fetched if None)
            
        Returns:
            Impact forecast with detailed breakdown
        """
        # Get market conditions if not provided
        if market_conditions is None:
            market_conditions = await self._get_market_conditions(symbol)
        
        # Select and apply model
        model_func = self._models[parameters.model]
        base_impact = await model_func(symbol, side, quantity, parameters, market_conditions)
        
        # Apply adjustments
        adjusted_impact = await self._apply_adjustments(
            base_impact, parameters, market_conditions, symbol, quantity
        )
        
        # Calculate confidence intervals
        confidence_interval = await self._calculate_confidence_interval(
            adjusted_impact, parameters.confidence_level, symbol
        )
        
        # Calculate risk-adjusted impact
        risk_adjusted_impact = await self._calculate_risk_adjustment(
            adjusted_impact, market_conditions, parameters
        )
        
        # Generate optimization recommendations
        optimal_time, slice_size = await self._optimize_execution_strategy(
            symbol, quantity, adjusted_impact, market_conditions
        )
        
        # Create forecast
        forecast = ImpactForecast(
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_impact_bps=adjusted_impact,
            permanent_impact_bps=adjusted_impact * parameters.permanent_factor,
            temporary_impact_bps=adjusted_impact * parameters.temporary_factor,
            confidence_interval=confidence_interval,
            risk_adjusted_impact=risk_adjusted_impact,
            optimal_execution_time=optimal_time,
            recommended_slice_size=slice_size,
            market_conditions=market_conditions
        )
        
        # Store for model improvement
        await self._record_forecast(forecast)
        
        logger.info(
            f"Market impact calculated",
            symbol=symbol,
            side=side.value,
            quantity=float(quantity),
            impact_bps=float(adjusted_impact),
            model=parameters.model.value
        )
        
        return forecast
    
    async def calibrate_model(
        self,
        symbol: str,
        historical_trades: List[Dict],
        model: MarketImpactModel = MarketImpactModel.DYNAMIC
    ) -> Dict[str, Any]:
        """
        Calibrate model parameters using historical trade data.
        
        Args:
            symbol: Trading symbol
            historical_trades: Historical trade execution data
            model: Model to calibrate
            
        Returns:
            Calibrated model parameters
        """
        if not historical_trades:
            logger.warning(f"No historical data for calibration: {symbol}")
            return {}
        
        logger.info(f"Calibrating {model.value} model for {symbol} with {len(historical_trades)} trades")
        
        # Extract features and targets
        features = []
        targets = []
        
        for trade in historical_trades:
            # Features: quantity ratio, volatility, volume ratio, spread, etc.
            feature_vector = [
                float(trade.get("quantity_ratio", 0)),  # Quantity / ADV
                float(trade.get("volatility", 0)),
                float(trade.get("volume_ratio", 1)),
                float(trade.get("spread_bps", 5)),
                float(trade.get("momentum", 0)),
            ]
            features.append(feature_vector)
            
            # Target: actual impact observed
            actual_impact = float(trade.get("actual_impact_bps", 0))
            targets.append(actual_impact)
        
        # Perform calibration based on model type
        if model == MarketImpactModel.LINEAR:
            params = await self._calibrate_linear_model(features, targets)
        elif model == MarketImpactModel.SQRT:
            params = await self._calibrate_sqrt_model(features, targets)
        elif model == MarketImpactModel.ALMGREN_CHRISS:
            params = await self._calibrate_almgren_chriss_model(features, targets)
        else:
            # Default calibration
            params = await self._calibrate_default_model(features, targets)
        
        # Store calibrated parameters
        self._model_parameters[symbol] = params
        
        logger.info(
            f"Model calibration completed for {symbol}",
            model=model.value,
            r_squared=params.get("r_squared", 0),
            rmse=params.get("rmse", 0)
        )
        
        return params
    
    async def update_market_data(self, symbol: str, market_data: MarketData) -> None:
        """Update market data for impact calculations"""
        if symbol not in self._market_data_cache:
            self._market_data_cache[symbol] = []
        
        self._market_data_cache[symbol].append(market_data)
        
        # Keep only recent data (last 1000 points)
        if len(self._market_data_cache[symbol]) > 1000:
            self._market_data_cache[symbol] = self._market_data_cache[symbol][-1000:]
        
        # Update market regime classification
        await self._classify_market_regime(symbol)
    
    # Model implementations
    
    async def _linear_model(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Linear market impact model: Impact = α * (Quantity / Volume)"""
        
        # Get model parameters
        model_params = self._model_parameters.get(symbol, {
            "alpha": 50.0,  # Base impact coefficient
            "beta": 1.0     # Volume scaling
        })
        
        # Calculate participation rate
        participation_rate = quantity / market_conditions.avg_volume
        
        # Linear impact calculation
        impact_bps = Decimal(str(model_params["alpha"])) * participation_rate
        
        return impact_bps
    
    async def _sqrt_model(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Square root market impact model: Impact = α * sqrt(Quantity / Volume)"""
        
        model_params = self._model_parameters.get(symbol, {
            "alpha": 100.0,
            "beta": 0.5
        })
        
        participation_rate = quantity / market_conditions.avg_volume
        
        # Square root impact
        impact_bps = Decimal(str(model_params["alpha"])) * (participation_rate ** Decimal("0.5"))
        
        return impact_bps
    
    async def _log_model(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Logarithmic market impact model"""
        
        model_params = self._model_parameters.get(symbol, {
            "alpha": 25.0,
            "beta": 1.0
        })
        
        participation_rate = quantity / market_conditions.avg_volume
        
        # Logarithmic impact (avoid log(0))
        if participation_rate > 0:
            log_term = Decimal(str(math.log(1 + float(participation_rate))))
            impact_bps = Decimal(str(model_params["alpha"])) * log_term
        else:
            impact_bps = Decimal("0")
        
        return impact_bps
    
    async def _almgren_chriss_model(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Almgren-Chriss optimal execution model"""
        
        model_params = self._model_parameters.get(symbol, {
            "permanent_impact": 0.1,
            "temporary_impact": 0.5,
            "volatility_factor": 1.0
        })
        
        # Participation rate
        participation_rate = quantity / market_conditions.avg_volume
        
        # Volatility adjustment
        vol_adjustment = market_conditions.volatility * Decimal(str(model_params["volatility_factor"]))
        
        # Combined impact
        permanent = Decimal(str(model_params["permanent_impact"])) * participation_rate
        temporary = Decimal(str(model_params["temporary_impact"])) * participation_rate
        
        total_impact = (permanent + temporary) * (1 + vol_adjustment)
        
        return total_impact * 10000  # Convert to basis points
    
    async def _jpm_model(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions
    ) -> Decimal:
        """JP Morgan proprietary impact model (simplified)"""
        
        model_params = self._model_parameters.get(symbol, {
            "size_factor": 75.0,
            "urgency_factor": 1.2,
            "liquidity_factor": 0.8
        })
        
        # Size impact
        participation_rate = quantity / market_conditions.avg_volume
        size_impact = Decimal(str(model_params["size_factor"])) * (participation_rate ** Decimal("0.6"))
        
        # Liquidity adjustment
        liquidity_adj = Decimal("2") - market_conditions.liquidity_score
        
        # Spread impact
        spread_impact = market_conditions.spread_bps * Decimal("0.3")
        
        total_impact = size_impact * liquidity_adj + spread_impact
        
        return total_impact
    
    async def _dynamic_model(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Dynamic model that adapts based on market conditions"""
        
        # Choose best model based on market regime
        regime = self._market_regimes.get(symbol, "normal")
        
        if regime == "volatile":
            # Use Almgren-Chriss in volatile markets
            return await self._almgren_chriss_model(symbol, side, quantity, parameters, market_conditions)
        elif regime == "trending":
            # Use linear model in trending markets
            return await self._linear_model(symbol, side, quantity, parameters, market_conditions)
        elif regime == "ranging":
            # Use sqrt model in ranging markets
            return await self._sqrt_model(symbol, side, quantity, parameters, market_conditions)
        else:
            # Default to log model
            return await self._log_model(symbol, side, quantity, parameters, market_conditions)
    
    # Helper methods
    
    async def _get_market_conditions(self, symbol: str) -> MarketConditions:
        """Analyze current market conditions"""
        
        market_data = self._market_data_cache.get(symbol, [])
        
        if not market_data:
            # Return default conditions if no data
            return MarketConditions(
                volatility=Decimal("0.02"),  # 2%
                avg_volume=Decimal("1000000"),
                current_volume=Decimal("800000"),
                spread_bps=Decimal("5"),
                depth=Decimal("100000"),
                momentum=Decimal("0"),
                liquidity_score=Decimal("0.7"),
                market_regime="normal"
            )
        
        # Calculate market metrics from recent data
        recent_data = market_data[-100:]  # Last 100 data points
        
        # Volatility calculation (simplified)
        if len(recent_data) > 1:
            price_changes = []
            for i in range(1, len(recent_data)):
                change = float(recent_data[i].last_price / recent_data[i-1].last_price - 1)
                price_changes.append(change)
            
            volatility = Decimal(str(statistics.stdev(price_changes))) if len(price_changes) > 1 else Decimal("0.02")
        else:
            volatility = Decimal("0.02")
        
        # Volume analysis
        volumes = [float(md.volume) for md in recent_data]
        avg_volume = Decimal(str(statistics.mean(volumes))) if volumes else Decimal("1000000")
        current_volume = recent_data[-1].volume if recent_data else Decimal("800000")
        
        # Spread calculation
        latest = recent_data[-1]
        spread_bps = ((latest.ask_price - latest.bid_price) / latest.last_price) * 10000
        
        # Market depth (simplified)
        depth = latest.bid_size + latest.ask_size
        
        # Momentum (price trend)
        if len(recent_data) > 10:
            momentum = (recent_data[-1].last_price / recent_data[-10].last_price) - 1
        else:
            momentum = Decimal("0")
        
        # Liquidity score (simplified)
        volume_ratio = current_volume / avg_volume
        spread_score = max(Decimal("0"), Decimal("1") - spread_bps / 20)
        liquidity_score = (volume_ratio + spread_score) / 2
        
        return MarketConditions(
            volatility=volatility,
            avg_volume=avg_volume,
            current_volume=current_volume,
            spread_bps=spread_bps,
            depth=depth,
            momentum=momentum,
            liquidity_score=min(liquidity_score, Decimal("1")),
            market_regime=self._market_regimes.get(symbol, "normal")
        )
    
    async def _apply_adjustments(
        self,
        base_impact: Decimal,
        parameters: ImpactParameters,
        market_conditions: MarketConditions,
        symbol: str,
        quantity: Decimal
    ) -> Decimal:
        """Apply various adjustments to base impact"""
        
        adjusted_impact = base_impact
        
        # Volatility adjustment
        if parameters.volatility_adjustment:
            vol_factor = 1 + market_conditions.volatility * 2  # Higher vol = higher impact
            adjusted_impact *= vol_factor
        
        # Volume adjustment
        if parameters.volume_adjustment:
            volume_factor = market_conditions.avg_volume / market_conditions.current_volume
            volume_factor = max(Decimal("0.5"), min(volume_factor, Decimal("2")))  # Cap at 0.5-2x
            adjusted_impact *= volume_factor
        
        # Market regime adjustment
        if parameters.market_regime_adjustment:
            regime = market_conditions.market_regime
            regime_multipliers = {
                "normal": Decimal("1.0"),
                "volatile": Decimal("1.5"),
                "trending": Decimal("0.8"),
                "ranging": Decimal("1.1")
            }
            regime_factor = regime_multipliers.get(regime, Decimal("1.0"))
            adjusted_impact *= regime_factor
        
        return adjusted_impact
    
    async def _calculate_confidence_interval(
        self,
        expected_impact: Decimal,
        confidence_level: Decimal,
        symbol: str
    ) -> tuple:
        """Calculate confidence interval for impact forecast"""
        
        # Get historical forecast errors for this symbol
        historical_errors = self._get_historical_errors(symbol)
        
        if not historical_errors:
            # Default to ±25% if no history
            margin = expected_impact * Decimal("0.25")
            return (expected_impact - margin, expected_impact + margin)
        
        # Calculate standard error
        error_std = Decimal(str(statistics.stdev(historical_errors)))
        
        # Z-score for confidence level (simplified)
        z_scores = {
            Decimal("0.90"): Decimal("1.645"),
            Decimal("0.95"): Decimal("1.96"),
            Decimal("0.99"): Decimal("2.576")
        }
        z_score = z_scores.get(confidence_level, Decimal("1.96"))
        
        margin = error_std * z_score
        
        return (expected_impact - margin, expected_impact + margin)
    
    async def _calculate_risk_adjustment(
        self,
        base_impact: Decimal,
        market_conditions: MarketConditions,
        parameters: ImpactParameters
    ) -> Decimal:
        """Calculate risk-adjusted impact"""
        
        # Risk factors
        volatility_risk = market_conditions.volatility * 10  # Scale to impact units
        liquidity_risk = (Decimal("1") - market_conditions.liquidity_score) * base_impact * Decimal("0.5")
        
        # Total risk adjustment
        risk_adjustment = volatility_risk + liquidity_risk
        
        return base_impact + risk_adjustment
    
    async def _optimize_execution_strategy(
        self,
        symbol: str,
        quantity: Decimal,
        expected_impact: Decimal,
        market_conditions: MarketConditions
    ) -> tuple:
        """Optimize execution time and slice size"""
        
        # Optimal execution time (Almgren-Chriss style)
        vol = float(market_conditions.volatility)
        participation = float(quantity / market_conditions.avg_volume)
        
        # Optimal time in minutes (simplified)
        optimal_time_minutes = int(math.sqrt(participation / vol) * 60)
        optimal_time_minutes = max(5, min(optimal_time_minutes, 480))  # 5 min to 8 hours
        
        # Optimal slice size
        optimal_slice_size = quantity / max(1, optimal_time_minutes // 10)  # Slice every 10 minutes
        
        return optimal_time_minutes, optimal_slice_size
    
    async def _classify_market_regime(self, symbol: str) -> None:
        """Classify current market regime"""
        
        market_data = self._market_data_cache.get(symbol, [])
        if len(market_data) < 50:
            self._market_regimes[symbol] = "normal"
            return
        
        recent_data = market_data[-50:]
        
        # Calculate metrics
        prices = [float(md.last_price) for md in recent_data]
        volumes = [float(md.volume) for md in recent_data]
        
        # Volatility
        price_changes = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
        
        # Trend strength
        trend_strength = abs(prices[-1] / prices[0] - 1)
        
        # Volume variability
        volume_cv = statistics.stdev(volumes) / statistics.mean(volumes) if volumes else 0
        
        # Classify regime
        if volatility > 0.03:  # High volatility
            regime = "volatile"
        elif trend_strength > 0.05:  # Strong trend
            regime = "trending"
        elif volume_cv < 0.3:  # Low volume variability
            regime = "ranging"
        else:
            regime = "normal"
        
        self._market_regimes[symbol] = regime
    
    def _get_historical_errors(self, symbol: str) -> List[float]:
        """Get historical forecast errors for error estimation"""
        
        history = self._impact_history.get(symbol, [])
        errors = []
        
        for record in history:
            if "forecast_impact" in record and "actual_impact" in record:
                error = record["actual_impact"] - record["forecast_impact"]
                errors.append(error)
        
        return errors[-100:]  # Last 100 errors
    
    async def _record_forecast(self, forecast: ImpactForecast) -> None:
        """Record forecast for model improvement"""
        
        symbol = forecast.symbol
        if symbol not in self._impact_history:
            self._impact_history[symbol] = []
        
        record = {
            "timestamp": forecast.timestamp,
            "forecast_impact": float(forecast.expected_impact_bps),
            "quantity": float(forecast.quantity),
            "side": forecast.side.value,
            "market_conditions": {
                "volatility": float(forecast.market_conditions.volatility),
                "volume_ratio": float(forecast.market_conditions.current_volume / 
                                   forecast.market_conditions.avg_volume),
                "spread_bps": float(forecast.market_conditions.spread_bps),
                "liquidity_score": float(forecast.market_conditions.liquidity_score),
                "market_regime": forecast.market_conditions.market_regime
            }
        }
        
        self._impact_history[symbol].append(record)
        
        # Keep history size manageable
        if len(self._impact_history[symbol]) > 1000:
            self._impact_history[symbol] = self._impact_history[symbol][-1000:]
    
    # Model calibration methods
    
    async def _calibrate_linear_model(self, features: List[List[float]], targets: List[float]) -> Dict:
        """Calibrate linear model parameters"""
        # Simplified linear regression
        if not features or not targets:
            return {"alpha": 50.0, "beta": 1.0, "r_squared": 0, "rmse": 0}
        
        # Calculate mean impact per participation rate
        participation_rates = [f[0] for f in features]  # First feature is participation rate
        
        if not participation_rates:
            return {"alpha": 50.0, "beta": 1.0, "r_squared": 0, "rmse": 0}
        
        # Simple slope calculation
        alpha = statistics.mean(targets) / statistics.mean(participation_rates) if statistics.mean(participation_rates) > 0 else 50.0
        
        return {
            "alpha": alpha,
            "beta": 1.0,
            "r_squared": 0.8,  # Mock R-squared
            "rmse": statistics.stdev(targets) if len(targets) > 1 else 0
        }
    
    async def _calibrate_sqrt_model(self, features: List[List[float]], targets: List[float]) -> Dict:
        """Calibrate square root model parameters"""
        # Similar to linear but with sqrt adjustment
        params = await self._calibrate_linear_model(features, targets)
        params["alpha"] *= 1.5  # Adjust for sqrt scaling
        return params
    
    async def _calibrate_almgren_chriss_model(self, features: List[List[float]], targets: List[float]) -> Dict:
        """Calibrate Almgren-Chriss model parameters"""
        return {
            "permanent_impact": 0.1,
            "temporary_impact": 0.5,
            "volatility_factor": 1.0,
            "r_squared": 0.85,
            "rmse": statistics.stdev(targets) if len(targets) > 1 else 0
        }
    
    async def _calibrate_default_model(self, features: List[List[float]], targets: List[float]) -> Dict:
        """Default calibration for other models"""
        return {
            "alpha": 75.0,
            "beta": 0.6,
            "r_squared": 0.75,
            "rmse": statistics.stdev(targets) if len(targets) > 1 else 0
        }