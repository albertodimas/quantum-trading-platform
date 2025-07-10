"""
Mean Reversion trading strategy implementation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.strategies.base import BaseStrategy
from src.core.logging import get_logger
from src.trading.models import Signal, OrderSide

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Identifies and trades assets that have deviated significantly from
    their mean, expecting them to revert to normal levels.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mean reversion strategy."""
        super().__init__("MeanReversionStrategy", config)
    
    def _initialize_parameters(self):
        """Initialize mean reversion-specific parameters."""
        # Bollinger Bands parameters
        self.bb_period = self.config.get("bb_period", 20)
        self.bb_std_dev = self.config.get("bb_std_dev", 2.0)
        
        # Z-score parameters
        self.zscore_period = self.config.get("zscore_period", 20)
        self.zscore_threshold = self.config.get("zscore_threshold", 2.0)
        
        # Mean reversion parameters
        self.lookback_period = self.config.get("lookback_period", 50)
        self.min_deviation = self.config.get("min_deviation", 0.02)  # 2% minimum deviation
        self.reversion_speed = self.config.get("reversion_speed", 0.5)
        
        # RSI for oversold/overbought
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_oversold = self.config.get("rsi_oversold", 20)
        self.rsi_overbought = self.config.get("rsi_overbought", 80)
        
        # Volume confirmation
        self.volume_ma_period = self.config.get("volume_ma_period", 20)
        self.min_volume_ratio = self.config.get("min_volume_ratio", 0.8)
        
        # Regime filter
        self.use_regime_filter = self.config.get("use_regime_filter", True)
        self.min_hurst_exponent = self.config.get("min_hurst_exponent", 0.4)  # Mean reverting if < 0.5
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for mean reversion opportunities.
        
        Args:
            market_data: OHLCV and other market data
            
        Returns:
            Mean reversion analysis results
        """
        try:
            # Prepare data
            df = self._prepare_dataframe(market_data)
            
            if len(df) < self.lookback_period:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate mean reversion indicators
            reversion_indicators = self._calculate_reversion_indicators(df)
            
            # Analyze price deviations
            deviation_analysis = self._analyze_deviations(df, reversion_indicators)
            
            # Check market regime
            regime_analysis = self._analyze_market_regime(df)
            
            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(df)
            
            # Calculate reversion probability
            reversion_probability = self._calculate_reversion_probability(
                deviation_analysis, regime_analysis, volume_analysis
            )
            
            # Identify entry points
            entry_signals = self._identify_entry_points(
                reversion_indicators, deviation_analysis, reversion_probability
            )
            
            # Calculate target levels
            target_levels = self._calculate_target_levels(df, reversion_indicators)
            
            analysis = {
                "symbol": market_data.get("symbol"),
                "timestamp": datetime.utcnow().isoformat(),
                "reversion_indicators": reversion_indicators,
                "deviation_analysis": deviation_analysis,
                "regime_analysis": regime_analysis,
                "volume_analysis": volume_analysis,
                "reversion_probability": reversion_probability,
                "entry_signals": entry_signals,
                "target_levels": target_levels,
                "confidence": self._calculate_confidence(
                    reversion_probability, regime_analysis, volume_analysis
                ),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Mean reversion analysis error: {e}")
            return {"error": str(e)}
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals from mean reversion analysis."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        confidence = analysis.get("confidence", 0)
        if confidence < self.min_confidence:
            return signals
        
        entry_signals = analysis.get("entry_signals", {})
        target_levels = analysis.get("target_levels", {})
        
        # Long signal (oversold, expecting bounce)
        if entry_signals.get("long_signal") and entry_signals.get("long_strength", 0) > 0.6:
            current_price = target_levels.get("current_price", 0)
            
            signal = Signal(
                symbol=analysis["symbol"],
                side=OrderSide.BUY,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, OrderSide.BUY),
                take_profit=target_levels.get("mean_target", current_price * 1.02),
                strategy=self.name,
                metadata={
                    "signal_type": "mean_reversion_long",
                    "deviation": analysis["deviation_analysis"].get("current_deviation"),
                    "z_score": analysis["reversion_indicators"].get("z_score", {}).get("value"),
                    "reversion_probability": analysis["reversion_probability"],
                    "target_levels": target_levels,
                }
            )
            
            if self.validate_signal(signal):
                signals.append(signal)
        
        # Short signal (overbought, expecting pullback)
        elif entry_signals.get("short_signal") and entry_signals.get("short_strength", 0) > 0.6:
            current_price = target_levels.get("current_price", 0)
            
            signal = Signal(
                symbol=analysis["symbol"],
                side=OrderSide.SELL,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, OrderSide.SELL),
                take_profit=target_levels.get("mean_target", current_price * 0.98),
                strategy=self.name,
                metadata={
                    "signal_type": "mean_reversion_short",
                    "deviation": analysis["deviation_analysis"].get("current_deviation"),
                    "z_score": analysis["reversion_indicators"].get("z_score", {}).get("value"),
                    "reversion_probability": analysis["reversion_probability"],
                    "target_levels": target_levels,
                }
            )
            
            if self.validate_signal(signal):
                signals.append(signal)
        
        return signals
    
    def _prepare_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare DataFrame from market data."""
        ohlcv = market_data.get("ohlcv", [])
        
        df = pd.DataFrame(ohlcv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df.sort_index()
    
    def _calculate_reversion_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mean reversion indicators."""
        indicators = {}
        
        # Bollinger Bands
        indicators["bollinger_bands"] = self._calculate_bollinger_bands(df)
        
        # Z-Score
        indicators["z_score"] = self._calculate_z_score(df)
        
        # Mean and standard deviation
        indicators["rolling_stats"] = self._calculate_rolling_statistics(df)
        
        # Relative position
        indicators["relative_position"] = self._calculate_relative_position(df)
        
        # Mean reversion speed
        indicators["reversion_speed"] = self._estimate_reversion_speed(df)
        
        # RSI for extreme conditions
        indicators["rsi"] = self._calculate_rsi(df)
        
        return indicators
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        close = df["close"]
        
        # Moving average
        ma = close.rolling(window=self.bb_period).mean()
        
        # Standard deviation
        std = close.rolling(window=self.bb_period).std()
        
        # Bands
        upper_band = ma + (std * self.bb_std_dev)
        lower_band = ma - (std * self.bb_std_dev)
        
        current_price = close.iloc[-1]
        current_ma = ma.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Band width
        band_width = (current_upper - current_lower) / current_ma if current_ma > 0 else 0
        
        # Position within bands (0 = lower band, 1 = upper band)
        if current_upper != current_lower:
            band_position = (current_price - current_lower) / (current_upper - current_lower)
        else:
            band_position = 0.5
        
        return {
            "ma": current_ma,
            "upper": current_upper,
            "lower": current_lower,
            "band_width": band_width,
            "band_position": band_position,
            "price_to_upper": (current_price - current_upper) / current_upper if current_upper > 0 else 0,
            "price_to_lower": (current_price - current_lower) / current_lower if current_lower > 0 else 0,
            "squeeze": band_width < 0.02,  # Bollinger squeeze
        }
    
    def _calculate_z_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price Z-score."""
        close = df["close"]
        
        # Rolling mean and std
        mean = close.rolling(window=self.zscore_period).mean()
        std = close.rolling(window=self.zscore_period).std()
        
        # Z-score
        current_price = close.iloc[-1]
        current_mean = mean.iloc[-1]
        current_std = std.iloc[-1]
        
        if current_std > 0:
            z_score = (current_price - current_mean) / current_std
        else:
            z_score = 0
        
        # Z-score momentum (change in z-score)
        z_scores = (close - mean) / std
        z_score_change = z_scores.diff(5).iloc[-1] if len(z_scores) > 5 else 0
        
        return {
            "value": z_score,
            "change": z_score_change,
            "extreme": abs(z_score) > self.zscore_threshold,
            "oversold": z_score < -self.zscore_threshold,
            "overbought": z_score > self.zscore_threshold,
            "reverting": (z_score > self.zscore_threshold and z_score_change < 0) or 
                       (z_score < -self.zscore_threshold and z_score_change > 0),
        }
    
    def _calculate_rolling_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate rolling statistics."""
        close = df["close"]
        
        stats = {}
        
        # Multiple timeframes
        for period in [20, 50, 100]:
            if len(close) >= period:
                mean = close.rolling(window=period).mean().iloc[-1]
                std = close.rolling(window=period).std().iloc[-1]
                
                stats[f"mean_{period}"] = mean
                stats[f"std_{period}"] = std
                stats[f"deviation_{period}"] = (close.iloc[-1] - mean) / mean if mean > 0 else 0
        
        # Historical percentile
        if len(close) >= self.lookback_period:
            percentile = (close.iloc[-1] > close.iloc[-self.lookback_period:]).sum() / self.lookback_period
            stats["percentile"] = percentile
        
        return stats
    
    def _calculate_relative_position(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate relative position indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # Price range position
        period_high = high.rolling(window=self.lookback_period).max().iloc[-1]
        period_low = low.rolling(window=self.lookback_period).min().iloc[-1]
        
        if period_high != period_low:
            range_position = (close.iloc[-1] - period_low) / (period_high - period_low)
        else:
            range_position = 0.5
        
        # Distance from mean
        mean = close.rolling(window=self.lookback_period).mean().iloc[-1]
        distance_from_mean = (close.iloc[-1] - mean) / mean if mean > 0 else 0
        
        return {
            "range_position": range_position,
            "distance_from_mean": distance_from_mean,
            "near_high": range_position > 0.8,
            "near_low": range_position < 0.2,
            "extreme_distance": abs(distance_from_mean) > self.min_deviation,
        }
    
    def _estimate_reversion_speed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate mean reversion speed using Ornstein-Uhlenbeck process."""
        close = df["close"]
        
        if len(close) < 20:
            return {"theta": 0, "half_life": np.inf, "fast_reversion": False}
        
        # Log prices
        log_prices = np.log(close)
        
        # Estimate theta (mean reversion speed)
        # Using AR(1) approximation: p(t) = alpha + beta * p(t-1) + epsilon
        # theta = -log(beta)
        
        y = log_prices.diff().dropna()
        x = log_prices.shift(1).dropna()[1:]
        
        if len(x) > 10 and len(y) > 10:
            # Simple linear regression
            x_mean = x.mean()
            y_mean = y.mean()
            
            beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            
            if beta < 1 and beta > -1:
                theta = -np.log(abs(beta))
                half_life = np.log(2) / theta if theta > 0 else np.inf
            else:
                theta = 0
                half_life = np.inf
        else:
            theta = 0
            half_life = np.inf
        
        return {
            "theta": theta,
            "half_life": half_life,
            "fast_reversion": half_life < 10,  # Fast if reverts in less than 10 periods
            "mean_reverting": theta > 0.01,
        }
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI for extreme conditions."""
        close = df["close"]
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        return {
            "value": current_rsi,
            "oversold": current_rsi < self.rsi_oversold,
            "overbought": current_rsi > self.rsi_overbought,
            "extreme": current_rsi < self.rsi_oversold or current_rsi > self.rsi_overbought,
        }
    
    def _analyze_deviations(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price deviations from mean."""
        close = df["close"]
        
        # Current deviation metrics
        bb = indicators.get("bollinger_bands", {})
        z_score = indicators.get("z_score", {})
        rel_pos = indicators.get("relative_position", {})
        
        # Deviation magnitude
        current_deviation = abs(rel_pos.get("distance_from_mean", 0))
        
        # Deviation persistence (how long price has been away from mean)
        mean = close.rolling(window=self.bb_period).mean()
        above_mean = close > mean
        
        # Count consecutive periods away from mean
        deviation_streak = 0
        for i in range(len(above_mean) - 1, -1, -1):
            if (above_mean.iloc[-1] and above_mean.iloc[i]) or \
               (not above_mean.iloc[-1] and not above_mean.iloc[i]):
                deviation_streak += 1
            else:
                break
        
        # Historical deviation analysis
        deviations = []
        if len(close) >= self.lookback_period:
            for i in range(self.lookback_period):
                idx = -self.lookback_period + i
                if idx < -len(close):
                    continue
                period_mean = close.iloc[max(idx-20, -len(close)):idx].mean()
                deviation = (close.iloc[idx] - period_mean) / period_mean if period_mean > 0 else 0
                deviations.append(abs(deviation))
        
        avg_deviation = np.mean(deviations) if deviations else 0
        
        return {
            "current_deviation": current_deviation,
            "deviation_streak": deviation_streak,
            "significant": current_deviation > self.min_deviation,
            "extreme": current_deviation > avg_deviation * 2,
            "z_score_extreme": z_score.get("extreme", False),
            "bb_extreme": bb.get("band_position", 0.5) < 0.1 or bb.get("band_position", 0.5) > 0.9,
            "direction": "above" if close.iloc[-1] > mean.iloc[-1] else "below",
        }
    
    def _analyze_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regime for mean reversion suitability."""
        close = df["close"]
        
        regime_indicators = {}
        
        # Hurst exponent (simplified)
        if self.use_regime_filter and len(close) >= 100:
            hurst = self._calculate_hurst_exponent(close.values)
            regime_indicators["hurst_exponent"] = hurst
            regime_indicators["mean_reverting"] = hurst < 0.5
            regime_indicators["trending"] = hurst > 0.5
        else:
            regime_indicators["hurst_exponent"] = 0.5
            regime_indicators["mean_reverting"] = True
            regime_indicators["trending"] = False
        
        # Volatility regime
        returns = close.pct_change().dropna()
        current_vol = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0
        historical_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        regime_indicators["current_volatility"] = current_vol
        regime_indicators["vol_regime"] = "high" if current_vol > historical_vol * 1.5 else "normal"
        
        # ADX for trend strength
        adx = self._calculate_simple_adx(df)
        regime_indicators["adx"] = adx
        regime_indicators["strong_trend"] = adx > 25
        
        # Suitable for mean reversion
        suitable = (
            regime_indicators.get("mean_reverting", True) and
            not regime_indicators.get("strong_trend", False) and
            regime_indicators.get("vol_regime") != "extreme"
        )
        
        regime_indicators["suitable_for_reversion"] = suitable
        
        return regime_indicators
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate simplified Hurst exponent."""
        if len(prices) < 100:
            return 0.5
        
        # Use R/S analysis (simplified)
        lags = range(2, min(20, len(prices) // 5))
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differences
            std_dev = np.std(np.diff(prices, lag))
            tau.append(std_dev)
        
        # Fit power law
        if len(tau) > 0:
            # log(R/S) = H * log(n) + c
            log_lags = np.log(list(lags))
            log_tau = np.log(tau)
            
            # Simple linear regression
            x_mean = np.mean(log_lags)
            y_mean = np.mean(log_tau)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(log_lags, log_tau))
            denominator = sum((x - x_mean) ** 2 for x in log_lags)
            
            if denominator > 0:
                hurst = numerator / denominator / 2  # Divide by 2 for Hurst scaling
                return max(0, min(1, hurst))  # Bound between 0 and 1
        
        return 0.5
    
    def _calculate_simple_adx(self, df: pd.DataFrame) -> float:
        """Calculate simplified ADX."""
        if len(df) < 14:
            return 0
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Directional indicators
        pos_di = 100 * pos_dm.rolling(window=14).mean() / atr
        neg_di = 100 * neg_dm.rolling(window=14).mean() / atr
        
        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=14).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
    
    def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for mean reversion."""
        volume = df["volume"]
        close = df["close"]
        
        # Volume moving average
        volume_ma = volume.rolling(window=self.volume_ma_period).mean()
        current_volume = volume.iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume at extremes
        bb_position = close.iloc[-1]
        period_high = close.rolling(window=20).max().iloc[-1]
        period_low = close.rolling(window=20).min().iloc[-1]
        
        at_extreme = (close.iloc[-1] > period_high * 0.98) or (close.iloc[-1] < period_low * 1.02)
        
        # Volume divergence (high volume at extremes often signals reversal)
        volume_divergence = at_extreme and volume_ratio > 1.5
        
        # Decreasing volume on move (exhaustion)
        if len(volume) >= 5:
            recent_volume_trend = volume.iloc[-5:].mean() < volume.iloc[-10:-5].mean()
        else:
            recent_volume_trend = False
        
        return {
            "current_volume": current_volume,
            "average_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "sufficient_volume": volume_ratio >= self.min_volume_ratio,
            "volume_divergence": volume_divergence,
            "volume_exhaustion": at_extreme and recent_volume_trend,
            "reversal_volume": volume_divergence or (at_extreme and recent_volume_trend),
        }
    
    def _calculate_reversion_probability(self, deviation_analysis: Dict[str, Any],
                                       regime_analysis: Dict[str, Any],
                                       volume_analysis: Dict[str, Any]) -> float:
        """Calculate probability of mean reversion."""
        probabilities = []
        
        # Deviation probability
        if deviation_analysis.get("significant"):
            dev_prob = min(deviation_analysis.get("current_deviation", 0) / 0.05, 1.0)
            if deviation_analysis.get("extreme"):
                dev_prob *= 1.2
            probabilities.append(dev_prob * 0.4)
        else:
            probabilities.append(0)
        
        # Regime probability
        if regime_analysis.get("suitable_for_reversion"):
            regime_prob = 0.8
            if regime_analysis.get("hurst_exponent", 0.5) < 0.4:
                regime_prob = 0.9
            probabilities.append(regime_prob * 0.3)
        else:
            probabilities.append(0.2)
        
        # Volume probability
        if volume_analysis.get("reversal_volume"):
            probabilities.append(0.8 * 0.2)
        elif volume_analysis.get("sufficient_volume"):
            probabilities.append(0.5 * 0.2)
        else:
            probabilities.append(0.2 * 0.2)
        
        # Time-based probability (longer deviations more likely to revert)
        streak = deviation_analysis.get("deviation_streak", 0)
        if streak > 10:
            time_prob = min(streak / 20, 1.0)
            probabilities.append(time_prob * 0.1)
        else:
            probabilities.append(0)
        
        return min(sum(probabilities), 0.95)  # Cap at 95%
    
    def _identify_entry_points(self, indicators: Dict[str, Any],
                             deviation_analysis: Dict[str, Any],
                             reversion_probability: float) -> Dict[str, Any]:
        """Identify mean reversion entry points."""
        entry_signals = {
            "long_signal": False,
            "short_signal": False,
            "long_strength": 0,
            "short_strength": 0,
        }
        
        if reversion_probability < 0.5:
            return entry_signals
        
        bb = indicators.get("bollinger_bands", {})
        z_score = indicators.get("z_score", {})
        rsi = indicators.get("rsi", {})
        
        # Long entry conditions (oversold)
        long_conditions = [
            bb.get("band_position", 0.5) < 0.2,  # Near lower band
            z_score.get("oversold", False),  # Z-score oversold
            rsi.get("oversold", False),  # RSI oversold
            deviation_analysis.get("direction") == "below",  # Below mean
            deviation_analysis.get("significant", False),  # Significant deviation
        ]
        
        # Short entry conditions (overbought)
        short_conditions = [
            bb.get("band_position", 0.5) > 0.8,  # Near upper band
            z_score.get("overbought", False),  # Z-score overbought
            rsi.get("overbought", False),  # RSI overbought
            deviation_analysis.get("direction") == "above",  # Above mean
            deviation_analysis.get("significant", False),  # Significant deviation
        ]
        
        # Calculate signal strength
        long_strength = sum(long_conditions) / len(long_conditions)
        short_strength = sum(short_conditions) / len(short_conditions)
        
        # Apply probability weight
        long_strength *= reversion_probability
        short_strength *= reversion_probability
        
        # Check for reversal signs
        if z_score.get("reverting", False):
            if z_score.get("value", 0) < 0:
                long_strength *= 1.2
            else:
                short_strength *= 1.2
        
        entry_signals["long_signal"] = long_strength > 0.5
        entry_signals["short_signal"] = short_strength > 0.5
        entry_signals["long_strength"] = long_strength
        entry_signals["short_strength"] = short_strength
        
        return entry_signals
    
    def _calculate_target_levels(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate target levels for mean reversion trades."""
        close = df["close"]
        current_price = close.iloc[-1]
        
        bb = indicators.get("bollinger_bands", {})
        stats = indicators.get("rolling_stats", {})
        
        # Primary targets
        targets = {
            "current_price": current_price,
            "bb_middle": bb.get("ma", current_price),
            "mean_20": stats.get("mean_20", current_price),
            "mean_50": stats.get("mean_50", current_price),
        }
        
        # Calculate weighted mean target
        weights = {"bb_middle": 0.4, "mean_20": 0.4, "mean_50": 0.2}
        mean_target = sum(targets[k] * v for k, v in weights.items() if k in targets)
        targets["mean_target"] = mean_target
        
        # Partial targets (50% and 75% reversion)
        if current_price != mean_target:
            targets["target_50pct"] = current_price + 0.5 * (mean_target - current_price)
            targets["target_75pct"] = current_price + 0.75 * (mean_target - current_price)
        
        # Band targets
        targets["upper_band"] = bb.get("upper", current_price * 1.02)
        targets["lower_band"] = bb.get("lower", current_price * 0.98)
        
        # Expected move based on reversion speed
        reversion_speed = indicators.get("reversion_speed", {})
        half_life = reversion_speed.get("half_life", 10)
        
        if half_life < np.inf:
            # Expected price after 1 half-life
            expected_move = 0.5 * (mean_target - current_price)
            targets["expected_1hl"] = current_price + expected_move
        
        return targets
    
    def _calculate_confidence(self, reversion_probability: float,
                            regime_analysis: Dict[str, Any],
                            volume_analysis: Dict[str, Any]) -> float:
        """Calculate overall signal confidence."""
        base_confidence = reversion_probability
        
        # Adjust for regime
        if regime_analysis.get("suitable_for_reversion"):
            base_confidence *= 1.1
        else:
            base_confidence *= 0.7
        
        # Adjust for volume
        if volume_analysis.get("reversal_volume"):
            base_confidence *= 1.2
        elif not volume_analysis.get("sufficient_volume"):
            base_confidence *= 0.8
        
        # Adjust for trend strength
        if regime_analysis.get("strong_trend"):
            base_confidence *= 0.6
        
        return min(base_confidence, 1.0)
    
    def _validate_signal_custom(self, signal: Signal) -> bool:
        """Custom validation for mean reversion signals."""
        # Check if we're not fighting a strong trend
        metadata = signal.metadata or {}
        
        # Ensure we have minimum deviation
        deviation = metadata.get("deviation", 0)
        if abs(deviation) < self.min_deviation:
            return False
        
        # Ensure reasonable z-score
        z_score = metadata.get("z_score", 0)
        if abs(z_score) < 1.5:  # Not extreme enough
            return False
        
        return True