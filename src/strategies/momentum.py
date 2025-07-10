"""
Momentum trading strategy implementation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.strategies.base import BaseStrategy
from src.core.logging import get_logger
from src.trading.models import Signal, OrderSide

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Identifies and trades assets showing strong directional momentum
    using various technical indicators and price action analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize momentum strategy."""
        super().__init__("MomentumStrategy", config)
    
    def _initialize_parameters(self):
        """Initialize momentum-specific parameters."""
        # Momentum indicators
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        
        # Moving averages
        self.fast_ma = self.config.get("fast_ma", 20)
        self.slow_ma = self.config.get("slow_ma", 50)
        self.volume_ma = self.config.get("volume_ma", 20)
        
        # Momentum parameters
        self.momentum_period = self.config.get("momentum_period", 10)
        self.min_momentum = self.config.get("min_momentum", 0.02)  # 2% minimum momentum
        self.volume_multiplier = self.config.get("volume_multiplier", 1.5)
        
        # ADX for trend strength
        self.adx_period = self.config.get("adx_period", 14)
        self.min_adx = self.config.get("min_adx", 25)  # Minimum trend strength
        
        # Breakout detection
        self.breakout_period = self.config.get("breakout_period", 20)
        self.breakout_threshold = self.config.get("breakout_threshold", 0.02)  # 2%
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for momentum opportunities.
        
        Args:
            market_data: OHLCV and other market data
            
        Returns:
            Momentum analysis results
        """
        try:
            # Prepare data
            df = self._prepare_dataframe(market_data)
            
            if len(df) < self.slow_ma:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(df)
            
            # Detect momentum patterns
            patterns = self._detect_momentum_patterns(df, momentum_indicators)
            
            # Analyze volume confirmation
            volume_analysis = self._analyze_volume(df)
            
            # Check for breakouts
            breakouts = self._detect_breakouts(df)
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(
                momentum_indicators, patterns, volume_analysis, breakouts
            )
            
            # Determine signal strength and direction
            signal_direction = self._determine_signal_direction(
                momentum_indicators, patterns, breakouts
            )
            
            analysis = {
                "symbol": market_data.get("symbol"),
                "timestamp": datetime.utcnow().isoformat(),
                "momentum_indicators": momentum_indicators,
                "patterns": patterns,
                "volume_analysis": volume_analysis,
                "breakouts": breakouts,
                "momentum_score": momentum_score,
                "signal_direction": signal_direction,
                "confidence": self._calculate_confidence(momentum_score, volume_analysis),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return {"error": str(e)}
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals from momentum analysis."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        confidence = analysis.get("confidence", 0)
        if confidence < self.min_confidence:
            return signals
        
        signal_direction = analysis.get("signal_direction")
        momentum_score = analysis.get("momentum_score", 0)
        
        if signal_direction == "bullish" and momentum_score > 0.6:
            # Generate buy signal
            entry_price = analysis.get("breakouts", {}).get("current_price", 0)
            
            signal = Signal(
                symbol=analysis["symbol"],
                side=OrderSide.BUY,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=self.calculate_stop_loss(entry_price, OrderSide.BUY),
                take_profit=self.calculate_take_profit(entry_price, OrderSide.BUY),
                strategy=self.name,
                metadata={
                    "momentum_score": momentum_score,
                    "patterns": analysis.get("patterns", {}),
                    "indicators": analysis.get("momentum_indicators", {}),
                }
            )
            
            if self.validate_signal(signal):
                signals.append(signal)
                
        elif signal_direction == "bearish" and momentum_score > 0.6:
            # Generate sell signal
            entry_price = analysis.get("breakouts", {}).get("current_price", 0)
            
            signal = Signal(
                symbol=analysis["symbol"],
                side=OrderSide.SELL,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=self.calculate_stop_loss(entry_price, OrderSide.SELL),
                take_profit=self.calculate_take_profit(entry_price, OrderSide.SELL),
                strategy=self.name,
                metadata={
                    "momentum_score": momentum_score,
                    "patterns": analysis.get("patterns", {}),
                    "indicators": analysis.get("momentum_indicators", {}),
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
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators."""
        indicators = {}
        
        # Price momentum
        indicators["momentum"] = self._calculate_momentum(df)
        
        # RSI
        indicators["rsi"] = self._calculate_rsi(df)
        
        # Moving averages
        indicators["moving_averages"] = self._calculate_moving_averages(df)
        
        # ADX for trend strength
        indicators["adx"] = self._calculate_adx(df)
        
        # MACD
        indicators["macd"] = self._calculate_macd(df)
        
        # Rate of Change (ROC)
        indicators["roc"] = self._calculate_roc(df)
        
        return indicators
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price momentum."""
        close_prices = df["close"]
        
        # Simple momentum
        momentum = close_prices.pct_change(self.momentum_period)
        current_momentum = momentum.iloc[-1]
        
        # Momentum acceleration
        momentum_change = momentum.diff()
        acceleration = momentum_change.iloc[-1]
        
        # Momentum strength (normalized)
        momentum_std = momentum.std()
        if momentum_std > 0:
            normalized_momentum = current_momentum / momentum_std
        else:
            normalized_momentum = 0
        
        return {
            "value": current_momentum,
            "acceleration": acceleration,
            "normalized": normalized_momentum,
            "strong": abs(current_momentum) > self.min_momentum,
            "direction": "bullish" if current_momentum > 0 else "bearish",
        }
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI."""
        close_prices = df["close"]
        
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # RSI momentum (change in RSI)
        rsi_momentum = rsi.diff(5).iloc[-1]
        
        return {
            "value": current_rsi,
            "momentum": rsi_momentum,
            "overbought": current_rsi > self.rsi_overbought,
            "oversold": current_rsi < self.rsi_oversold,
            "signal": self._get_rsi_signal(current_rsi, rsi_momentum),
        }
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving averages and crossovers."""
        close_prices = df["close"]
        
        fast_ma = close_prices.rolling(window=self.fast_ma).mean()
        slow_ma = close_prices.rolling(window=self.slow_ma).mean()
        
        current_price = close_prices.iloc[-1]
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        
        # MA slopes
        fast_slope = (current_fast - fast_ma.iloc[-5]) / 5 if len(fast_ma) > 5 else 0
        slow_slope = (current_slow - slow_ma.iloc[-5]) / 5 if len(slow_ma) > 5 else 0
        
        # Crossover detection
        crossover = None
        if len(fast_ma) > 2 and len(slow_ma) > 2:
            if fast_ma.iloc[-2] <= slow_ma.iloc[-2] and current_fast > current_slow:
                crossover = "golden_cross"
            elif fast_ma.iloc[-2] >= slow_ma.iloc[-2] and current_fast < current_slow:
                crossover = "death_cross"
        
        return {
            "fast_ma": current_fast,
            "slow_ma": current_slow,
            "fast_slope": fast_slope,
            "slow_slope": slow_slope,
            "price_above_fast": current_price > current_fast,
            "price_above_slow": current_price > current_slow,
            "crossover": crossover,
            "trend": "bullish" if current_fast > current_slow else "bearish",
        }
    
    def _calculate_adx(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Average Directional Index (ADX)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=self.adx_period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=self.adx_period).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=self.adx_period).mean()
        
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        current_plus_di = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0
        current_minus_di = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
        
        return {
            "value": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "strong_trend": current_adx > self.min_adx,
            "trend_direction": "bullish" if current_plus_di > current_minus_di else "bearish",
        }
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD."""
        close_prices = df["close"]
        
        # MACD line
        exp1 = close_prices.ewm(span=12, adjust=False).mean()
        exp2 = close_prices.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        
        # Signal line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Check for crossovers
        crossover = None
        if len(histogram) > 2:
            if histogram.iloc[-2] < 0 and current_histogram > 0:
                crossover = "bullish"
            elif histogram.iloc[-2] > 0 and current_histogram < 0:
                crossover = "bearish"
        
        return {
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_histogram,
            "crossover": crossover,
            "momentum": "bullish" if current_histogram > 0 else "bearish",
        }
    
    def _calculate_roc(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Rate of Change (ROC)."""
        close_prices = df["close"]
        
        # ROC calculation
        roc = ((close_prices - close_prices.shift(self.momentum_period)) / 
               close_prices.shift(self.momentum_period)) * 100
        
        current_roc = roc.iloc[-1]
        roc_ma = roc.rolling(window=5).mean().iloc[-1]
        
        return {
            "value": current_roc,
            "ma": roc_ma,
            "strong": abs(current_roc) > 5,  # 5% ROC threshold
            "direction": "bullish" if current_roc > 0 else "bearish",
        }
    
    def _detect_momentum_patterns(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect momentum patterns."""
        patterns = {
            "higher_highs": self._detect_higher_highs(df),
            "lower_lows": self._detect_lower_lows(df),
            "momentum_divergence": self._detect_divergence(df, indicators),
            "consolidation_breakout": self._detect_consolidation_breakout(df),
        }
        
        # Pattern strength
        bullish_patterns = sum([
            patterns["higher_highs"],
            patterns.get("momentum_divergence") == "bullish",
            patterns.get("consolidation_breakout") == "bullish",
        ])
        
        bearish_patterns = sum([
            patterns["lower_lows"],
            patterns.get("momentum_divergence") == "bearish",
            patterns.get("consolidation_breakout") == "bearish",
        ])
        
        patterns["pattern_strength"] = {
            "bullish": bullish_patterns / 3,
            "bearish": bearish_patterns / 3,
        }
        
        return patterns
    
    def _detect_higher_highs(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect higher highs pattern."""
        if len(df) < lookback:
            return False
        
        highs = df["high"].iloc[-lookback:]
        
        # Find peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                peaks.append((i, highs.iloc[i]))
        
        # Check if peaks are ascending
        if len(peaks) >= 2:
            return all(peaks[i][1] < peaks[i+1][1] for i in range(len(peaks)-1))
        
        return False
    
    def _detect_lower_lows(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect lower lows pattern."""
        if len(df) < lookback:
            return False
        
        lows = df["low"].iloc[-lookback:]
        
        # Find troughs
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                troughs.append((i, lows.iloc[i]))
        
        # Check if troughs are descending
        if len(troughs) >= 2:
            return all(troughs[i][1] > troughs[i+1][1] for i in range(len(troughs)-1))
        
        return False
    
    def _detect_divergence(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[str]:
        """Detect momentum divergence."""
        if len(df) < 20:
            return None
        
        # Price trend
        price_trend = df["close"].iloc[-10] < df["close"].iloc[-1]
        
        # RSI trend
        rsi_values = self._calculate_rsi_series(df)
        if len(rsi_values) >= 10:
            rsi_trend = rsi_values.iloc[-10] < rsi_values.iloc[-1]
            
            # Bullish divergence: price down, RSI up
            if not price_trend and rsi_trend:
                return "bullish"
            # Bearish divergence: price up, RSI down
            elif price_trend and not rsi_trend:
                return "bearish"
        
        return None
    
    def _detect_consolidation_breakout(self, df: pd.DataFrame) -> Optional[str]:
        """Detect breakout from consolidation."""
        if len(df) < 20:
            return None
        
        # Calculate recent volatility
        recent_range = df["high"].iloc[-20:] - df["low"].iloc[-20:]
        avg_range = recent_range.mean()
        
        # Check if price was consolidating
        consolidation = recent_range.iloc[-10:-1].mean() < avg_range * 0.7
        
        if consolidation:
            current_price = df["close"].iloc[-1]
            high_20 = df["high"].iloc[-20:-1].max()
            low_20 = df["low"].iloc[-20:-1].min()
            
            # Breakout detection
            if current_price > high_20:
                return "bullish"
            elif current_price < low_20:
                return "bearish"
        
        return None
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns."""
        volume = df["volume"]
        volume_ma = volume.rolling(window=self.volume_ma).mean()
        
        current_volume = volume.iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        # Volume analysis
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        volume_trend = "increasing" if volume.iloc[-5:].mean() > avg_volume else "decreasing"
        
        # Price-volume correlation
        price_change = df["close"].pct_change().iloc[-1]
        volume_confirms = (price_change > 0 and volume_ratio > 1) or (price_change < 0 and volume_ratio > 1)
        
        return {
            "current": current_volume,
            "average": avg_volume,
            "ratio": volume_ratio,
            "trend": volume_trend,
            "high_volume": volume_ratio > self.volume_multiplier,
            "confirms_price": volume_confirms,
        }
    
    def _detect_breakouts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect price breakouts."""
        current_price = df["close"].iloc[-1]
        
        # Recent high/low
        period_high = df["high"].iloc[-self.breakout_period:-1].max()
        period_low = df["low"].iloc[-self.breakout_period:-1].min()
        
        # Breakout detection
        breakout_up = current_price > period_high * (1 + self.breakout_threshold)
        breakout_down = current_price < period_low * (1 - self.breakout_threshold)
        
        # Breakout strength
        if breakout_up:
            breakout_strength = (current_price - period_high) / period_high
        elif breakout_down:
            breakout_strength = (period_low - current_price) / period_low
        else:
            breakout_strength = 0
        
        return {
            "current_price": current_price,
            "period_high": period_high,
            "period_low": period_low,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "breakout_strength": breakout_strength,
            "direction": "bullish" if breakout_up else "bearish" if breakout_down else "none",
        }
    
    def _calculate_momentum_score(self, indicators: Dict[str, Any], patterns: Dict[str, Any],
                                volume: Dict[str, Any], breakouts: Dict[str, Any]) -> float:
        """Calculate overall momentum score."""
        scores = []
        
        # Momentum indicator score
        momentum = indicators.get("momentum", {})
        if momentum.get("strong"):
            scores.append(0.8 if momentum.get("normalized", 0) > 1 else 0.6)
        else:
            scores.append(0.3)
        
        # RSI score
        rsi = indicators.get("rsi", {})
        if rsi.get("signal") in ["bullish", "bearish"]:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # ADX trend strength
        adx = indicators.get("adx", {})
        if adx.get("strong_trend"):
            scores.append(0.8)
        else:
            scores.append(0.3)
        
        # Pattern score
        pattern_strength = patterns.get("pattern_strength", {})
        max_pattern = max(pattern_strength.get("bullish", 0), pattern_strength.get("bearish", 0))
        scores.append(max_pattern)
        
        # Volume confirmation
        if volume.get("high_volume") and volume.get("confirms_price"):
            scores.append(0.9)
        elif volume.get("confirms_price"):
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        # Breakout score
        if breakouts.get("direction") != "none":
            scores.append(min(breakouts.get("breakout_strength", 0) * 10, 1.0))
        else:
            scores.append(0.2)
        
        return np.mean(scores) if scores else 0
    
    def _determine_signal_direction(self, indicators: Dict[str, Any], patterns: Dict[str, Any],
                                  breakouts: Dict[str, Any]) -> str:
        """Determine overall signal direction."""
        bullish_count = 0
        bearish_count = 0
        
        # Check each indicator
        if indicators.get("momentum", {}).get("direction") == "bullish":
            bullish_count += 1
        else:
            bearish_count += 1
        
        if indicators.get("moving_averages", {}).get("trend") == "bullish":
            bullish_count += 1
        else:
            bearish_count += 1
        
        if indicators.get("adx", {}).get("trend_direction") == "bullish":
            bullish_count += 1
        else:
            bearish_count += 1
        
        if indicators.get("macd", {}).get("momentum") == "bullish":
            bullish_count += 1
        else:
            bearish_count += 1
        
        # Check patterns
        pattern_strength = patterns.get("pattern_strength", {})
        if pattern_strength.get("bullish", 0) > pattern_strength.get("bearish", 0):
            bullish_count += 1
        elif pattern_strength.get("bearish", 0) > pattern_strength.get("bullish", 0):
            bearish_count += 1
        
        # Check breakouts
        if breakouts.get("direction") == "bullish":
            bullish_count += 2  # Double weight for breakouts
        elif breakouts.get("direction") == "bearish":
            bearish_count += 2
        
        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_confidence(self, momentum_score: float, volume_analysis: Dict[str, Any]) -> float:
        """Calculate signal confidence."""
        base_confidence = momentum_score
        
        # Adjust for volume
        if volume_analysis.get("high_volume") and volume_analysis.get("confirms_price"):
            base_confidence *= 1.2
        elif not volume_analysis.get("confirms_price"):
            base_confidence *= 0.8
        
        return min(base_confidence, 1.0)
    
    def _get_rsi_signal(self, rsi: float, rsi_momentum: float) -> str:
        """Determine RSI signal."""
        if rsi < self.rsi_oversold and rsi_momentum > 0:
            return "bullish"
        elif rsi > self.rsi_overbought and rsi_momentum < 0:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_rsi_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI series for divergence detection."""
        close_prices = df["close"]
        
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi