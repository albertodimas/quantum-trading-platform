"""
Technical Analysis Agent using indicators and patterns.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.agents.base import BaseAgent
from src.core.logging import get_logger
from src.trading.models import OrderSide

logger = get_logger(__name__)


class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent specialized in technical analysis of market data.
    
    Analyzes price patterns, indicators, and market structure
    to generate trading signals.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize technical analysis agent."""
        super().__init__("TechnicalAnalysisAgent", config)
        
        # Default indicator settings
        self.indicators_config = {
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "std_dev": 2},
            "ema": {"periods": [9, 21, 50, 200]},
            "volume": {"ma_period": 20},
        }
        
        # Update with custom config
        if config and "indicators" in config:
            self.indicators_config.update(config["indicators"])
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform technical analysis on market data.
        
        Args:
            data: Market data including OHLCV
            
        Returns:
            Technical analysis results
        """
        if not await self.validate_data(data):
            return {"error": "Invalid data provided"}
        
        try:
            # Convert to DataFrame for easier analysis
            df = self._prepare_dataframe(data)
            
            # Calculate indicators
            indicators = await self._calculate_indicators(df)
            
            # Detect patterns
            patterns = await self._detect_patterns(df, indicators)
            
            # Analyze market structure
            structure = await self._analyze_market_structure(df, indicators)
            
            # Generate overall analysis
            analysis = {
                "symbol": data["symbol"],
                "timeframe": data.get("timeframe", "1h"),
                "indicators": indicators,
                "patterns": patterns,
                "market_structure": structure,
                "signal_strength": self._calculate_signal_strength({
                    "indicators": indicators.get("composite_score", 0.5),
                    "patterns": patterns.get("strength", 0.5),
                    "structure": structure.get("trend_strength", 0.5),
                }),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            await self._store_analysis(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {"error": str(e)}
    
    async def get_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from technical analysis."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        signal_strength = analysis.get("signal_strength", 0)
        
        # Only generate signals above confidence threshold
        if signal_strength < self._confidence_threshold:
            return signals
        
        # Determine signal direction
        indicators = analysis.get("indicators", {})
        patterns = analysis.get("patterns", {})
        structure = analysis.get("market_structure", {})
        
        # Bull signals
        bull_score = 0
        bear_score = 0
        
        # RSI signals
        rsi = indicators.get("rsi", {}).get("value", 50)
        if rsi < 30:
            bull_score += 0.3
        elif rsi > 70:
            bear_score += 0.3
        
        # MACD signals
        macd_signal = indicators.get("macd", {}).get("signal", "neutral")
        if macd_signal == "bullish":
            bull_score += 0.3
        elif macd_signal == "bearish":
            bear_score += 0.3
        
        # Pattern signals
        if patterns.get("bullish_patterns"):
            bull_score += 0.2 * len(patterns["bullish_patterns"])
        if patterns.get("bearish_patterns"):
            bear_score += 0.2 * len(patterns["bearish_patterns"])
        
        # Trend signals
        trend = structure.get("trend", "neutral")
        if trend == "uptrend":
            bull_score += 0.2
        elif trend == "downtrend":
            bear_score += 0.2
        
        # Generate signal if clear direction
        if bull_score > bear_score and bull_score > 0.5:
            signals.append({
                "type": "technical",
                "action": "buy",
                "symbol": analysis["symbol"],
                "confidence": min(bull_score, 1.0),
                "reasons": self._get_signal_reasons(analysis, "bullish"),
                "entry_price": structure.get("current_price"),
                "stop_loss": structure.get("support_levels", [0])[0] if structure.get("support_levels") else None,
                "take_profit": structure.get("resistance_levels", [0])[0] if structure.get("resistance_levels") else None,
            })
        elif bear_score > bull_score and bear_score > 0.5:
            signals.append({
                "type": "technical",
                "action": "sell",
                "symbol": analysis["symbol"],
                "confidence": min(bear_score, 1.0),
                "reasons": self._get_signal_reasons(analysis, "bearish"),
                "entry_price": structure.get("current_price"),
                "stop_loss": structure.get("resistance_levels", [0])[0] if structure.get("resistance_levels") else None,
                "take_profit": structure.get("support_levels", [0])[0] if structure.get("support_levels") else None,
            })
        
        return signals
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for technical analysis."""
        return ["symbol", "ohlcv"]
    
    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare DataFrame from OHLCV data."""
        ohlcv = data["ohlcv"]
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df.sort_index()
    
    async def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators."""
        indicators = {}
        
        # RSI
        indicators["rsi"] = self._calculate_rsi(df)
        
        # MACD
        indicators["macd"] = self._calculate_macd(df)
        
        # Bollinger Bands
        indicators["bollinger"] = self._calculate_bollinger_bands(df)
        
        # Moving Averages
        indicators["ema"] = self._calculate_emas(df)
        
        # Volume indicators
        indicators["volume"] = self._calculate_volume_indicators(df)
        
        # Composite score
        indicators["composite_score"] = self._calculate_composite_score(indicators)
        
        return indicators
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        period = self.indicators_config["rsi"]["period"]
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        return {
            "value": current_rsi,
            "oversold": current_rsi < self.indicators_config["rsi"]["oversold"],
            "overbought": current_rsi > self.indicators_config["rsi"]["overbought"],
            "trend": "bullish" if current_rsi > 50 else "bearish",
        }
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD indicator."""
        config = self.indicators_config["macd"]
        
        exp1 = df["close"].ewm(span=config["fast"], adjust=False).mean()
        exp2 = df["close"].ewm(span=config["slow"], adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=config["signal"], adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        return {
            "macd": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": current_hist,
            "signal": "bullish" if current_hist > 0 and current_hist > prev_hist else "bearish",
            "divergence": self._check_divergence(df["close"], macd_line),
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        config = self.indicators_config["bollinger"]
        
        sma = df["close"].rolling(window=config["period"]).mean()
        std = df["close"].rolling(window=config["period"]).std()
        
        upper = sma + (std * config["std_dev"])
        lower = sma - (std * config["std_dev"])
        
        current_price = df["close"].iloc[-1]
        
        return {
            "upper": upper.iloc[-1],
            "middle": sma.iloc[-1],
            "lower": lower.iloc[-1],
            "position": (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]),
            "squeeze": std.iloc[-1] < std.rolling(window=20).mean().iloc[-1],
        }
    
    def _calculate_emas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Exponential Moving Averages."""
        emas = {}
        current_price = df["close"].iloc[-1]
        
        for period in self.indicators_config["ema"]["periods"]:
            ema = df["close"].ewm(span=period, adjust=False).mean()
            emas[f"ema_{period}"] = {
                "value": ema.iloc[-1],
                "trend": "above" if current_price > ema.iloc[-1] else "below",
                "slope": (ema.iloc[-1] - ema.iloc[-5]) / 5 if len(ema) > 5 else 0,
            }
        
        return emas
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators."""
        volume_ma = df["volume"].rolling(window=self.indicators_config["volume"]["ma_period"]).mean()
        current_volume = df["volume"].iloc[-1]
        
        return {
            "current": current_volume,
            "average": volume_ma.iloc[-1],
            "ratio": current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1,
            "trend": "increasing" if current_volume > volume_ma.iloc[-1] else "decreasing",
        }
    
    async def _detect_patterns(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect chart patterns."""
        patterns = {
            "bullish_patterns": [],
            "bearish_patterns": [],
            "neutral_patterns": [],
        }
        
        # Candlestick patterns
        if self._is_hammer(df):
            patterns["bullish_patterns"].append("hammer")
        if self._is_shooting_star(df):
            patterns["bearish_patterns"].append("shooting_star")
        if self._is_doji(df):
            patterns["neutral_patterns"].append("doji")
        
        # Chart patterns
        if self._is_double_bottom(df):
            patterns["bullish_patterns"].append("double_bottom")
        if self._is_double_top(df):
            patterns["bearish_patterns"].append("double_top")
        
        # Calculate pattern strength
        total_patterns = len(patterns["bullish_patterns"]) + len(patterns["bearish_patterns"])
        patterns["strength"] = min(total_patterns * 0.2, 1.0)
        
        return patterns
    
    async def _analyze_market_structure(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market structure."""
        current_price = df["close"].iloc[-1]
        
        # Trend analysis
        trend = self._determine_trend(df)
        
        # Support/Resistance
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)
        
        # Volatility
        volatility = df["close"].pct_change().std() * np.sqrt(252)
        
        return {
            "current_price": current_price,
            "trend": trend["direction"],
            "trend_strength": trend["strength"],
            "support_levels": support_levels[:3],  # Top 3 levels
            "resistance_levels": resistance_levels[:3],
            "volatility": volatility,
            "market_phase": self._determine_market_phase(trend, volatility),
        }
    
    def _calculate_composite_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate composite score from all indicators."""
        scores = []
        
        # RSI score
        rsi = indicators.get("rsi", {})
        if rsi.get("oversold"):
            scores.append(0.8)
        elif rsi.get("overbought"):
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        # MACD score
        macd = indicators.get("macd", {})
        if macd.get("signal") == "bullish":
            scores.append(0.7)
        elif macd.get("signal") == "bearish":
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Bollinger score
        bb = indicators.get("bollinger", {})
        if bb.get("position", 0.5) < 0.2:
            scores.append(0.8)
        elif bb.get("position", 0.5) > 0.8:
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        return np.mean(scores) if scores else 0.5
    
    def _is_hammer(self, df: pd.DataFrame) -> bool:
        """Check if last candle is a hammer pattern."""
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        lower_shadow = min(last["open"], last["close"]) - last["low"]
        upper_shadow = last["high"] - max(last["open"], last["close"])
        
        return lower_shadow > 2 * body and upper_shadow < body * 0.5
    
    def _is_shooting_star(self, df: pd.DataFrame) -> bool:
        """Check if last candle is a shooting star pattern."""
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        lower_shadow = min(last["open"], last["close"]) - last["low"]
        upper_shadow = last["high"] - max(last["open"], last["close"])
        
        return upper_shadow > 2 * body and lower_shadow < body * 0.5
    
    def _is_doji(self, df: pd.DataFrame) -> bool:
        """Check if last candle is a doji pattern."""
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        total_range = last["high"] - last["low"]
        
        return body < total_range * 0.1
    
    def _is_double_bottom(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect double bottom pattern."""
        if len(df) < lookback:
            return False
        
        lows = df["low"].iloc[-lookback:].values
        # Simplified detection - find two similar lows
        min_low = np.min(lows)
        similar_lows = np.where(np.abs(lows - min_low) < min_low * 0.02)[0]
        
        return len(similar_lows) >= 2 and similar_lows[-1] - similar_lows[0] > 5
    
    def _is_double_top(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect double top pattern."""
        if len(df) < lookback:
            return False
        
        highs = df["high"].iloc[-lookback:].values
        # Simplified detection - find two similar highs
        max_high = np.max(highs)
        similar_highs = np.where(np.abs(highs - max_high) < max_high * 0.02)[0]
        
        return len(similar_highs) >= 2 and similar_highs[-1] - similar_highs[0] > 5
    
    def _determine_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine market trend."""
        # Simple trend detection using moving averages
        ma_short = df["close"].rolling(window=20).mean()
        ma_long = df["close"].rolling(window=50).mean()
        
        if len(df) < 50:
            return {"direction": "neutral", "strength": 0.5}
        
        current_short = ma_short.iloc[-1]
        current_long = ma_long.iloc[-1]
        
        if current_short > current_long * 1.02:
            direction = "uptrend"
            strength = min((current_short / current_long - 1) * 10, 1.0)
        elif current_short < current_long * 0.98:
            direction = "downtrend"
            strength = min((1 - current_short / current_long) * 10, 1.0)
        else:
            direction = "neutral"
            strength = 0.5
        
        return {"direction": direction, "strength": strength}
    
    def _find_support_levels(self, df: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Find support levels."""
        if len(df) < lookback:
            lookback = len(df)
        
        lows = df["low"].iloc[-lookback:].values
        # Find local minima
        support_levels = []
        
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(float(lows[i]))
        
        return sorted(set(support_levels))
    
    def _find_resistance_levels(self, df: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Find resistance levels."""
        if len(df) < lookback:
            lookback = len(df)
        
        highs = df["high"].iloc[-lookback:].values
        # Find local maxima
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(float(highs[i]))
        
        return sorted(set(resistance_levels), reverse=True)
    
    def _determine_market_phase(self, trend: Dict[str, Any], volatility: float) -> str:
        """Determine current market phase."""
        if trend["strength"] > 0.7:
            return "trending"
        elif volatility > 0.3:
            return "volatile"
        elif volatility < 0.1:
            return "consolidating"
        else:
            return "ranging"
    
    def _check_divergence(self, price: pd.Series, indicator: pd.Series) -> Optional[str]:
        """Check for divergence between price and indicator."""
        if len(price) < 10 or len(indicator) < 10:
            return None
        
        # Check last 10 periods
        price_trend = price.iloc[-10:].iloc[-1] > price.iloc[-10:].iloc[0]
        indicator_trend = indicator.iloc[-10:].iloc[-1] > indicator.iloc[-10:].iloc[0]
        
        if price_trend and not indicator_trend:
            return "bearish_divergence"
        elif not price_trend and indicator_trend:
            return "bullish_divergence"
        
        return None
    
    def _get_signal_reasons(self, analysis: Dict[str, Any], direction: str) -> List[str]:
        """Get reasons for the signal."""
        reasons = []
        
        indicators = analysis.get("indicators", {})
        patterns = analysis.get("patterns", {})
        structure = analysis.get("market_structure", {})
        
        if direction == "bullish":
            if indicators.get("rsi", {}).get("oversold"):
                reasons.append("RSI oversold")
            if indicators.get("macd", {}).get("signal") == "bullish":
                reasons.append("MACD bullish crossover")
            if patterns.get("bullish_patterns"):
                reasons.extend(patterns["bullish_patterns"])
            if structure.get("trend") == "uptrend":
                reasons.append("Uptrend confirmed")
        else:
            if indicators.get("rsi", {}).get("overbought"):
                reasons.append("RSI overbought")
            if indicators.get("macd", {}).get("signal") == "bearish":
                reasons.append("MACD bearish crossover")
            if patterns.get("bearish_patterns"):
                reasons.extend(patterns["bearish_patterns"])
            if structure.get("trend") == "downtrend":
                reasons.append("Downtrend confirmed")
        
        return reasons