"""
AI-driven trading strategy implementation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio

from src.strategies.base import BaseStrategy
from src.agents.orchestrator import AgentOrchestrator
from src.core.logging import get_logger
from src.trading.models import Signal, OrderSide

logger = get_logger(__name__)


class AIStrategy(BaseStrategy):
    """
    AI-driven trading strategy using multi-agent system.
    
    Combines insights from technical analysis, sentiment analysis,
    and risk management agents to make intelligent trading decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AI strategy."""
        super().__init__("AIStrategy", config)
        
        # Initialize agent orchestrator
        orchestrator_config = config.get("orchestrator_config", {})
        self.orchestrator = AgentOrchestrator(orchestrator_config)
        
        # Strategy state
        self._orchestrator_started = False
        self.recent_predictions = []
        self.model_performance = {
            "predictions": 0,
            "correct": 0,
            "accuracy": 0,
        }
    
    def _initialize_parameters(self):
        """Initialize AI strategy parameters."""
        # Confidence thresholds
        self.min_agent_agreement = self.config.get("min_agent_agreement", 2)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.75)
        
        # Risk parameters
        self.max_daily_signals = self.config.get("max_daily_signals", 10)
        self.cooldown_period = self.config.get("cooldown_period", 300)  # 5 minutes
        
        # Model parameters
        self.use_ensemble = self.config.get("use_ensemble", True)
        self.prediction_horizon = self.config.get("prediction_horizon", 3600)  # 1 hour
        
        # Market regime adaptation
        self.adapt_to_regime = self.config.get("adapt_to_regime", True)
        self.regime_window = self.config.get("regime_window", 24)  # hours
        
        # Signal filtering
        self.filter_contradictory = self.config.get("filter_contradictory", True)
        self.require_volume_confirmation = self.config.get("require_volume_confirmation", True)
    
    async def start(self):
        """Start the AI strategy and agent orchestrator."""
        await super().start()
        
        if not self._orchestrator_started:
            await self.orchestrator.start()
            self._orchestrator_started = True
            logger.info("AI Strategy agent orchestrator started")
    
    async def stop(self):
        """Stop the AI strategy and agent orchestrator."""
        await super().stop()
        
        if self._orchestrator_started:
            await self.orchestrator.stop()
            self._orchestrator_started = False
            logger.info("AI Strategy agent orchestrator stopped")
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using AI agents.
        
        Args:
            market_data: Comprehensive market data
            
        Returns:
            AI-driven analysis results
        """
        try:
            # Ensure orchestrator is running
            if not self._orchestrator_started:
                await self.start()
            
            # Prepare enhanced market data
            enhanced_data = await self._enhance_market_data(market_data)
            
            # Get multi-agent analysis
            agent_analysis = await self.orchestrator.analyze_market(enhanced_data)
            
            # Extract insights
            insights = self._extract_insights(agent_analysis)
            
            # Make predictions
            predictions = await self._make_predictions(enhanced_data, insights)
            
            # Determine market regime
            market_regime = self._identify_market_regime(enhanced_data, insights)
            
            # Calculate signal strength
            signal_data = self._calculate_signal_strength(insights, predictions, market_regime)
            
            # Risk assessment
            risk_assessment = self._assess_signal_risk(signal_data, insights)
            
            analysis = {
                "symbol": market_data.get("symbol"),
                "timestamp": datetime.utcnow().isoformat(),
                "agent_analysis": agent_analysis,
                "insights": insights,
                "predictions": predictions,
                "market_regime": market_regime,
                "signal_data": signal_data,
                "risk_assessment": risk_assessment,
                "confidence": self._calculate_overall_confidence(
                    insights, predictions, risk_assessment
                ),
            }
            
            # Store for learning
            self._store_prediction(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI strategy analysis error: {e}")
            return {"error": str(e)}
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals from AI analysis."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        confidence = analysis.get("confidence", 0)
        if confidence < self.min_confidence:
            return signals
        
        signal_data = analysis.get("signal_data", {})
        risk_assessment = analysis.get("risk_assessment", {})
        
        # Check if we should generate a signal
        if not self._should_generate_signal(signal_data, risk_assessment):
            return signals
        
        # Determine signal type
        signal_type = signal_data.get("signal_type")
        
        if signal_type == "buy":
            signal = await self._create_buy_signal(analysis)
            if signal and self.validate_signal(signal):
                signals.append(signal)
                
        elif signal_type == "sell":
            signal = await self._create_sell_signal(analysis)
            if signal and self.validate_signal(signal):
                signals.append(signal)
        
        # Apply final filters
        filtered_signals = self._apply_signal_filters(signals, analysis)
        
        return filtered_signals
    
    async def _enhance_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance market data with additional features."""
        enhanced = market_data.copy()
        
        # Add derived features
        if "ohlcv" in market_data:
            df = pd.DataFrame(market_data["ohlcv"])
            
            # Price features
            enhanced["returns"] = df["close"].pct_change().tolist()
            enhanced["volatility"] = df["close"].pct_change().std() * np.sqrt(252)
            enhanced["volume_profile"] = self._calculate_volume_profile(df)
            
            # Microstructure features
            enhanced["spread"] = ((df["high"] - df["low"]) / df["close"]).mean()
            enhanced["liquidity"] = df["volume"].mean()
        
        # Add market context
        enhanced["market_hours"] = self._get_market_hours()
        enhanced["day_of_week"] = datetime.utcnow().weekday()
        
        return enhanced
    
    def _extract_insights(self, agent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from agent analysis."""
        insights = {
            "technical": {},
            "sentiment": {},
            "risk": {},
            "consensus": {},
        }
        
        individual_analyses = agent_analysis.get("individual_analyses", {})
        
        # Technical insights
        if "technical" in individual_analyses:
            tech = individual_analyses["technical"]
            insights["technical"] = {
                "trend": tech.get("market_structure", {}).get("trend"),
                "strength": tech.get("signal_strength", 0),
                "patterns": tech.get("patterns", {}),
                "key_levels": tech.get("support_resistance", {}),
            }
        
        # Sentiment insights
        if "sentiment" in individual_analyses:
            sent = individual_analyses["sentiment"]
            insights["sentiment"] = {
                "overall": sent.get("overall_sentiment", {}),
                "news_impact": sent.get("news_sentiment", {}).get("impact"),
                "social_buzz": sent.get("social_sentiment", {}).get("volume"),
                "sentiment_shift": sent.get("sentiment_trend"),
            }
        
        # Risk insights
        if "risk" in individual_analyses:
            risk = individual_analyses["risk"]
            insights["risk"] = {
                "portfolio_risk": risk.get("portfolio_metrics", {}).get("var_95", 0),
                "market_risk": risk.get("market_risk", {}).get("risk_level"),
                "position_risks": risk.get("position_risks", {}),
                "recommendations": risk.get("recommendations", []),
            }
        
        # Consensus insights
        aggregated = agent_analysis.get("aggregated", {})
        insights["consensus"] = {
            "market_view": aggregated.get("market_consensus"),
            "confidence": aggregated.get("confidence", 0),
            "key_factors": aggregated.get("key_factors", []),
        }
        
        return insights
    
    async def _make_predictions(self, market_data: Dict[str, Any], 
                              insights: Dict[str, Any]) -> Dict[str, Any]:
        """Make price predictions using AI insights."""
        predictions = {
            "direction": "neutral",
            "probability": 0.5,
            "expected_move": 0,
            "time_horizon": self.prediction_horizon,
        }
        
        # Aggregate directional signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Technical signals
        tech_trend = insights["technical"].get("trend")
        if tech_trend == "uptrend":
            bullish_signals += insights["technical"].get("strength", 0)
        elif tech_trend == "downtrend":
            bearish_signals += insights["technical"].get("strength", 0)
        
        # Sentiment signals
        sentiment_direction = insights["sentiment"].get("overall", {}).get("direction")
        if sentiment_direction == "bullish":
            bullish_signals += 0.8
        elif sentiment_direction == "bearish":
            bearish_signals += 0.8
        
        # Market consensus
        consensus = insights["consensus"].get("market_view")
        if consensus == "bullish":
            bullish_signals += insights["consensus"].get("confidence", 0)
        elif consensus == "bearish":
            bearish_signals += insights["consensus"].get("confidence", 0)
        
        # Calculate prediction
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            bullish_prob = bullish_signals / total_signals
            
            if bullish_prob > 0.6:
                predictions["direction"] = "up"
                predictions["probability"] = bullish_prob
            elif bullish_prob < 0.4:
                predictions["direction"] = "down"
                predictions["probability"] = 1 - bullish_prob
            
            # Estimate expected move based on volatility
            volatility = market_data.get("volatility", 0.2)
            time_factor = np.sqrt(self.prediction_horizon / 86400)  # Daily vol adjustment
            predictions["expected_move"] = volatility * time_factor * \
                                         (1 if predictions["direction"] == "up" else -1)
        
        # Add confidence intervals
        predictions["confidence_interval"] = {
            "lower": predictions["expected_move"] * 0.5,
            "upper": predictions["expected_move"] * 1.5,
        }
        
        return predictions
    
    def _identify_market_regime(self, market_data: Dict[str, Any], 
                              insights: Dict[str, Any]) -> Dict[str, Any]:
        """Identify current market regime."""
        regime = {
            "type": "normal",
            "volatility_regime": "medium",
            "trend_regime": "sideways",
            "sentiment_regime": "neutral",
        }
        
        # Volatility regime
        vol = market_data.get("volatility", 0.2)
        if vol > 0.4:
            regime["volatility_regime"] = "high"
            regime["type"] = "volatile"
        elif vol < 0.1:
            regime["volatility_regime"] = "low"
        
        # Trend regime
        tech_trend = insights["technical"].get("trend")
        if tech_trend == "uptrend":
            regime["trend_regime"] = "bullish"
            if regime["type"] == "normal":
                regime["type"] = "trending"
        elif tech_trend == "downtrend":
            regime["trend_regime"] = "bearish"
            if regime["type"] == "normal":
                regime["type"] = "trending"
        
        # Sentiment regime
        sentiment_score = insights["sentiment"].get("overall", {}).get("score", 0)
        if sentiment_score > 0.5:
            regime["sentiment_regime"] = "positive"
        elif sentiment_score < -0.5:
            regime["sentiment_regime"] = "negative"
        
        # Special regimes
        if regime["volatility_regime"] == "high" and abs(sentiment_score) > 0.7:
            regime["type"] = "crisis" if sentiment_score < 0 else "euphoria"
        
        return regime
    
    def _calculate_signal_strength(self, insights: Dict[str, Any],
                                 predictions: Dict[str, Any],
                                 market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal strength and type."""
        signal_data = {
            "signal_type": "none",
            "strength": 0,
            "reasons": [],
        }
        
        # Check prediction confidence
        if predictions["probability"] < 0.65:
            return signal_data
        
        # Determine signal type
        if predictions["direction"] == "up":
            signal_data["signal_type"] = "buy"
        elif predictions["direction"] == "down":
            signal_data["signal_type"] = "sell"
        else:
            return signal_data
        
        # Calculate base strength
        base_strength = predictions["probability"]
        
        # Adjust for consensus
        if insights["consensus"].get("market_view") == signal_data["signal_type"].replace("buy", "bullish").replace("sell", "bearish"):
            base_strength *= 1.2
            signal_data["reasons"].append("Strong agent consensus")
        
        # Adjust for market regime
        regime_adjustments = {
            "trending": 1.1,
            "volatile": 0.8,
            "crisis": 0.6,
            "euphoria": 0.7,
            "normal": 1.0,
        }
        regime_factor = regime_adjustments.get(market_regime["type"], 1.0)
        base_strength *= regime_factor
        
        # Technical confirmation
        tech_strength = insights["technical"].get("strength", 0)
        if tech_strength > 0.7:
            base_strength *= 1.1
            signal_data["reasons"].append("Strong technical signals")
        
        # Sentiment alignment
        sentiment_direction = insights["sentiment"].get("overall", {}).get("direction")
        expected_sentiment = "bullish" if signal_data["signal_type"] == "buy" else "bearish"
        if sentiment_direction == expected_sentiment:
            base_strength *= 1.1
            signal_data["reasons"].append("Aligned sentiment")
        
        signal_data["strength"] = min(base_strength, 1.0)
        
        return signal_data
    
    def _assess_signal_risk(self, signal_data: Dict[str, Any], 
                          insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of the trading signal."""
        risk_assessment = {
            "risk_level": "medium",
            "risk_score": 0.5,
            "concerns": [],
            "proceed": True,
        }
        
        # Check risk recommendations
        risk_recommendations = insights["risk"].get("recommendations", [])
        for rec in risk_recommendations:
            if rec.get("action") in ["reduce_position", "close_position"]:
                risk_assessment["risk_score"] += 0.2
                risk_assessment["concerns"].append(rec.get("reason"))
        
        # Portfolio risk
        portfolio_risk = insights["risk"].get("portfolio_risk", 0)
        if portfolio_risk > 0.02:  # 2% VaR
            risk_assessment["risk_score"] += 0.1
            risk_assessment["concerns"].append("Elevated portfolio risk")
        
        # Market risk
        market_risk = insights["risk"].get("market_risk")
        if market_risk == "high":
            risk_assessment["risk_score"] += 0.2
            risk_assessment["concerns"].append("High market risk environment")
        
        # Determine risk level
        if risk_assessment["risk_score"] > 0.7:
            risk_assessment["risk_level"] = "high"
            risk_assessment["proceed"] = signal_data["strength"] > 0.85
        elif risk_assessment["risk_score"] < 0.3:
            risk_assessment["risk_level"] = "low"
        
        return risk_assessment
    
    def _calculate_overall_confidence(self, insights: Dict[str, Any],
                                   predictions: Dict[str, Any],
                                   risk_assessment: Dict[str, Any]) -> float:
        """Calculate overall confidence in the signal."""
        # Base confidence from prediction probability
        confidence = predictions["probability"]
        
        # Adjust for consensus confidence
        consensus_confidence = insights["consensus"].get("confidence", 0.5)
        confidence = confidence * 0.6 + consensus_confidence * 0.4
        
        # Adjust for risk
        risk_factor = 1.0 - (risk_assessment["risk_score"] * 0.3)
        confidence *= risk_factor
        
        # Boost for multiple confirmations
        confirmations = 0
        if insights["technical"].get("strength", 0) > 0.7:
            confirmations += 1
        if insights["sentiment"].get("overall", {}).get("confidence", 0) > 0.7:
            confirmations += 1
        if insights["consensus"].get("confidence", 0) > 0.8:
            confirmations += 1
        
        if confirmations >= 2:
            confidence *= 1.1
        
        return min(confidence, 0.95)
    
    def _should_generate_signal(self, signal_data: Dict[str, Any],
                               risk_assessment: Dict[str, Any]) -> bool:
        """Determine if we should generate a trading signal."""
        # Check signal strength
        if signal_data.get("strength", 0) < self.confidence_threshold:
            return False
        
        # Check risk assessment
        if not risk_assessment.get("proceed", True):
            return False
        
        # Check daily signal limit
        today_signals = sum(1 for p in self.recent_predictions 
                          if (datetime.utcnow() - datetime.fromisoformat(p["timestamp"])).days == 0)
        if today_signals >= self.max_daily_signals:
            logger.warning("Daily signal limit reached")
            return False
        
        # Check cooldown
        if self.recent_predictions:
            last_signal_time = datetime.fromisoformat(self.recent_predictions[-1]["timestamp"])
            if (datetime.utcnow() - last_signal_time).seconds < self.cooldown_period:
                return False
        
        return True
    
    async def _create_buy_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        """Create a buy signal from analysis."""
        predictions = analysis.get("predictions", {})
        signal_data = analysis.get("signal_data", {})
        insights = analysis.get("insights", {})
        
        # Get current price from analysis
        current_price = analysis.get("current_price", 0)
        if not current_price:
            logger.warning("No current price available for signal")
            return None
        
        # Calculate entry price with slippage buffer
        entry_price = current_price * 1.001  # 0.1% slippage
        
        # Calculate targets based on prediction
        expected_move = abs(predictions.get("expected_move", 0.02))
        take_profit = entry_price * (1 + expected_move * 0.8)  # Conservative target
        
        # Dynamic stop loss based on volatility and risk
        volatility = analysis.get("market_data", {}).get("volatility", 0.2)
        stop_loss_pct = max(self.stop_loss_pct, volatility / 10)  # At least 1/10 of volatility
        stop_loss = entry_price * (1 - stop_loss_pct)
        
        # Create signal
        signal = Signal(
            symbol=analysis["symbol"],
            side=OrderSide.BUY,
            confidence=analysis["confidence"],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.name,
            metadata={
                "ai_prediction": predictions,
                "signal_reasons": signal_data.get("reasons", []),
                "market_regime": analysis.get("market_regime", {}).get("type"),
                "agent_consensus": insights["consensus"].get("market_view"),
                "risk_level": analysis["risk_assessment"].get("risk_level"),
            }
        )
        
        return signal
    
    async def _create_sell_signal(self, analysis: Dict[str, Any]) -> Optional[Signal]:
        """Create a sell signal from analysis."""
        predictions = analysis.get("predictions", {})
        signal_data = analysis.get("signal_data", {})
        insights = analysis.get("insights", {})
        
        # Get current price from analysis
        current_price = analysis.get("current_price", 0)
        if not current_price:
            logger.warning("No current price available for signal")
            return None
        
        # Calculate entry price with slippage buffer
        entry_price = current_price * 0.999  # 0.1% slippage
        
        # Calculate targets based on prediction
        expected_move = abs(predictions.get("expected_move", 0.02))
        take_profit = entry_price * (1 - expected_move * 0.8)  # Conservative target
        
        # Dynamic stop loss based on volatility and risk
        volatility = analysis.get("market_data", {}).get("volatility", 0.2)
        stop_loss_pct = max(self.stop_loss_pct, volatility / 10)
        stop_loss = entry_price * (1 + stop_loss_pct)
        
        # Create signal
        signal = Signal(
            symbol=analysis["symbol"],
            side=OrderSide.SELL,
            confidence=analysis["confidence"],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.name,
            metadata={
                "ai_prediction": predictions,
                "signal_reasons": signal_data.get("reasons", []),
                "market_regime": analysis.get("market_regime", {}).get("type"),
                "agent_consensus": insights["consensus"].get("market_view"),
                "risk_level": analysis["risk_assessment"].get("risk_level"),
            }
        )
        
        return signal
    
    def _apply_signal_filters(self, signals: List[Signal], 
                            analysis: Dict[str, Any]) -> List[Signal]:
        """Apply final filters to signals."""
        filtered = []
        
        for signal in signals:
            # Volume confirmation filter
            if self.require_volume_confirmation:
                volume_profile = analysis.get("market_data", {}).get("volume_profile", {})
                if volume_profile.get("current_vs_average", 1) < 0.8:
                    logger.debug("Signal filtered: low volume")
                    continue
            
            # Contradictory signals filter
            if self.filter_contradictory:
                insights = analysis.get("insights", {})
                tech_view = insights["technical"].get("trend")
                sentiment_view = insights["sentiment"].get("overall", {}).get("direction")
                
                # Check for major contradictions
                if (signal.side == OrderSide.BUY and 
                    (tech_view == "downtrend" or sentiment_view == "bearish")):
                    logger.debug("Signal filtered: contradictory indicators")
                    continue
                elif (signal.side == OrderSide.SELL and 
                      (tech_view == "uptrend" or sentiment_view == "bullish")):
                    logger.debug("Signal filtered: contradictory indicators")
                    continue
            
            filtered.append(signal)
        
        return filtered
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile metrics."""
        if len(df) < 20:
            return {}
        
        volume = df["volume"]
        price = df["close"]
        
        # Current vs average volume
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        
        # Volume-weighted average price (VWAP)
        vwap = (price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        # Price-volume correlation
        price_returns = price.pct_change()
        volume_changes = volume.pct_change()
        
        if len(price_returns.dropna()) > 10:
            correlation = price_returns.corr(volume_changes)
        else:
            correlation = 0
        
        return {
            "current_vs_average": current_volume / avg_volume if avg_volume > 0 else 1,
            "vwap": vwap.iloc[-1],
            "price_volume_correlation": correlation,
            "volume_trend": "increasing" if volume.iloc[-5:].mean() > avg_volume else "decreasing",
        }
    
    def _get_market_hours(self) -> str:
        """Get current market hours status."""
        current_hour = datetime.utcnow().hour
        
        # Simplified market hours (UTC)
        if 13 <= current_hour <= 20:  # US market hours
            return "us_active"
        elif 7 <= current_hour <= 15:  # European hours
            return "europe_active"
        elif 23 <= current_hour or current_hour <= 7:  # Asian hours
            return "asia_active"
        else:
            return "transitional"
    
    def _store_prediction(self, analysis: Dict[str, Any]):
        """Store prediction for future learning."""
        prediction_record = {
            "timestamp": analysis["timestamp"],
            "symbol": analysis["symbol"],
            "prediction": analysis["predictions"],
            "confidence": analysis["confidence"],
            "signal_generated": analysis.get("signal_data", {}).get("signal_type") != "none",
        }
        
        self.recent_predictions.append(prediction_record)
        
        # Keep only recent predictions (last 100)
        if len(self.recent_predictions) > 100:
            self.recent_predictions = self.recent_predictions[-100:]
    
    def _validate_signal_custom(self, signal: Signal) -> bool:
        """Custom validation for AI strategy signals."""
        metadata = signal.metadata or {}
        
        # Ensure we have AI prediction
        if "ai_prediction" not in metadata:
            return False
        
        # Check risk level
        risk_level = metadata.get("risk_level")
        if risk_level == "high" and signal.confidence < 0.85:
            return False
        
        # Ensure we have reasoning
        if not metadata.get("signal_reasons"):
            return False
        
        return True
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get AI model performance metrics."""
        metrics = self.model_performance.copy()
        
        # Add recent prediction statistics
        if self.recent_predictions:
            recent_confidence = [p["confidence"] for p in self.recent_predictions]
            metrics["avg_confidence"] = np.mean(recent_confidence)
            metrics["signal_rate"] = sum(1 for p in self.recent_predictions 
                                       if p["signal_generated"]) / len(self.recent_predictions)
        
        # Add agent status
        metrics["agent_status"] = self.orchestrator.get_agent_status()
        
        return metrics