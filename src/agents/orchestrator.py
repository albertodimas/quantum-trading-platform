"""
Agent Orchestrator to coordinate multiple AI agents.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
from collections import defaultdict

from src.agents.base import BaseAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.risk import RiskManagementAgent
from src.core.logging import get_logger
from src.trading.models import Signal, OrderSide

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple AI agents to make coordinated trading decisions.
    
    Manages agent lifecycle, data flow, and decision aggregation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent orchestrator."""
        self.config = config or {}
        self.agents: Dict[str, BaseAgent] = {}
        self._running = False
        
        # Decision weights
        self.agent_weights = {
            "technical": self.config.get("technical_weight", 0.4),
            "sentiment": self.config.get("sentiment_weight", 0.3),
            "risk": self.config.get("risk_weight", 0.3),
        }
        
        # Consensus parameters
        self.min_agent_agreement = self.config.get("min_agent_agreement", 2)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all agents."""
        # Technical Analysis Agent
        self.agents["technical"] = TechnicalAnalysisAgent(
            self.config.get("technical_config", {})
        )
        
        # Sentiment Analysis Agent
        self.agents["sentiment"] = SentimentAnalysisAgent(
            self.config.get("sentiment_config", {})
        )
        
        # Risk Management Agent
        self.agents["risk"] = RiskManagementAgent(
            self.config.get("risk_config", {})
        )
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def start(self):
        """Start all agents."""
        self._running = True
        
        # Start agents concurrently
        await asyncio.gather(
            *[agent.start() for agent in self.agents.values()]
        )
        
        logger.info("Agent orchestrator started")
    
    async def stop(self):
        """Stop all agents."""
        self._running = False
        
        # Stop agents concurrently
        await asyncio.gather(
            *[agent.stop() for agent in self.agents.values()]
        )
        
        logger.info("Agent orchestrator stopped")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate market analysis across all agents.
        
        Args:
            market_data: Comprehensive market data for analysis
            
        Returns:
            Aggregated analysis from all agents
        """
        if not self._running:
            logger.warning("Orchestrator not running")
            return {"error": "Orchestrator not started"}
        
        # Prepare data for each agent
        technical_data = self._prepare_technical_data(market_data)
        sentiment_data = self._prepare_sentiment_data(market_data)
        risk_data = self._prepare_risk_data(market_data)
        
        # Run analyses concurrently
        analyses = await asyncio.gather(
            self.agents["technical"].analyze(technical_data),
            self.agents["sentiment"].analyze(sentiment_data),
            self.agents["risk"].analyze(risk_data),
            return_exceptions=True
        )
        
        # Handle results
        results = {}
        for agent_name, analysis in zip(["technical", "sentiment", "risk"], analyses):
            if isinstance(analysis, Exception):
                logger.error(f"Agent {agent_name} analysis failed: {analysis}")
                results[agent_name] = {"error": str(analysis)}
            else:
                results[agent_name] = analysis
        
        # Aggregate insights
        aggregated = self._aggregate_analyses(results)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "individual_analyses": results,
            "aggregated": aggregated,
        }
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on multi-agent analysis.
        
        Args:
            analysis: Aggregated analysis from all agents
            
        Returns:
            List of trading signals
        """
        if "error" in analysis:
            return []
        
        individual_analyses = analysis.get("individual_analyses", {})
        
        # Get signals from each agent
        all_signals = []
        
        for agent_name, agent_analysis in individual_analyses.items():
            if "error" not in agent_analysis:
                agent = self.agents[agent_name]
                signals = await agent.get_signals(agent_analysis)
                
                # Add agent source to signals
                for signal in signals:
                    signal["agent"] = agent_name
                
                all_signals.extend(signals)
        
        # Aggregate and filter signals
        consensus_signals = self._build_consensus_signals(all_signals)
        
        # Apply risk management overlay
        final_signals = await self._apply_risk_management(consensus_signals, individual_analyses)
        
        # Convert to Signal objects
        trading_signals = []
        for signal_data in final_signals:
            trading_signal = self._create_trading_signal(signal_data)
            if trading_signal:
                trading_signals.append(trading_signal)
        
        return trading_signals
    
    def _prepare_technical_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for technical analysis agent."""
        return {
            "symbol": market_data.get("symbol"),
            "ohlcv": market_data.get("ohlcv", []),
            "timeframe": market_data.get("timeframe", "1h"),
        }
    
    def _prepare_sentiment_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for sentiment analysis agent."""
        return {
            "symbol": market_data.get("symbol"),
            "news": market_data.get("news", []),
            "social": market_data.get("social", []),
        }
    
    def _prepare_risk_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for risk management agent."""
        return {
            "portfolio": market_data.get("portfolio", {}),
            "market_data": {
                market_data.get("symbol"): {
                    "price": market_data.get("current_price"),
                    "volatility": market_data.get("volatility", 0.2),
                    "returns": market_data.get("returns", []),
                }
            },
            "proposed_trade": market_data.get("proposed_trade"),
        }
    
    def _aggregate_analyses(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate insights from multiple agents."""
        aggregated = {
            "market_consensus": "neutral",
            "confidence": 0,
            "key_factors": [],
            "risk_assessment": "medium",
        }
        
        # Count bullish/bearish signals
        sentiments = defaultdict(int)
        confidences = []
        
        # Technical analysis
        if "technical" in analyses and "error" not in analyses["technical"]:
            tech = analyses["technical"]
            market_structure = tech.get("market_structure", {})
            
            if market_structure.get("trend") == "uptrend":
                sentiments["bullish"] += 1
            elif market_structure.get("trend") == "downtrend":
                sentiments["bearish"] += 1
            else:
                sentiments["neutral"] += 1
            
            confidences.append(tech.get("signal_strength", 0))
            
            # Add key factors
            if tech.get("patterns", {}).get("bullish_patterns"):
                aggregated["key_factors"].append("Bullish technical patterns detected")
            if tech.get("patterns", {}).get("bearish_patterns"):
                aggregated["key_factors"].append("Bearish technical patterns detected")
        
        # Sentiment analysis
        if "sentiment" in analyses and "error" not in analyses["sentiment"]:
            sent = analyses["sentiment"]
            overall_sent = sent.get("overall_sentiment", {})
            
            if overall_sent.get("direction") == "bullish":
                sentiments["bullish"] += 1
            elif overall_sent.get("direction") == "bearish":
                sentiments["bearish"] += 1
            else:
                sentiments["neutral"] += 1
            
            confidences.append(sent.get("confidence", 0))
            
            # Add key factors
            if overall_sent.get("score", 0) > 0.5:
                aggregated["key_factors"].append("Strong positive market sentiment")
            elif overall_sent.get("score", 0) < -0.5:
                aggregated["key_factors"].append("Strong negative market sentiment")
        
        # Risk analysis
        if "risk" in analyses and "error" not in analyses["risk"]:
            risk = analyses["risk"]
            risk_score = risk.get("risk_score", 0.5)
            
            if risk_score > 0.7:
                aggregated["risk_assessment"] = "high"
                aggregated["key_factors"].append("High risk environment detected")
            elif risk_score < 0.3:
                aggregated["risk_assessment"] = "low"
            
            # Check for violations
            if risk.get("risk_limits", {}).get("violations"):
                aggregated["key_factors"].append("Risk limit violations detected")
        
        # Determine consensus
        max_sentiment = max(sentiments.items(), key=lambda x: x[1])[0] if sentiments else "neutral"
        aggregated["market_consensus"] = max_sentiment
        
        # Calculate average confidence
        if confidences:
            aggregated["confidence"] = sum(confidences) / len(confidences)
        
        return aggregated
    
    def _build_consensus_signals(self, all_signals: List[Dict]) -> List[Dict]:
        """Build consensus signals from individual agent signals."""
        # Group signals by symbol and action
        signal_groups = defaultdict(list)
        
        for signal in all_signals:
            key = (signal.get("symbol"), signal.get("action"))
            signal_groups[key].append(signal)
        
        # Build consensus
        consensus_signals = []
        
        for (symbol, action), signals in signal_groups.items():
            # Check if enough agents agree
            if len(signals) >= self.min_agent_agreement:
                # Calculate weighted confidence
                total_weight = 0
                weighted_confidence = 0
                reasons = []
                
                for signal in signals:
                    agent = signal.get("agent")
                    weight = self.agent_weights.get(agent, 0.33)
                    confidence = signal.get("confidence", 0)
                    
                    weighted_confidence += weight * confidence
                    total_weight += weight
                    
                    # Collect reasons
                    if "reasons" in signal:
                        reasons.extend(signal["reasons"])
                    elif "reason" in signal:
                        reasons.append(signal["reason"])
                
                final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
                
                # Only include if confidence is high enough
                if final_confidence >= self.min_confidence:
                    consensus_signals.append({
                        "symbol": symbol,
                        "action": action,
                        "confidence": final_confidence,
                        "reasons": list(set(reasons)),  # Unique reasons
                        "agent_count": len(signals),
                        "contributing_agents": [s["agent"] for s in signals],
                    })
        
        return consensus_signals
    
    async def _apply_risk_management(self, signals: List[Dict], 
                                   analyses: Dict[str, Dict]) -> List[Dict]:
        """Apply risk management overlay to signals."""
        risk_analysis = analyses.get("risk", {})
        
        if "error" in risk_analysis:
            logger.warning("Risk analysis not available, proceeding with caution")
            # Reduce confidence on all signals
            for signal in signals:
                signal["confidence"] *= 0.7
            return signals
        
        # Get risk recommendations
        recommendations = risk_analysis.get("recommendations", [])
        risk_score = risk_analysis.get("risk_score", 0.5)
        
        # Filter signals based on risk
        filtered_signals = []
        
        for signal in signals:
            # Check if any risk recommendation contradicts the signal
            contradict = False
            
            for rec in recommendations:
                if (rec.get("action") in ["reduce_position", "close_position"] and 
                    signal["action"] == "buy"):
                    contradict = True
                    break
            
            if not contradict:
                # Adjust confidence based on risk score
                if risk_score > 0.7:  # High risk
                    signal["confidence"] *= 0.5
                elif risk_score > 0.5:  # Medium risk
                    signal["confidence"] *= 0.8
                
                # Add risk warnings
                if risk_score > 0.5:
                    signal["risk_warning"] = f"Elevated risk environment (score: {risk_score:.2f})"
                
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _create_trading_signal(self, signal_data: Dict) -> Optional[Signal]:
        """Create a trading Signal object from consensus data."""
        try:
            # Map action to OrderSide
            side_map = {
                "buy": OrderSide.BUY,
                "sell": OrderSide.SELL,
            }
            
            side = side_map.get(signal_data["action"])
            if not side:
                logger.warning(f"Unknown action: {signal_data['action']}")
                return None
            
            # Create metadata
            metadata = {
                "reasons": signal_data.get("reasons", []),
                "agent_count": signal_data.get("agent_count", 0),
                "contributing_agents": signal_data.get("contributing_agents", []),
                "risk_warning": signal_data.get("risk_warning"),
            }
            
            # Create signal
            signal = Signal(
                symbol=signal_data["symbol"],
                side=side,
                confidence=signal_data["confidence"],
                strategy="multi_agent_consensus",
                metadata=metadata,
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to create trading signal: {e}")
            return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {
            "orchestrator_running": self._running,
            "agents": {},
        }
        
        for name, agent in self.agents.items():
            status["agents"][name] = {
                "running": agent.is_running(),
                "metrics": agent.get_metrics(),
            }
        
        return status
    
    def update_agent_weights(self, weights: Dict[str, float]):
        """Update agent decision weights."""
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            logger.error("Invalid weights: sum is zero")
            return
        
        # Normalize weights
        for agent, weight in weights.items():
            if agent in self.agent_weights:
                self.agent_weights[agent] = weight / total_weight
        
        logger.info(f"Updated agent weights: {self.agent_weights}")
    
    def set_consensus_threshold(self, min_agents: int, min_confidence: float):
        """Update consensus thresholds."""
        self.min_agent_agreement = max(1, min(len(self.agents), min_agents))
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        
        logger.info(f"Updated consensus thresholds - agents: {self.min_agent_agreement}, "
                   f"confidence: {self.min_confidence}")
    
    async def backtest_agents(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Backtest agent performance on historical data."""
        results = {
            "total_signals": 0,
            "accurate_signals": 0,
            "agent_performance": defaultdict(lambda: {"total": 0, "accurate": 0}),
        }
        
        for data_point in historical_data:
            # Get analysis
            analysis = await self.analyze_market(data_point)
            
            # Generate signals
            signals = await self.generate_signals(analysis)
            
            # Check accuracy (simplified - would need actual outcome data)
            for signal in signals:
                results["total_signals"] += 1
                
                # Track by agent
                for agent in signal.metadata.get("contributing_agents", []):
                    results["agent_performance"][agent]["total"] += 1
        
        return results