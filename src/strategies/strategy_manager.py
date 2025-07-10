"""
Strategy Manager for coordinating multiple trading strategies.
"""

from typing import Any, Dict, List, Optional, Type
import asyncio
from datetime import datetime
from collections import defaultdict

from src.strategies.base import BaseStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.arbitrage import ArbitrageStrategy
from src.strategies.ai_strategy import AIStrategy
from src.core.logging import get_logger
from src.trading.models import Signal

logger = get_logger(__name__)


class StrategyManager:
    """
    Manages multiple trading strategies and coordinates their execution.
    
    Features:
    - Strategy lifecycle management
    - Signal aggregation and conflict resolution
    - Performance tracking
    - Dynamic strategy allocation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy manager."""
        self.config = config or {}
        self.strategies: Dict[str, BaseStrategy] = {}
        self._running = False
        
        # Signal management
        self.signal_history = []
        self.active_signals = {}
        
        # Performance tracking
        self.strategy_performance = defaultdict(lambda: {
            "signals": 0,
            "successful": 0,
            "failed": 0,
            "total_pnl": 0,
        })
        
        # Configuration
        self.max_concurrent_signals = self.config.get("max_concurrent_signals", 10)
        self.signal_timeout = self.config.get("signal_timeout", 3600)  # 1 hour
        self.conflict_resolution = self.config.get("conflict_resolution", "highest_confidence")
        
        # Strategy allocation
        self.allocation_mode = self.config.get("allocation_mode", "equal")  # equal, performance, dynamic
        self.strategy_weights = {}
        
        # Available strategy types
        self.available_strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "arbitrage": ArbitrageStrategy,
            "ai": AIStrategy,
        }
    
    async def initialize_strategies(self, strategy_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize strategies based on configuration.
        
        Args:
            strategy_configs: Dictionary of strategy names and their configurations
        """
        for name, config in strategy_configs.items():
            if name in self.available_strategies:
                strategy_class = self.available_strategies[name]
                strategy = strategy_class(config)
                self.strategies[name] = strategy
                logger.info(f"Initialized strategy: {name}")
            else:
                logger.warning(f"Unknown strategy type: {name}")
        
        # Initialize weights
        self._update_strategy_weights()
    
    async def start(self):
        """Start all strategies."""
        self._running = True
        
        # Start strategies concurrently
        start_tasks = []
        for name, strategy in self.strategies.items():
            if isinstance(strategy, AIStrategy):
                # AI strategy needs special handling for agent startup
                start_tasks.append(strategy.start())
            else:
                strategy.start()
        
        if start_tasks:
            await asyncio.gather(*start_tasks)
        
        logger.info("Strategy manager started with {} strategies".format(len(self.strategies)))
    
    async def stop(self):
        """Stop all strategies."""
        self._running = False
        
        # Stop strategies concurrently
        stop_tasks = []
        for name, strategy in self.strategies.items():
            if isinstance(strategy, AIStrategy):
                stop_tasks.append(strategy.stop())
            else:
                strategy.stop()
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
        
        logger.info("Strategy manager stopped")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run market analysis across all active strategies.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Aggregated analysis results
        """
        if not self._running:
            return {"error": "Strategy manager not running"}
        
        # Get active strategies based on weights
        active_strategies = self._get_active_strategies()
        
        # Run analyses concurrently
        analysis_tasks = {}
        for name, strategy in active_strategies.items():
            analysis_tasks[name] = strategy.analyze(market_data)
        
        # Gather results
        analysis_results = {}
        for name, task in analysis_tasks.items():
            try:
                result = await task
                analysis_results[name] = result
            except Exception as e:
                logger.error(f"Strategy {name} analysis failed: {e}")
                analysis_results[name] = {"error": str(e)}
        
        # Aggregate insights
        aggregated = self._aggregate_analyses(analysis_results)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": market_data.get("symbol"),
            "strategy_analyses": analysis_results,
            "aggregated": aggregated,
            "active_strategies": list(active_strategies.keys()),
        }
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals from strategy analyses.
        
        Args:
            analysis: Aggregated analysis results
            
        Returns:
            List of trading signals
        """
        all_signals = []
        strategy_analyses = analysis.get("strategy_analyses", {})
        
        # Generate signals from each strategy
        signal_tasks = {}
        for name, strategy_analysis in strategy_analyses.items():
            if "error" not in strategy_analysis and name in self.strategies:
                strategy = self.strategies[name]
                signal_tasks[name] = strategy.generate_signals(strategy_analysis)
        
        # Gather signals
        for name, task in signal_tasks.items():
            try:
                signals = await task
                # Tag signals with strategy name
                for signal in signals:
                    signal.metadata = signal.metadata or {}
                    signal.metadata["source_strategy"] = name
                    all_signals.append(signal)
                    
                # Track signal generation
                self.strategy_performance[name]["signals"] += len(signals)
                
            except Exception as e:
                logger.error(f"Strategy {name} signal generation failed: {e}")
        
        # Resolve conflicts and filter signals
        filtered_signals = self._filter_and_resolve_signals(all_signals)
        
        # Apply position limits
        final_signals = self._apply_position_limits(filtered_signals)
        
        # Store signals
        self._store_signals(final_signals)
        
        return final_signals
    
    def _get_active_strategies(self) -> Dict[str, BaseStrategy]:
        """Get active strategies based on weights and allocation."""
        active = {}
        
        for name, strategy in self.strategies.items():
            if strategy.is_active() and self.strategy_weights.get(name, 0) > 0:
                active[name] = strategy
        
        return active
    
    def _update_strategy_weights(self):
        """Update strategy allocation weights."""
        if self.allocation_mode == "equal":
            # Equal weight to all strategies
            weight = 1.0 / len(self.strategies) if self.strategies else 0
            self.strategy_weights = {name: weight for name in self.strategies}
            
        elif self.allocation_mode == "performance":
            # Weight based on historical performance
            total_score = 0
            scores = {}
            
            for name in self.strategies:
                perf = self.strategy_performance[name]
                # Simple scoring: win rate * avg PnL
                win_rate = perf["successful"] / perf["signals"] if perf["signals"] > 0 else 0.5
                avg_pnl = perf["total_pnl"] / perf["signals"] if perf["signals"] > 0 else 0
                
                score = (win_rate * 0.7 + min(avg_pnl, 1) * 0.3)
                scores[name] = max(score, 0.1)  # Minimum weight
                total_score += scores[name]
            
            # Normalize
            if total_score > 0:
                self.strategy_weights = {name: score / total_score 
                                       for name, score in scores.items()}
            else:
                # Fallback to equal weights
                weight = 1.0 / len(self.strategies)
                self.strategy_weights = {name: weight for name in self.strategies}
                
        elif self.allocation_mode == "dynamic":
            # Dynamic allocation based on market conditions
            # This is a simplified version - could be more sophisticated
            self.strategy_weights = self._calculate_dynamic_weights()
    
    def _calculate_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic strategy weights based on market conditions."""
        weights = {}
        
        # Default weights
        base_weights = {
            "momentum": 0.3,
            "mean_reversion": 0.3,
            "arbitrage": 0.2,
            "ai": 0.2,
        }
        
        # Adjust based on recent market conditions
        # This is simplified - in practice, would analyze volatility, trends, etc.
        recent_volatility = self._estimate_recent_volatility()
        
        if recent_volatility > 0.3:  # High volatility
            # Favor mean reversion and arbitrage
            weights["mean_reversion"] = 0.4
            weights["arbitrage"] = 0.3
            weights["momentum"] = 0.2
            weights["ai"] = 0.1
        elif recent_volatility < 0.1:  # Low volatility
            # Favor momentum
            weights["momentum"] = 0.4
            weights["mean_reversion"] = 0.2
            weights["arbitrage"] = 0.2
            weights["ai"] = 0.2
        else:
            weights = base_weights
        
        # Only include weights for available strategies
        return {name: weights.get(name, 0.25) for name in self.strategies}
    
    def _estimate_recent_volatility(self) -> float:
        """Estimate recent market volatility from signals."""
        # Simplified estimation based on recent signal frequency
        recent_signals = [s for s in self.signal_history 
                         if (datetime.utcnow() - s["timestamp"]).seconds < 3600]
        
        # High signal frequency might indicate high volatility
        signal_rate = len(recent_signals) / max(len(self.strategies), 1)
        
        return min(signal_rate * 0.1, 0.5)  # Cap at 0.5
    
    def _aggregate_analyses(self, analysis_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate insights from multiple strategy analyses."""
        aggregated = {
            "market_view": "neutral",
            "confidence": 0,
            "signal_count": 0,
            "opportunities": [],
            "warnings": [],
        }
        
        # Collect views and confidences
        views = defaultdict(float)
        total_confidence = 0
        opportunity_count = 0
        
        for name, analysis in analysis_results.items():
            if "error" in analysis:
                continue
            
            weight = self.strategy_weights.get(name, 0)
            
            # Extract market view (strategy-specific)
            if name == "momentum":
                if analysis.get("signal_direction") == "bullish":
                    views["bullish"] += weight
                elif analysis.get("signal_direction") == "bearish":
                    views["bearish"] += weight
                    
            elif name == "mean_reversion":
                entry_signals = analysis.get("entry_signals", {})
                if entry_signals.get("long_signal"):
                    views["bullish"] += weight * 0.8
                elif entry_signals.get("short_signal"):
                    views["bearish"] += weight * 0.8
                    
            elif name == "ai":
                consensus = analysis.get("insights", {}).get("consensus", {}).get("market_view")
                if consensus == "bullish":
                    views["bullish"] += weight
                elif consensus == "bearish":
                    views["bearish"] += weight
            
            # Aggregate confidence
            confidence = analysis.get("confidence", 0)
            total_confidence += confidence * weight
            
            # Count opportunities
            if name == "arbitrage":
                opportunity_count += analysis.get("viable_opportunities", 0)
        
        # Determine overall market view
        if views["bullish"] > views["bearish"] * 1.2:
            aggregated["market_view"] = "bullish"
        elif views["bearish"] > views["bullish"] * 1.2:
            aggregated["market_view"] = "bearish"
        
        aggregated["confidence"] = total_confidence
        aggregated["opportunity_count"] = opportunity_count
        
        return aggregated
    
    def _filter_and_resolve_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter and resolve conflicts between signals."""
        if not signals:
            return []
        
        # Group signals by symbol
        symbol_signals = defaultdict(list)
        for signal in signals:
            symbol_signals[signal.symbol].append(signal)
        
        resolved_signals = []
        
        for symbol, symbol_group in symbol_signals.items():
            if len(symbol_group) == 1:
                resolved_signals.append(symbol_group[0])
            else:
                # Resolve conflicts
                resolved = self._resolve_signal_conflict(symbol_group)
                if resolved:
                    resolved_signals.append(resolved)
        
        return resolved_signals
    
    def _resolve_signal_conflict(self, signals: List[Signal]) -> Optional[Signal]:
        """Resolve conflicts between multiple signals for the same symbol."""
        if not signals:
            return None
        
        # Check for contradictory signals (buy vs sell)
        sides = {signal.side for signal in signals}
        if len(sides) > 1:
            # Contradictory signals
            if self.conflict_resolution == "highest_confidence":
                # Choose highest confidence signal
                return max(signals, key=lambda s: s.confidence)
            elif self.conflict_resolution == "cancel":
                # Cancel on contradiction
                logger.warning(f"Contradictory signals for {signals[0].symbol}, cancelling")
                return None
            elif self.conflict_resolution == "first":
                # Use first signal
                return signals[0]
        else:
            # Same direction signals - combine or choose best
            if self.conflict_resolution == "highest_confidence":
                return max(signals, key=lambda s: s.confidence)
            elif self.conflict_resolution == "combine":
                # Create combined signal
                combined = signals[0]
                combined.confidence = max(s.confidence for s in signals)
                combined.metadata = combined.metadata or {}
                combined.metadata["combined_from"] = [
                    s.metadata.get("source_strategy") for s in signals
                ]
                return combined
            else:
                return signals[0]
    
    def _apply_position_limits(self, signals: List[Signal]) -> List[Signal]:
        """Apply position and risk limits to signals."""
        # Count current active signals
        active_count = len([s for s in self.active_signals.values() 
                          if (datetime.utcnow() - s["timestamp"]).seconds < self.signal_timeout])
        
        available_slots = self.max_concurrent_signals - active_count
        
        if available_slots <= 0:
            logger.warning("Maximum concurrent signals reached")
            return []
        
        # Sort by confidence and take top signals
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
        
        return sorted_signals[:available_slots]
    
    def _store_signals(self, signals: List[Signal]):
        """Store signals for tracking."""
        for signal in signals:
            signal_record = {
                "signal": signal,
                "timestamp": datetime.utcnow(),
                "status": "active",
            }
            
            # Add to history
            self.signal_history.append(signal_record)
            
            # Add to active signals
            signal_id = f"{signal.symbol}_{datetime.utcnow().timestamp()}"
            self.active_signals[signal_id] = signal_record
        
        # Clean up old signals
        self._cleanup_old_signals()
    
    def _cleanup_old_signals(self):
        """Remove old signals from active tracking."""
        current_time = datetime.utcnow()
        
        # Remove expired active signals
        expired = []
        for signal_id, record in self.active_signals.items():
            if (current_time - record["timestamp"]).seconds > self.signal_timeout:
                expired.append(signal_id)
        
        for signal_id in expired:
            del self.active_signals[signal_id]
        
        # Limit history size
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def update_signal_result(self, signal_id: str, result: Dict[str, Any]):
        """
        Update the result of a signal execution.
        
        Args:
            signal_id: Signal identifier
            result: Execution result including PnL
        """
        if signal_id in self.active_signals:
            record = self.active_signals[signal_id]
            record["status"] = "completed"
            record["result"] = result
            
            # Update strategy performance
            source_strategy = record["signal"].metadata.get("source_strategy")
            if source_strategy:
                perf = self.strategy_performance[source_strategy]
                
                pnl = result.get("pnl", 0)
                perf["total_pnl"] += pnl
                
                if pnl > 0:
                    perf["successful"] += 1
                else:
                    perf["failed"] += 1
            
            # Remove from active
            del self.active_signals[signal_id]
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies."""
        status = {
            "running": self._running,
            "active_strategies": list(self._get_active_strategies().keys()),
            "total_strategies": len(self.strategies),
            "strategy_weights": self.strategy_weights,
            "active_signals": len(self.active_signals),
            "allocation_mode": self.allocation_mode,
        }
        
        # Add individual strategy status
        strategy_details = {}
        for name, strategy in self.strategies.items():
            strategy_details[name] = {
                "active": strategy.is_active(),
                "config": strategy.get_config(),
                "performance": strategy.get_performance_metrics(),
                "weight": self.strategy_weights.get(name, 0),
            }
        
        status["strategies"] = strategy_details
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all strategies."""
        summary = {
            "total_signals": sum(p["signals"] for p in self.strategy_performance.values()),
            "total_pnl": sum(p["total_pnl"] for p in self.strategy_performance.values()),
            "strategy_performance": dict(self.strategy_performance),
        }
        
        # Calculate aggregate metrics
        total_successful = sum(p["successful"] for p in self.strategy_performance.values())
        total_trades = sum(p["successful"] + p["failed"] for p in self.strategy_performance.values())
        
        if total_trades > 0:
            summary["overall_win_rate"] = total_successful / total_trades
            summary["average_pnl"] = summary["total_pnl"] / total_trades
        else:
            summary["overall_win_rate"] = 0
            summary["average_pnl"] = 0
        
        # Best performing strategy
        if self.strategy_performance:
            best_strategy = max(self.strategy_performance.items(), 
                              key=lambda x: x[1]["total_pnl"])
            summary["best_strategy"] = {
                "name": best_strategy[0],
                "pnl": best_strategy[1]["total_pnl"],
            }
        
        return summary
    
    def adjust_strategy_allocation(self, new_weights: Optional[Dict[str, float]] = None):
        """
        Adjust strategy allocation weights.
        
        Args:
            new_weights: Optional manual weight overrides
        """
        if new_weights:
            # Manual weight adjustment
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                # Normalize weights
                self.strategy_weights = {
                    name: weight / total_weight 
                    for name, weight in new_weights.items()
                    if name in self.strategies
                }
                logger.info(f"Strategy weights manually updated: {self.strategy_weights}")
        else:
            # Automatic rebalancing
            self._update_strategy_weights()
            logger.info(f"Strategy weights rebalanced: {self.strategy_weights}")
    
    async def add_strategy(self, name: str, strategy_type: str, config: Dict[str, Any]):
        """
        Add a new strategy dynamically.
        
        Args:
            name: Strategy instance name
            strategy_type: Type of strategy (momentum, mean_reversion, etc.)
            config: Strategy configuration
        """
        if strategy_type not in self.available_strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        if name in self.strategies:
            raise ValueError(f"Strategy {name} already exists")
        
        # Create and initialize strategy
        strategy_class = self.available_strategies[strategy_type]
        strategy = strategy_class(config)
        
        # Start if manager is running
        if self._running:
            if isinstance(strategy, AIStrategy):
                await strategy.start()
            else:
                strategy.start()
        
        self.strategies[name] = strategy
        
        # Update weights
        self._update_strategy_weights()
        
        logger.info(f"Added strategy: {name} (type: {strategy_type})")
    
    async def remove_strategy(self, name: str):
        """
        Remove a strategy dynamically.
        
        Args:
            name: Strategy instance name
        """
        if name not in self.strategies:
            raise ValueError(f"Strategy {name} not found")
        
        strategy = self.strategies[name]
        
        # Stop strategy
        if strategy.is_active():
            if isinstance(strategy, AIStrategy):
                await strategy.stop()
            else:
                strategy.stop()
        
        # Remove from tracking
        del self.strategies[name]
        if name in self.strategy_weights:
            del self.strategy_weights[name]
        
        # Update weights
        self._update_strategy_weights()
        
        logger.info(f"Removed strategy: {name}")