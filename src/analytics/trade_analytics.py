"""
Trade Analytics Module

Provides detailed trade analysis including win/loss patterns, holding periods,
entry/exit quality, and trade clustering analysis.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
from enum import Enum

from ..core.logger import get_logger
from ..models.trading import Trade, OrderSide

logger = get_logger(__name__)


class TradeOutcome(Enum):
    """Trade outcome classification"""
    BIG_WIN = "big_win"         # > 2% profit
    WIN = "win"                  # 0-2% profit  
    BREAKEVEN = "breakeven"      # -0.5% to 0.5%
    LOSS = "loss"               # -2% to -0.5%
    BIG_LOSS = "big_loss"       # < -2% loss


class EntryQuality(Enum):
    """Entry quality classification"""
    EXCELLENT = "excellent"      # Near perfect timing
    GOOD = "good"               # Good timing
    AVERAGE = "average"         # Average timing
    POOR = "poor"               # Poor timing
    TERRIBLE = "terrible"       # Very poor timing


@dataclass
class TradeStatistics:
    """Comprehensive trade statistics"""
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    # P&L stats
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    
    # Duration stats
    avg_duration_hours: float = 0.0
    avg_winning_duration: float = 0.0
    avg_losing_duration: float = 0.0
    longest_trade_hours: float = 0.0
    shortest_trade_hours: float = 0.0
    
    # Streak stats
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0


class TradeAnalytics:
    """Analyzes individual trade performance and patterns"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    async def analyze_trades(
        self,
        trades: List[Trade],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive trade analysis
        
        Args:
            trades: List of trades to analyze
            market_data: Optional market data for context
            
        Returns:
            Comprehensive trade analysis
        """
        if not trades:
            return {}
        
        # Calculate basic statistics
        stats = self._calculate_trade_statistics(trades)
        
        # Analyze trade outcomes
        outcome_analysis = self._analyze_trade_outcomes(trades)
        
        # Analyze entry/exit quality
        entry_exit_analysis = await self._analyze_entry_exit_quality(trades, market_data)
        
        # Analyze holding periods
        holding_period_analysis = self._analyze_holding_periods(trades)
        
        # Analyze trade clustering
        clustering_analysis = self._analyze_trade_clustering(trades)
        
        # Analyze by time periods
        time_analysis = self._analyze_by_time_periods(trades)
        
        # Analyze by market conditions
        market_condition_analysis = await self._analyze_by_market_conditions(trades, market_data)
        
        # Analyze consecutive trades
        consecutive_analysis = self._analyze_consecutive_trades(trades)
        
        # Generate insights
        insights = self._generate_trade_insights(
            stats, outcome_analysis, entry_exit_analysis, holding_period_analysis
        )
        
        return {
            "statistics": stats.__dict__,
            "outcome_analysis": outcome_analysis,
            "entry_exit_quality": entry_exit_analysis,
            "holding_periods": holding_period_analysis,
            "clustering": clustering_analysis,
            "time_analysis": time_analysis,
            "market_conditions": market_condition_analysis,
            "consecutive_trades": consecutive_analysis,
            "insights": insights
        }
    
    def _calculate_trade_statistics(self, trades: List[Trade]) -> TradeStatistics:
        """Calculate comprehensive trade statistics"""
        stats = TradeStatistics()
        
        if not trades:
            return stats
        
        # Basic counts
        stats.total_trades = len(trades)
        stats.winning_trades = sum(1 for t in trades if t.realized_pnl > 0)
        stats.losing_trades = sum(1 for t in trades if t.realized_pnl < 0)
        stats.breakeven_trades = stats.total_trades - stats.winning_trades - stats.losing_trades
        
        # P&L calculations
        winning_pnls = [t.realized_pnl for t in trades if t.realized_pnl > 0]
        losing_pnls = [t.realized_pnl for t in trades if t.realized_pnl < 0]
        
        stats.total_pnl = sum(t.realized_pnl for t in trades)
        stats.gross_profit = sum(winning_pnls)
        stats.gross_loss = sum(losing_pnls)
        
        stats.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        stats.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        stats.largest_win = max(winning_pnls) if winning_pnls else 0
        stats.largest_loss = min(losing_pnls) if losing_pnls else 0
        
        # Ratios
        stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0
        stats.profit_factor = abs(stats.gross_profit / stats.gross_loss) if stats.gross_loss != 0 else float('inf')
        stats.expectancy = stats.total_pnl / stats.total_trades if stats.total_trades > 0 else 0
        stats.payoff_ratio = abs(stats.avg_win / stats.avg_loss) if stats.avg_loss != 0 else float('inf')
        
        # Duration statistics
        durations = []
        winning_durations = []
        losing_durations = []
        
        for trade in trades:
            if trade.holding_time:
                duration_hours = trade.holding_time.total_seconds() / 3600
                durations.append(duration_hours)
                
                if trade.realized_pnl > 0:
                    winning_durations.append(duration_hours)
                elif trade.realized_pnl < 0:
                    losing_durations.append(duration_hours)
        
        if durations:
            stats.avg_duration_hours = np.mean(durations)
            stats.longest_trade_hours = max(durations)
            stats.shortest_trade_hours = min(durations)
        
        if winning_durations:
            stats.avg_winning_duration = np.mean(winning_durations)
        
        if losing_durations:
            stats.avg_losing_duration = np.mean(losing_durations)
        
        # Streak analysis
        self._calculate_streak_statistics(trades, stats)
        
        return stats
    
    def _calculate_streak_statistics(self, trades: List[Trade], stats: TradeStatistics):
        """Calculate win/loss streak statistics"""
        if not trades:
            return
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        current_streak = 0
        win_streaks = []
        loss_streaks = []
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in sorted_trades:
            if trade.realized_pnl > 0:  # Win
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
                current_win_streak += 1
                current_streak = current_win_streak
            elif trade.realized_pnl < 0:  # Loss
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                current_loss_streak += 1
                current_streak = -current_loss_streak
            else:  # Breakeven
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
                current_streak = 0
        
        # Add final streaks
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        # Update statistics
        stats.current_streak = current_streak
        stats.max_win_streak = max(win_streaks) if win_streaks else 0
        stats.max_loss_streak = max(loss_streaks) if loss_streaks else 0
        stats.avg_win_streak = np.mean(win_streaks) if win_streaks else 0
        stats.avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
    
    def _analyze_trade_outcomes(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze trade outcomes distribution"""
        outcomes = defaultdict(int)
        outcome_pnls = defaultdict(list)
        
        for trade in trades:
            # Calculate return percentage
            entry_value = trade.quantity * trade.entry_price
            return_pct = (trade.realized_pnl / entry_value) * 100 if entry_value > 0 else 0
            
            # Classify outcome
            if return_pct > 2:
                outcome = TradeOutcome.BIG_WIN
            elif return_pct > 0.5:
                outcome = TradeOutcome.WIN
            elif return_pct >= -0.5:
                outcome = TradeOutcome.BREAKEVEN
            elif return_pct >= -2:
                outcome = TradeOutcome.LOSS
            else:
                outcome = TradeOutcome.BIG_LOSS
            
            outcomes[outcome.value] += 1
            outcome_pnls[outcome.value].append(trade.realized_pnl)
        
        # Calculate outcome statistics
        total_trades = len(trades)
        outcome_stats = {}
        
        for outcome in TradeOutcome:
            count = outcomes[outcome.value]
            pnls = outcome_pnls[outcome.value]
            
            outcome_stats[outcome.value] = {
                "count": count,
                "percentage": count / total_trades if total_trades > 0 else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0
            }
        
        return {
            "distribution": outcome_stats,
            "return_distribution": self._calculate_return_distribution(trades)
        }
    
    def _calculate_return_distribution(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate return distribution statistics"""
        returns = []
        
        for trade in trades:
            entry_value = trade.quantity * trade.entry_price
            if entry_value > 0:
                return_pct = (trade.realized_pnl / entry_value) * 100
                returns.append(return_pct)
        
        if not returns:
            return {}
        
        return {
            "mean": np.mean(returns),
            "median": np.median(returns),
            "std": np.std(returns),
            "skew": self._calculate_skew(returns),
            "kurtosis": self._calculate_kurtosis(returns),
            "percentiles": {
                "5%": np.percentile(returns, 5),
                "25%": np.percentile(returns, 25),
                "50%": np.percentile(returns, 50),
                "75%": np.percentile(returns, 75),
                "95%": np.percentile(returns, 95)
            }
        }
    
    async def _analyze_entry_exit_quality(
        self,
        trades: List[Trade],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze entry and exit quality"""
        entry_qualities = defaultdict(int)
        exit_qualities = defaultdict(int)
        
        for trade in trades:
            # Simple quality assessment based on P&L
            # In production, would compare against market highs/lows
            entry_quality = self._assess_entry_quality(trade, market_data)
            exit_quality = self._assess_exit_quality(trade, market_data)
            
            entry_qualities[entry_quality.value] += 1
            exit_qualities[exit_quality.value] += 1
        
        total_trades = len(trades)
        
        return {
            "entry_quality": {
                quality: {
                    "count": count,
                    "percentage": count / total_trades if total_trades > 0 else 0
                }
                for quality, count in entry_qualities.items()
            },
            "exit_quality": {
                quality: {
                    "count": count,
                    "percentage": count / total_trades if total_trades > 0 else 0
                }
                for quality, count in exit_qualities.items()
            },
            "quality_scores": {
                "avg_entry_score": self._calculate_avg_quality_score(entry_qualities, total_trades),
                "avg_exit_score": self._calculate_avg_quality_score(exit_qualities, total_trades)
            }
        }
    
    def _assess_entry_quality(
        self,
        trade: Trade,
        market_data: Optional[Dict[str, Any]]
    ) -> EntryQuality:
        """Assess entry quality of a trade"""
        # Simplified assessment based on outcome
        # In production, would analyze against market data
        return_pct = (trade.realized_pnl / (trade.quantity * trade.entry_price)) * 100
        
        if return_pct > 5:
            return EntryQuality.EXCELLENT
        elif return_pct > 2:
            return EntryQuality.GOOD
        elif return_pct > -1:
            return EntryQuality.AVERAGE
        elif return_pct > -3:
            return EntryQuality.POOR
        else:
            return EntryQuality.TERRIBLE
    
    def _assess_exit_quality(
        self,
        trade: Trade,
        market_data: Optional[Dict[str, Any]]
    ) -> EntryQuality:
        """Assess exit quality of a trade"""
        # For now, use same logic as entry
        return self._assess_entry_quality(trade, market_data)
    
    def _analyze_holding_periods(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze holding period patterns"""
        holding_periods = defaultdict(list)
        period_pnls = defaultdict(list)
        
        # Categorize by holding period
        for trade in trades:
            if not trade.holding_time:
                continue
            
            hours = trade.holding_time.total_seconds() / 3600
            
            if hours < 1:
                category = "scalp"  # < 1 hour
            elif hours < 4:
                category = "intraday"  # 1-4 hours
            elif hours < 24:
                category = "day_trade"  # 4-24 hours
            elif hours < 168:
                category = "swing"  # 1-7 days
            else:
                category = "position"  # > 7 days
            
            holding_periods[category].append(hours)
            period_pnls[category].append(trade.realized_pnl)
        
        # Calculate statistics for each category
        period_stats = {}
        
        for category in ["scalp", "intraday", "day_trade", "swing", "position"]:
            periods = holding_periods[category]
            pnls = period_pnls[category]
            
            period_stats[category] = {
                "count": len(periods),
                "avg_duration_hours": np.mean(periods) if periods else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            }
        
        return {
            "by_category": period_stats,
            "optimal_holding": self._find_optimal_holding_period(trades)
        }
    
    def _find_optimal_holding_period(self, trades: List[Trade]) -> Dict[str, Any]:
        """Find optimal holding period based on profitability"""
        # Group trades by holding period buckets
        buckets = defaultdict(list)
        
        for trade in trades:
            if not trade.holding_time:
                continue
            
            hours = trade.holding_time.total_seconds() / 3600
            bucket = int(hours / 4) * 4  # 4-hour buckets
            buckets[bucket].append(trade.realized_pnl)
        
        # Find bucket with best average P&L
        best_bucket = None
        best_avg_pnl = float('-inf')
        
        bucket_stats = []
        for bucket, pnls in buckets.items():
            avg_pnl = np.mean(pnls)
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
            
            bucket_stats.append({
                "hours_range": f"{bucket}-{bucket+4}",
                "avg_pnl": avg_pnl,
                "win_rate": win_rate,
                "trade_count": len(pnls)
            })
            
            if avg_pnl > best_avg_pnl and len(pnls) >= 5:  # Minimum 5 trades
                best_avg_pnl = avg_pnl
                best_bucket = bucket
        
        return {
            "optimal_range_hours": f"{best_bucket}-{best_bucket+4}" if best_bucket is not None else "Unknown",
            "expected_pnl": best_avg_pnl if best_bucket is not None else 0,
            "all_buckets": sorted(bucket_stats, key=lambda x: x["avg_pnl"], reverse=True)[:5]
        }
    
    def _analyze_trade_clustering(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze trade clustering patterns"""
        if not trades:
            return {}
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        # Analyze time between trades
        inter_trade_times = []
        for i in range(1, len(sorted_trades)):
            time_diff = (sorted_trades[i].timestamp - sorted_trades[i-1].timestamp).total_seconds() / 3600
            inter_trade_times.append(time_diff)
        
        if not inter_trade_times:
            return {}
        
        # Identify clusters (trades within 1 hour of each other)
        clusters = []
        current_cluster = [sorted_trades[0]]
        
        for i in range(1, len(sorted_trades)):
            if inter_trade_times[i-1] <= 1:  # Within 1 hour
                current_cluster.append(sorted_trades[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [sorted_trades[i]]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        # Analyze cluster performance
        cluster_stats = []
        for cluster in clusters:
            cluster_pnl = sum(t.realized_pnl for t in cluster)
            cluster_stats.append({
                "size": len(cluster),
                "total_pnl": cluster_pnl,
                "avg_pnl": cluster_pnl / len(cluster),
                "duration_hours": (cluster[-1].timestamp - cluster[0].timestamp).total_seconds() / 3600
            })
        
        return {
            "avg_time_between_trades": np.mean(inter_trade_times),
            "median_time_between_trades": np.median(inter_trade_times),
            "cluster_count": len(clusters),
            "avg_cluster_size": np.mean([c["size"] for c in cluster_stats]) if cluster_stats else 0,
            "cluster_performance": {
                "profitable_clusters": sum(1 for c in cluster_stats if c["total_pnl"] > 0),
                "avg_cluster_pnl": np.mean([c["total_pnl"] for c in cluster_stats]) if cluster_stats else 0
            }
        }
    
    def _analyze_by_time_periods(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze trades by various time periods"""
        by_hour = defaultdict(list)
        by_day_of_week = defaultdict(list)
        by_month = defaultdict(list)
        
        for trade in trades:
            by_hour[trade.timestamp.hour].append(trade.realized_pnl)
            by_day_of_week[trade.timestamp.weekday()].append(trade.realized_pnl)
            by_month[trade.timestamp.month].append(trade.realized_pnl)
        
        # Calculate statistics for each period
        hour_stats = {}
        for hour in range(24):
            pnls = by_hour[hour]
            hour_stats[hour] = {
                "count": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            }
        
        day_stats = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day_num, day_name in enumerate(days):
            pnls = by_day_of_week[day_num]
            day_stats[day_name] = {
                "count": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            }
        
        # Find best trading times
        best_hours = sorted(hour_stats.items(), key=lambda x: x[1]["avg_pnl"], reverse=True)[:3]
        best_days = sorted(day_stats.items(), key=lambda x: x[1]["avg_pnl"], reverse=True)[:3]
        
        return {
            "by_hour": hour_stats,
            "by_day_of_week": day_stats,
            "best_hours": [{"hour": h, "avg_pnl": s["avg_pnl"]} for h, s in best_hours],
            "best_days": [{"day": d, "avg_pnl": s["avg_pnl"]} for d, s in best_days]
        }
    
    async def _analyze_by_market_conditions(
        self,
        trades: List[Trade],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trades by market conditions"""
        # Simplified analysis without real market data
        volatility_trades = defaultdict(list)
        trend_trades = defaultdict(list)
        
        for trade in trades:
            # Mock volatility classification
            if trade.timestamp.hour < 8:
                volatility = "low"
            elif trade.timestamp.hour < 16:
                volatility = "medium"
            else:
                volatility = "high"
            
            volatility_trades[volatility].append(trade.realized_pnl)
            
            # Mock trend classification based on P&L
            if trade.realized_pnl > 0:
                trend = "uptrend"
            else:
                trend = "downtrend"
            
            trend_trades[trend].append(trade.realized_pnl)
        
        # Calculate statistics
        volatility_stats = {}
        for vol_level in ["low", "medium", "high"]:
            pnls = volatility_trades[vol_level]
            volatility_stats[vol_level] = {
                "count": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            }
        
        return {
            "by_volatility": volatility_stats,
            "by_trend": {
                trend: {
                    "count": len(pnls),
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls) if pnls else 0
                }
                for trend, pnls in trend_trades.items()
            }
        }
    
    def _analyze_consecutive_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze patterns in consecutive trades"""
        if len(trades) < 2:
            return {}
        
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        # Analyze win/loss patterns
        after_win_stats = {"wins": 0, "losses": 0, "total": 0}
        after_loss_stats = {"wins": 0, "losses": 0, "total": 0}
        
        for i in range(1, len(sorted_trades)):
            prev_trade = sorted_trades[i-1]
            curr_trade = sorted_trades[i]
            
            if prev_trade.realized_pnl > 0:  # After win
                after_win_stats["total"] += 1
                if curr_trade.realized_pnl > 0:
                    after_win_stats["wins"] += 1
                else:
                    after_win_stats["losses"] += 1
            elif prev_trade.realized_pnl < 0:  # After loss
                after_loss_stats["total"] += 1
                if curr_trade.realized_pnl > 0:
                    after_loss_stats["wins"] += 1
                else:
                    after_loss_stats["losses"] += 1
        
        return {
            "after_win": {
                "win_rate": after_win_stats["wins"] / after_win_stats["total"] if after_win_stats["total"] > 0 else 0,
                "sample_size": after_win_stats["total"]
            },
            "after_loss": {
                "win_rate": after_loss_stats["wins"] / after_loss_stats["total"] if after_loss_stats["total"] > 0 else 0,
                "sample_size": after_loss_stats["total"]
            },
            "pattern_analysis": self._analyze_win_loss_patterns(sorted_trades)
        }
    
    def _analyze_win_loss_patterns(self, sorted_trades: List[Trade]) -> Dict[str, Any]:
        """Analyze win/loss patterns"""
        patterns = defaultdict(int)
        
        # Look for 3-trade patterns
        for i in range(2, len(sorted_trades)):
            pattern = ""
            for j in range(3):
                if sorted_trades[i-2+j].realized_pnl > 0:
                    pattern += "W"
                else:
                    pattern += "L"
            patterns[pattern] += 1
        
        total_patterns = sum(patterns.values())
        
        return {
            pattern: {
                "count": count,
                "percentage": count / total_patterns if total_patterns > 0 else 0
            }
            for pattern, count in patterns.items()
        }
    
    def _generate_trade_insights(
        self,
        stats: TradeStatistics,
        outcome_analysis: Dict[str, Any],
        entry_exit_analysis: Dict[str, Any],
        holding_period_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate actionable insights from trade analysis"""
        insights = []
        
        # Win rate insights
        if stats.win_rate < 0.4:
            insights.append({
                "category": "Win Rate",
                "severity": "High",
                "insight": f"Low win rate of {stats.win_rate:.1%}. Consider reviewing entry criteria.",
                "recommendation": "Focus on higher probability setups or adjust stop loss levels."
            })
        
        # Profit factor insights
        if stats.profit_factor < 1.5:
            insights.append({
                "category": "Risk/Reward",
                "severity": "Medium",
                "insight": f"Profit factor of {stats.profit_factor:.2f} is below optimal levels.",
                "recommendation": "Improve risk/reward ratio by targeting larger profits or reducing losses."
            })
        
        # Holding period insights
        optimal_holding = holding_period_analysis.get("optimal_holding", {})
        if optimal_holding.get("optimal_range_hours"):
            insights.append({
                "category": "Timing",
                "severity": "Low",
                "insight": f"Best performance in {optimal_holding['optimal_range_hours']} hour holding period.",
                "recommendation": "Consider adjusting exit timing to match optimal holding period."
            })
        
        # Streak insights
        if stats.max_loss_streak > 5:
            insights.append({
                "category": "Risk Management",
                "severity": "High",
                "insight": f"Maximum losing streak of {stats.max_loss_streak} trades detected.",
                "recommendation": "Implement circuit breakers or position size reduction after consecutive losses."
            })
        
        # Entry quality insights
        entry_scores = entry_exit_analysis.get("quality_scores", {})
        if entry_scores.get("avg_entry_score", 0) < 3:
            insights.append({
                "category": "Entry Timing",
                "severity": "Medium",
                "insight": "Below average entry quality detected.",
                "recommendation": "Refine entry signals or wait for better setups."
            })
        
        return insights
    
    def _calculate_avg_quality_score(
        self,
        quality_distribution: Dict[str, int],
        total_trades: int
    ) -> float:
        """Calculate average quality score from distribution"""
        if total_trades == 0:
            return 0
        
        quality_scores = {
            EntryQuality.EXCELLENT.value: 5,
            EntryQuality.GOOD.value: 4,
            EntryQuality.AVERAGE.value: 3,
            EntryQuality.POOR.value: 2,
            EntryQuality.TERRIBLE.value: 1
        }
        
        total_score = 0
        for quality, count in quality_distribution.items():
            total_score += quality_scores.get(quality, 3) * count
        
        return total_score / total_trades
    
    def _calculate_skew(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0
        
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0
        
        return np.mean(((returns - mean) / std) ** 4) - 3