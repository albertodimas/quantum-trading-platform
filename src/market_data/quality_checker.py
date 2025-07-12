"""
Market Data Quality Checker

Validates incoming market data for anomalies, errors, and quality issues.
Includes statistical checks, outlier detection, and consistency validation.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import deque
import numpy as np

from ..core.interfaces import Injectable
from ..core.decorators import injectable
from ..core.logger import get_logger
from ..models.market_data import MarketData, OHLCV, Ticker, OrderBook, Trade

logger = get_logger(__name__)


class QualityIssue(Enum):
    """Types of quality issues"""
    STALE_DATA = "stale_data"
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    SPREAD_ANOMALY = "spread_anomaly"
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    TIMESTAMP_ERROR = "timestamp_error"
    NEGATIVE_PRICE = "negative_price"
    CROSSED_QUOTES = "crossed_quotes"
    OUTLIER = "outlier"


@dataclass
class QualityCheck:
    """Quality check result"""
    passed: bool
    issues: List[QualityIssue]
    confidence: float
    details: Dict[str, Any]


@dataclass
class QualityConfig:
    """Quality check configuration"""
    # Staleness thresholds (seconds)
    ticker_stale_threshold: int = 60
    orderbook_stale_threshold: int = 30
    trade_stale_threshold: int = 120
    
    # Price movement thresholds
    max_price_change_pct: float = 20.0  # 20% max change
    max_spread_pct: float = 5.0  # 5% max spread
    
    # Statistical parameters
    outlier_std_threshold: float = 3.0  # 3 standard deviations
    min_samples_for_stats: int = 100
    
    # Volume thresholds
    min_volume: float = 0.0
    max_volume_change_pct: float = 500.0  # 500% max change
    
    # Enable/disable checks
    check_staleness: bool = True
    check_price_spikes: bool = True
    check_spreads: bool = True
    check_volumes: bool = True
    check_timestamps: bool = True
    check_duplicates: bool = True


class StatisticalAnalyzer:
    """Performs statistical analysis on market data"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.spread_history = deque(maxlen=window_size)
        
    def update(self, price: float, volume: float = None, spread: float = None):
        """Update historical data"""
        self.price_history.append(price)
        if volume is not None:
            self.volume_history.append(volume)
        if spread is not None:
            self.spread_history.append(spread)
    
    def is_price_outlier(self, price: float, threshold: float) -> bool:
        """Check if price is an outlier"""
        if len(self.price_history) < 10:
            return False
        
        mean = statistics.mean(self.price_history)
        std = statistics.stdev(self.price_history)
        
        z_score = abs((price - mean) / std) if std > 0 else 0
        return z_score > threshold
    
    def is_volume_anomaly(self, volume: float, threshold_pct: float) -> bool:
        """Check if volume is anomalous"""
        if len(self.volume_history) < 10:
            return False
        
        median_volume = statistics.median(self.volume_history)
        if median_volume == 0:
            return volume > 0
        
        change_pct = abs((volume - median_volume) / median_volume) * 100
        return change_pct > threshold_pct
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = {}
        
        if self.price_history:
            stats["price"] = {
                "mean": statistics.mean(self.price_history),
                "std": statistics.stdev(self.price_history) if len(self.price_history) > 1 else 0,
                "min": min(self.price_history),
                "max": max(self.price_history)
            }
        
        if self.volume_history:
            stats["volume"] = {
                "mean": statistics.mean(self.volume_history),
                "median": statistics.median(self.volume_history),
                "total": sum(self.volume_history)
            }
        
        if self.spread_history:
            stats["spread"] = {
                "mean": statistics.mean(self.spread_history),
                "max": max(self.spread_history)
            }
        
        return stats


@injectable
class MarketDataQualityChecker(Injectable):
    """Checks market data quality and detects anomalies"""
    
    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()
        self._analyzers: Dict[str, StatisticalAnalyzer] = {}
        self._last_data: Dict[str, MarketData] = {}
        self._duplicate_cache: deque = deque(maxlen=10000)
        self._issue_counts: Dict[QualityIssue, int] = {issue: 0 for issue in QualityIssue}
        
    async def check(self, data: MarketData) -> QualityCheck:
        """Check market data quality"""
        issues = []
        details = {}
        
        # Get or create analyzer for this symbol
        analyzer_key = f"{data.exchange}:{data.symbol}"
        if analyzer_key not in self._analyzers:
            self._analyzers[analyzer_key] = StatisticalAnalyzer()
        analyzer = self._analyzers[analyzer_key]
        
        # Perform checks based on data type
        if isinstance(data.data, Ticker):
            issues.extend(await self._check_ticker(data, analyzer, details))
        elif isinstance(data.data, OrderBook):
            issues.extend(await self._check_orderbook(data, analyzer, details))
        elif isinstance(data.data, Trade):
            issues.extend(await self._check_trade(data, analyzer, details))
        elif isinstance(data.data, OHLCV):
            issues.extend(await self._check_ohlcv(data, analyzer, details))
        
        # Common checks
        if self.config.check_timestamps:
            issues.extend(self._check_timestamp(data, details))
        
        if self.config.check_duplicates:
            issues.extend(self._check_duplicate(data, details))
        
        # Update issue counts
        for issue in issues:
            self._issue_counts[issue] += 1
        
        # Calculate confidence
        confidence = self._calculate_confidence(issues, analyzer)
        
        # Store for future comparison
        self._last_data[analyzer_key] = data
        
        return QualityCheck(
            passed=len(issues) == 0,
            issues=issues,
            confidence=confidence,
            details=details
        )
    
    async def _check_ticker(
        self,
        data: MarketData,
        analyzer: StatisticalAnalyzer,
        details: Dict[str, Any]
    ) -> List[QualityIssue]:
        """Check ticker data quality"""
        issues = []
        ticker = data.data
        
        # Check negative prices
        if ticker.bid and ticker.bid < 0:
            issues.append(QualityIssue.NEGATIVE_PRICE)
            details["negative_bid"] = ticker.bid
        
        if ticker.ask and ticker.ask < 0:
            issues.append(QualityIssue.NEGATIVE_PRICE)
            details["negative_ask"] = ticker.ask
        
        # Check crossed quotes
        if ticker.bid and ticker.ask and ticker.bid >= ticker.ask:
            issues.append(QualityIssue.CROSSED_QUOTES)
            details["bid"] = ticker.bid
            details["ask"] = ticker.ask
        
        # Check spread
        if ticker.bid and ticker.ask and self.config.check_spreads:
            spread_pct = ((ticker.ask - ticker.bid) / ticker.bid) * 100
            if spread_pct > self.config.max_spread_pct:
                issues.append(QualityIssue.SPREAD_ANOMALY)
                details["spread_pct"] = spread_pct
            analyzer.update(ticker.last, spread=spread_pct)
        else:
            analyzer.update(ticker.last)
        
        # Check price spike
        if self.config.check_price_spikes and ticker.last:
            if analyzer.is_price_outlier(ticker.last, self.config.outlier_std_threshold):
                issues.append(QualityIssue.PRICE_SPIKE)
                details["price"] = ticker.last
                details["stats"] = analyzer.get_statistics()
        
        # Check volume
        if ticker.volume and self.config.check_volumes:
            if ticker.volume < self.config.min_volume:
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["volume"] = ticker.volume
            elif analyzer.is_volume_anomaly(ticker.volume, self.config.max_volume_change_pct):
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["volume"] = ticker.volume
        
        return issues
    
    async def _check_orderbook(
        self,
        data: MarketData,
        analyzer: StatisticalAnalyzer,
        details: Dict[str, Any]
    ) -> List[QualityIssue]:
        """Check orderbook data quality"""
        issues = []
        orderbook = data.data
        
        # Check for empty orderbook
        if not orderbook.bids and not orderbook.asks:
            issues.append(QualityIssue.MISSING_DATA)
            details["empty_orderbook"] = True
            return issues
        
        # Check best bid/ask
        best_bid = orderbook.bids[0][0] if orderbook.bids else None
        best_ask = orderbook.asks[0][0] if orderbook.asks else None
        
        # Check negative prices
        if best_bid and best_bid < 0:
            issues.append(QualityIssue.NEGATIVE_PRICE)
            details["negative_bid"] = best_bid
        
        if best_ask and best_ask < 0:
            issues.append(QualityIssue.NEGATIVE_PRICE)
            details["negative_ask"] = best_ask
        
        # Check crossed quotes
        if best_bid and best_ask and best_bid >= best_ask:
            issues.append(QualityIssue.CROSSED_QUOTES)
            details["best_bid"] = best_bid
            details["best_ask"] = best_ask
        
        # Check spread
        if best_bid and best_ask and self.config.check_spreads:
            spread_pct = ((best_ask - best_bid) / best_bid) * 100
            if spread_pct > self.config.max_spread_pct:
                issues.append(QualityIssue.SPREAD_ANOMALY)
                details["spread_pct"] = spread_pct
            
            mid_price = (best_bid + best_ask) / 2
            analyzer.update(mid_price, spread=spread_pct)
        
        # Check orderbook depth
        total_bid_volume = sum(level[1] for level in orderbook.bids)
        total_ask_volume = sum(level[1] for level in orderbook.asks)
        
        if self.config.check_volumes:
            if total_bid_volume < self.config.min_volume:
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["bid_volume"] = total_bid_volume
            
            if total_ask_volume < self.config.min_volume:
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["ask_volume"] = total_ask_volume
        
        return issues
    
    async def _check_trade(
        self,
        data: MarketData,
        analyzer: StatisticalAnalyzer,
        details: Dict[str, Any]
    ) -> List[QualityIssue]:
        """Check trade data quality"""
        issues = []
        trade = data.data
        
        # Check negative price
        if trade.price < 0:
            issues.append(QualityIssue.NEGATIVE_PRICE)
            details["price"] = trade.price
        
        # Check price spike
        if self.config.check_price_spikes:
            if analyzer.is_price_outlier(trade.price, self.config.outlier_std_threshold):
                issues.append(QualityIssue.PRICE_SPIKE)
                details["price"] = trade.price
                details["stats"] = analyzer.get_statistics()
        
        analyzer.update(trade.price, volume=trade.amount)
        
        # Check volume
        if self.config.check_volumes:
            if trade.amount < self.config.min_volume:
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["amount"] = trade.amount
            elif analyzer.is_volume_anomaly(trade.amount, self.config.max_volume_change_pct):
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["amount"] = trade.amount
        
        return issues
    
    async def _check_ohlcv(
        self,
        data: MarketData,
        analyzer: StatisticalAnalyzer,
        details: Dict[str, Any]
    ) -> List[QualityIssue]:
        """Check OHLCV data quality"""
        issues = []
        ohlcv = data.data
        
        # Check price consistency
        if ohlcv.high < ohlcv.low:
            issues.append(QualityIssue.MISSING_DATA)
            details["high"] = ohlcv.high
            details["low"] = ohlcv.low
        
        if ohlcv.open < ohlcv.low or ohlcv.open > ohlcv.high:
            issues.append(QualityIssue.MISSING_DATA)
            details["open"] = ohlcv.open
        
        if ohlcv.close < ohlcv.low or ohlcv.close > ohlcv.high:
            issues.append(QualityIssue.MISSING_DATA)
            details["close"] = ohlcv.close
        
        # Check negative prices
        for price_type, price in [("open", ohlcv.open), ("high", ohlcv.high),
                                  ("low", ohlcv.low), ("close", ohlcv.close)]:
            if price < 0:
                issues.append(QualityIssue.NEGATIVE_PRICE)
                details[f"negative_{price_type}"] = price
        
        # Check price movement
        if self.config.check_price_spikes:
            typical_price = (ohlcv.high + ohlcv.low + ohlcv.close) / 3
            if analyzer.is_price_outlier(typical_price, self.config.outlier_std_threshold):
                issues.append(QualityIssue.PRICE_SPIKE)
                details["typical_price"] = typical_price
        
        analyzer.update(ohlcv.close, volume=ohlcv.volume)
        
        # Check volume
        if self.config.check_volumes:
            if ohlcv.volume < self.config.min_volume:
                issues.append(QualityIssue.VOLUME_ANOMALY)
                details["volume"] = ohlcv.volume
        
        return issues
    
    def _check_timestamp(self, data: MarketData, details: Dict[str, Any]) -> List[QualityIssue]:
        """Check timestamp validity"""
        issues = []
        
        # Check if timestamp is in the future
        if data.timestamp > datetime.utcnow() + timedelta(seconds=60):
            issues.append(QualityIssue.TIMESTAMP_ERROR)
            details["future_timestamp"] = data.timestamp
        
        # Check staleness
        if self.config.check_staleness:
            age_seconds = (datetime.utcnow() - data.timestamp).total_seconds()
            
            if isinstance(data.data, Ticker) and age_seconds > self.config.ticker_stale_threshold:
                issues.append(QualityIssue.STALE_DATA)
                details["age_seconds"] = age_seconds
            elif isinstance(data.data, OrderBook) and age_seconds > self.config.orderbook_stale_threshold:
                issues.append(QualityIssue.STALE_DATA)
                details["age_seconds"] = age_seconds
            elif isinstance(data.data, Trade) and age_seconds > self.config.trade_stale_threshold:
                issues.append(QualityIssue.STALE_DATA)
                details["age_seconds"] = age_seconds
        
        return issues
    
    def _check_duplicate(self, data: MarketData, details: Dict[str, Any]) -> List[QualityIssue]:
        """Check for duplicate data"""
        issues = []
        
        # Create hash of the data
        data_hash = hash((
            data.exchange,
            data.symbol,
            data.timestamp,
            data.data_type,
            str(data.data)
        ))
        
        if data_hash in self._duplicate_cache:
            issues.append(QualityIssue.DUPLICATE_DATA)
            details["duplicate_hash"] = data_hash
        else:
            self._duplicate_cache.append(data_hash)
        
        return issues
    
    def _calculate_confidence(self, issues: List[QualityIssue], analyzer: StatisticalAnalyzer) -> float:
        """Calculate confidence score"""
        if not issues:
            return 1.0
        
        # Weight different issues
        weights = {
            QualityIssue.NEGATIVE_PRICE: 0.0,  # Critical
            QualityIssue.CROSSED_QUOTES: 0.1,
            QualityIssue.TIMESTAMP_ERROR: 0.2,
            QualityIssue.DUPLICATE_DATA: 0.3,
            QualityIssue.PRICE_SPIKE: 0.5,
            QualityIssue.SPREAD_ANOMALY: 0.6,
            QualityIssue.VOLUME_ANOMALY: 0.7,
            QualityIssue.STALE_DATA: 0.8,
            QualityIssue.MISSING_DATA: 0.4,
            QualityIssue.OUTLIER: 0.5
        }
        
        # Get minimum weight
        min_weight = min(weights.get(issue, 0.5) for issue in issues)
        
        # Consider statistical confidence
        if len(analyzer.price_history) < self.config.min_samples_for_stats:
            min_weight = min(min_weight, 0.7)
        
        return max(0.0, min_weight)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality checker statistics"""
        return {
            "analyzers": len(self._analyzers),
            "issue_counts": dict(self._issue_counts),
            "cache_size": len(self._duplicate_cache)
        }