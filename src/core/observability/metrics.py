"""
Advanced metrics collection system with OpenTelemetry and custom trading metrics.

Provides comprehensive metrics for:
- Trading operations and performance
- System health and resource usage
- Business KPIs and analytics
- Real-time dashboards and alerting
"""

import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import statistics
from contextlib import contextmanager

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.metrics import Counter, Histogram, Gauge, UpDownCounter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

import psutil


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class MetricUnit(Enum):
    """Standard metric units."""
    DIMENSIONLESS = ""
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "μs"
    BYTES = "bytes"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    PERCENT = "%"
    COUNT = "count"
    RATE_PER_SECOND = "/s"
    CURRENCY_USD = "USD"
    CURRENCY_BTC = "BTC"
    CURRENCY_ETH = "ETH"


@dataclass
class MetricValue:
    """Container for metric values with metadata."""
    name: str
    value: Union[int, float]
    unit: MetricUnit = MetricUnit.DIMENSIONLESS
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class HistogramStats:
    """Statistical summary for histogram metrics."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float


class MetricsStorage:
    """Thread-safe storage for metrics data."""
    
    def __init__(self, max_samples: int = 10000):
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.RLock()
    
    def record(self, metric: MetricValue):
        """Record a metric value."""
        with self._lock:
            key = f"{metric.name}:{','.join(f'{k}={v}' for k, v in sorted(metric.tags.items()))}"
            self._data[key].append(metric)
    
    def get_recent(self, name: str, tags: Optional[Dict[str, str]] = None, limit: int = 100) -> List[MetricValue]:
        """Get recent values for a metric."""
        with self._lock:
            tag_str = ','.join(f'{k}={v}' for k, v in sorted((tags or {}).items()))
            key = f"{name}:{tag_str}"
            return list(self._data[key])[-limit:]
    
    def get_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[HistogramStats]:
        """Get statistical summary for a metric."""
        values = self.get_recent(name, tags)
        if not values:
            return None
        
        numeric_values = [v.value for v in values]
        return HistogramStats(
            count=len(numeric_values),
            sum=sum(numeric_values),
            min=min(numeric_values),
            max=max(numeric_values),
            mean=statistics.mean(numeric_values),
            median=statistics.median(numeric_values),
            p95=statistics.quantiles(numeric_values, n=20)[18] if len(numeric_values) > 1 else numeric_values[0],
            p99=statistics.quantiles(numeric_values, n=100)[98] if len(numeric_values) > 1 else numeric_values[0]
        )


class BaseMetricsCollector(ABC):
    """Abstract base class for metrics collectors."""
    
    def __init__(self, name: str, storage: Optional[MetricsStorage] = None):
        self.name = name
        self.storage = storage or MetricsStorage()
        self._enabled = True
    
    def enable(self):
        """Enable metrics collection."""
        self._enabled = True
    
    def disable(self):
        """Disable metrics collection."""
        self._enabled = False
    
    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect current metrics."""
        pass
    
    def record_metric(self, name: str, value: Union[int, float], unit: MetricUnit = MetricUnit.DIMENSIONLESS, 
                     tags: Optional[Dict[str, str]] = None, description: str = ""):
        """Record a metric value."""
        if not self._enabled:
            return
        
        metric = MetricValue(
            name=f"{self.name}.{name}",
            value=value,
            unit=unit,
            tags=tags or {},
            description=description
        )
        self.storage.record(metric)


class SystemMetrics(BaseMetricsCollector):
    """Collector for system health and resource metrics."""
    
    def __init__(self, storage: Optional[MetricsStorage] = None):
        super().__init__("system", storage)
        self._process = psutil.Process()
    
    def collect(self) -> List[MetricValue]:
        """Collect system metrics."""
        if not self._enabled:
            return []
        
        metrics = []
        
        # CPU metrics
        cpu_percent = self._process.cpu_percent()
        system_cpu = psutil.cpu_percent()
        metrics.append(MetricValue("system.cpu.process_percent", cpu_percent, MetricUnit.PERCENT))
        metrics.append(MetricValue("system.cpu.system_percent", system_cpu, MetricUnit.PERCENT))
        
        # Memory metrics
        memory_info = self._process.memory_info()
        system_memory = psutil.virtual_memory()
        metrics.append(MetricValue("system.memory.rss", memory_info.rss, MetricUnit.BYTES))
        metrics.append(MetricValue("system.memory.vms", memory_info.vms, MetricUnit.BYTES))
        metrics.append(MetricValue("system.memory.available", system_memory.available, MetricUnit.BYTES))
        metrics.append(MetricValue("system.memory.percent", system_memory.percent, MetricUnit.PERCENT))
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(MetricValue("system.disk.read_bytes", disk_io.read_bytes, MetricUnit.BYTES))
                metrics.append(MetricValue("system.disk.write_bytes", disk_io.write_bytes, MetricUnit.BYTES))
        except:
            pass
        
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.append(MetricValue("system.network.bytes_sent", net_io.bytes_sent, MetricUnit.BYTES))
                metrics.append(MetricValue("system.network.bytes_recv", net_io.bytes_recv, MetricUnit.BYTES))
        except:
            pass
        
        # File descriptors
        try:
            num_fds = self._process.num_fds()
            metrics.append(MetricValue("system.process.file_descriptors", num_fds, MetricUnit.COUNT))
        except:
            pass
        
        # Record all metrics
        for metric in metrics:
            self.storage.record(metric)
        
        return metrics


class TradingMetrics(BaseMetricsCollector):
    """Collector for trading-specific metrics."""
    
    def __init__(self, storage: Optional[MetricsStorage] = None):
        super().__init__("trading", storage)
        self._trade_count = 0
        self._total_volume = 0.0
        self._profit_loss = 0.0
        self._last_price = {}
    
    def record_trade(self, symbol: str, side: str, quantity: float, price: float, 
                    commission: float = 0.0, profit_loss: float = 0.0):
        """Record a trade execution."""
        if not self._enabled:
            return
        
        self._trade_count += 1
        volume = quantity * price
        self._total_volume += volume
        self._profit_loss += profit_loss
        self._last_price[symbol] = price
        
        tags = {"symbol": symbol, "side": side}
        
        self.record_metric("trades.count", 1, MetricUnit.COUNT, tags)
        self.record_metric("trades.volume", volume, MetricUnit.CURRENCY_USD, tags)
        self.record_metric("trades.quantity", quantity, MetricUnit.DIMENSIONLESS, tags)
        self.record_metric("trades.price", price, MetricUnit.CURRENCY_USD, tags)
        self.record_metric("trades.commission", commission, MetricUnit.CURRENCY_USD, tags)
        self.record_metric("trades.profit_loss", profit_loss, MetricUnit.CURRENCY_USD, tags)
    
    def record_order(self, symbol: str, side: str, order_type: str, status: str, 
                    quantity: float, price: Optional[float] = None):
        """Record an order event."""
        if not self._enabled:
            return
        
        tags = {"symbol": symbol, "side": side, "type": order_type, "status": status}
        
        self.record_metric("orders.count", 1, MetricUnit.COUNT, tags)
        self.record_metric("orders.quantity", quantity, MetricUnit.DIMENSIONLESS, tags)
        if price:
            self.record_metric("orders.price", price, MetricUnit.CURRENCY_USD, tags)
    
    def record_market_data_latency(self, symbol: str, latency_ms: float):
        """Record market data latency."""
        if not self._enabled:
            return
        
        tags = {"symbol": symbol}
        self.record_metric("market_data.latency", latency_ms, MetricUnit.MILLISECONDS, tags)
    
    def record_strategy_performance(self, strategy_name: str, return_pct: float, 
                                  sharpe_ratio: Optional[float] = None, 
                                  max_drawdown: Optional[float] = None):
        """Record strategy performance metrics."""
        if not self._enabled:
            return
        
        tags = {"strategy": strategy_name}
        
        self.record_metric("strategy.return", return_pct, MetricUnit.PERCENT, tags)
        if sharpe_ratio:
            self.record_metric("strategy.sharpe_ratio", sharpe_ratio, MetricUnit.DIMENSIONLESS, tags)
        if max_drawdown:
            self.record_metric("strategy.max_drawdown", max_drawdown, MetricUnit.PERCENT, tags)
    
    def record_risk_metric(self, metric_name: str, value: float, symbol: Optional[str] = None):
        """Record risk management metrics."""
        if not self._enabled:
            return
        
        tags = {"symbol": symbol} if symbol else {}
        self.record_metric(f"risk.{metric_name}", value, MetricUnit.DIMENSIONLESS, tags)
    
    def collect(self) -> List[MetricValue]:
        """Collect current trading metrics."""
        if not self._enabled:
            return []
        
        metrics = []
        
        # Aggregate metrics
        metrics.append(MetricValue("trading.total_trades", self._trade_count, MetricUnit.COUNT))
        metrics.append(MetricValue("trading.total_volume", self._total_volume, MetricUnit.CURRENCY_USD))
        metrics.append(MetricValue("trading.total_pnl", self._profit_loss, MetricUnit.CURRENCY_USD))
        
        # Latest prices
        for symbol, price in self._last_price.items():
            metrics.append(MetricValue("trading.last_price", price, MetricUnit.CURRENCY_USD, {"symbol": symbol}))
        
        # Record all metrics
        for metric in metrics:
            self.storage.record(metric)
        
        return metrics


class MetricsCollector:
    """
    Main metrics collection system with multiple collectors.
    
    Features:
    - Multiple metric collectors (system, trading, custom)
    - OpenTelemetry integration
    - Periodic collection and export
    - Real-time dashboards support
    - Alert integration for thresholds
    """
    
    def __init__(self, enable_otel: bool = True):
        self.storage = MetricsStorage()
        self.collectors: Dict[str, BaseMetricsCollector] = {}
        self.enable_otel = enable_otel and OTEL_AVAILABLE
        self._collection_interval = 60  # seconds
        self._collection_thread = None
        self._running = False
        
        # Initialize default collectors
        self.add_collector("system", SystemMetrics(self.storage))
        self.add_collector("trading", TradingMetrics(self.storage))
        
        # Initialize OpenTelemetry if available
        if self.enable_otel:
            self._setup_otel()
    
    def _setup_otel(self):
        """Setup OpenTelemetry metrics."""
        if not OTEL_AVAILABLE:
            return
        
        self.meter = otel_metrics.get_meter(__name__)
        
        # Create OpenTelemetry instruments
        self.otel_counters = {}
        self.otel_histograms = {}
        self.otel_gauges = {}
    
    def add_collector(self, name: str, collector: BaseMetricsCollector):
        """Add a metrics collector."""
        self.collectors[name] = collector
    
    def get_collector(self, name: str) -> Optional[BaseMetricsCollector]:
        """Get a metrics collector by name."""
        return self.collectors.get(name)
    
    def start_collection(self, interval: int = 60):
        """Start periodic metrics collection."""
        self._collection_interval = interval
        self._running = True
        
        def collect_loop():
            while self._running:
                try:
                    self.collect_all()
                    time.sleep(self._collection_interval)
                except Exception as e:
                    # Log error but continue
                    print(f"Error in metrics collection: {e}")
                    time.sleep(self._collection_interval)
        
        self._collection_thread = threading.Thread(target=collect_loop, daemon=True)
        self._collection_thread.start()
    
    def stop_collection(self):
        """Stop periodic metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
    
    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect metrics from all collectors."""
        results = {}
        for name, collector in self.collectors.items():
            try:
                results[name] = collector.collect()
            except Exception as e:
                print(f"Error collecting metrics from {name}: {e}")
                results[name] = []
        
        return results
    
    def get_metric_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[HistogramStats]:
        """Get statistical summary for a metric."""
        return self.storage.get_stats(name, tags)
    
    def get_recent_metrics(self, name: str, tags: Optional[Dict[str, str]] = None, 
                          limit: int = 100) -> List[MetricValue]:
        """Get recent values for a metric."""
        return self.storage.get_recent(name, tags, limit)
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            metric = MetricValue(
                name=name,
                value=duration,
                unit=MetricUnit.MILLISECONDS,
                tags=tags or {}
            )
            self.storage.record(metric)
    
    def record_counter(self, name: str, value: Union[int, float] = 1, 
                      tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        metric = MetricValue(
            name=name,
            value=value,
            unit=MetricUnit.COUNT,
            tags=tags or {}
        )
        self.storage.record(metric)
    
    def record_gauge(self, name: str, value: Union[int, float], 
                    unit: MetricUnit = MetricUnit.DIMENSIONLESS,
                    tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        metric = MetricValue(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        self.storage.record(metric)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for dashboards."""
        data = {}
        
        # System metrics summary
        system_metrics = self.get_collector("system")
        if system_metrics:
            recent_system = system_metrics.collect()
            data["system"] = {metric.name.split(".")[-1]: metric.value for metric in recent_system}
        
        # Trading metrics summary
        trading_metrics = self.get_collector("trading")
        if trading_metrics:
            recent_trading = trading_metrics.collect()
            data["trading"] = {metric.name.split(".")[-1]: metric.value for metric in recent_trading}
        
        return data


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def initialize_metrics(enable_otel: bool = True, collection_interval: int = 60) -> MetricsCollector:
    """Initialize the global metrics system."""
    global _global_collector
    _global_collector = MetricsCollector(enable_otel=enable_otel)
    _global_collector.start_collection(collection_interval)
    return _global_collector