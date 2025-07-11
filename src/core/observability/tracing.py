"""
OpenTelemetry distributed tracing system for Quantum Trading Platform.

Provides comprehensive tracing for:
- Request flow across microservices
- Trading operations and latency tracking
- Database queries and external API calls
- Performance bottleneck identification
- Distributed debugging capabilities
"""

import functools
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union, Callable, List
from dataclasses import dataclass
from enum import Enum

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Fallback types
    class Span:
        pass
    class Status:
        pass
    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"


class SpanType(Enum):
    """Types of spans for categorization."""
    HTTP_REQUEST = "http.request"
    DATABASE_QUERY = "db.query"
    EXTERNAL_API = "external.api"
    TRADING_OPERATION = "trading.operation"
    MARKET_DATA = "market.data"
    STRATEGY_EXECUTION = "strategy.execution"
    RISK_CALCULATION = "risk.calculation"
    INTERNAL_FUNCTION = "internal.function"


@dataclass
class SpanContext:
    """Container for span context and metadata."""
    operation_name: str
    span_type: SpanType
    tags: Dict[str, Any]
    service_name: str = "quantum-trading"
    
    def to_attributes(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry attributes."""
        attrs = dict(self.tags)
        attrs.update({
            "service.name": self.service_name,
            "span.type": self.span_type.value,
            "operation.name": self.operation_name,
        })
        return attrs


class TracingBackend(ABC):
    """Abstract base for tracing backends."""
    
    @abstractmethod
    def create_span(self, name: str, context: SpanContext) -> Any:
        """Create a new span."""
        pass
    
    @abstractmethod
    def finish_span(self, span: Any, status: str = "OK", error: Optional[Exception] = None):
        """Finish a span."""
        pass


class OpenTelemetryBackend(TracingBackend):
    """OpenTelemetry tracing backend."""
    
    def __init__(self, tracer: Any):
        self.tracer = tracer
    
    def create_span(self, name: str, context: SpanContext) -> Span:
        """Create a new OpenTelemetry span."""
        span = self.tracer.start_span(name)
        
        # Set attributes
        for key, value in context.to_attributes().items():
            span.set_attribute(key, str(value))
        
        return span
    
    def finish_span(self, span: Span, status: str = "OK", error: Optional[Exception] = None):
        """Finish an OpenTelemetry span."""
        if error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        else:
            span.set_status(Status(StatusCode.OK))
        
        span.end()


class NoOpBackend(TracingBackend):
    """No-op tracing backend for when OpenTelemetry is not available."""
    
    def create_span(self, name: str, context: SpanContext) -> dict:
        """Create a no-op span."""
        return {
            "name": name,
            "start_time": time.time(),
            "context": context
        }
    
    def finish_span(self, span: dict, status: str = "OK", error: Optional[Exception] = None):
        """Finish a no-op span."""
        span["end_time"] = time.time()
        span["duration"] = span["end_time"] - span["start_time"]
        span["status"] = status
        span["error"] = error


class TracingManager:
    """
    Central tracing manager for the application.
    
    Features:
    - OpenTelemetry integration with fallback
    - Trading-specific span types and tags
    - Automatic instrumentation decorators
    - Context propagation across async boundaries
    - Performance metrics integration
    """
    
    def __init__(self, service_name: str = "quantum-trading", service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self._initialized = False
        
        if OTEL_AVAILABLE:
            self._setup_opentelemetry()
            self.backend = OpenTelemetryBackend(self.tracer)
        else:
            self.backend = NoOpBackend()
            self.tracer = None
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing."""
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        
        # Configure exporters
        # Note: In production, you'd configure actual exporters
        # For now, we'll set up a basic console exporter
        try:
            # Try to setup OTLP exporter if endpoint is configured
            otlp_exporter = OTLPSpanExporter(
                endpoint="http://localhost:4317",
                insecure=True
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(span_processor)
        except:
            # Fallback to console exporter
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            provider.add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        self._initialized = True
    
    def is_available(self) -> bool:
        """Check if tracing is available and initialized."""
        return self._initialized and OTEL_AVAILABLE
    
    @contextmanager
    def span(self, name: str, span_type: SpanType = SpanType.INTERNAL_FUNCTION, 
             tags: Optional[Dict[str, Any]] = None, service_name: Optional[str] = None):
        """Create a traced span context manager."""
        context = SpanContext(
            operation_name=name,
            span_type=span_type,
            tags=tags or {},
            service_name=service_name or self.service_name
        )
        
        span = self.backend.create_span(name, context)
        start_time = time.time()
        
        try:
            yield span
            self.backend.finish_span(span, "OK")
        except Exception as e:
            self.backend.finish_span(span, "ERROR", e)
            raise
        finally:
            # Record timing metrics
            duration = (time.time() - start_time) * 1000  # milliseconds
            # Could integrate with metrics system here
    
    def trace_function(self, name: Optional[str] = None, span_type: SpanType = SpanType.INTERNAL_FUNCTION,
                      tags: Optional[Dict[str, Any]] = None):
        """Decorator to trace function calls."""
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.span(span_name, span_type, tags):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.span(span_name, span_type, tags):
                        return func(*args, **kwargs)
                return sync_wrapper
        return decorator
    
    def trace_trading_operation(self, operation: str, symbol: str, side: Optional[str] = None):
        """Specialized decorator for trading operations."""
        tags = {"trading.symbol": symbol}
        if side:
            tags["trading.side"] = side
        
        return self.trace_function(
            name=f"trading.{operation}",
            span_type=SpanType.TRADING_OPERATION,
            tags=tags
        )
    
    def trace_market_data(self, symbol: str, data_type: str):
        """Specialized decorator for market data operations."""
        tags = {
            "market.symbol": symbol,
            "market.data_type": data_type
        }
        
        return self.trace_function(
            name=f"market_data.{data_type}",
            span_type=SpanType.MARKET_DATA,
            tags=tags
        )
    
    def trace_strategy(self, strategy_name: str):
        """Specialized decorator for strategy execution."""
        tags = {"strategy.name": strategy_name}
        
        return self.trace_function(
            name=f"strategy.{strategy_name}",
            span_type=SpanType.STRATEGY_EXECUTION,
            tags=tags
        )
    
    def trace_database_query(self, query_type: str, table: Optional[str] = None):
        """Specialized decorator for database queries."""
        tags = {"db.operation": query_type}
        if table:
            tags["db.table"] = table
        
        return self.trace_function(
            name=f"db.{query_type}",
            span_type=SpanType.DATABASE_QUERY,
            tags=tags
        )
    
    def trace_external_api(self, service: str, endpoint: str):
        """Specialized decorator for external API calls."""
        tags = {
            "external.service": service,
            "external.endpoint": endpoint
        }
        
        return self.trace_function(
            name=f"external.{service}",
            span_type=SpanType.EXTERNAL_API,
            tags=tags
        )
    
    def add_span_attribute(self, key: str, value: Any):
        """Add attribute to current span."""
        if self.is_available():
            current_span = trace.get_current_span()
            if current_span.is_recording():
                current_span.set_attribute(key, str(value))
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if self.is_available():
            current_span = trace.get_current_span()
            if current_span.is_recording():
                current_span.add_event(name, attributes or {})
    
    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        if self.is_available():
            current_span = trace.get_current_span()
            if current_span.is_recording():
                return format(current_span.get_span_context().trace_id, "032x")
        return None
    
    def get_span_id(self) -> Optional[str]:
        """Get current span ID."""
        if self.is_available():
            current_span = trace.get_current_span()
            if current_span.is_recording():
                return format(current_span.get_span_context().span_id, "016x")
        return None


# Global tracing manager instance
_global_tracer: Optional[TracingManager] = None


def get_tracer() -> TracingManager:
    """Get the global tracing manager."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = TracingManager()
    return _global_tracer


def initialize_tracing(service_name: str = "quantum-trading", 
                      service_version: str = "1.0.0") -> TracingManager:
    """Initialize the global tracing system."""
    global _global_tracer
    _global_tracer = TracingManager(service_name, service_version)
    return _global_tracer


# Convenience decorators using global tracer
def trace_sync(name: Optional[str] = None, span_type: SpanType = SpanType.INTERNAL_FUNCTION,
               tags: Optional[Dict[str, Any]] = None):
    """Convenience decorator for synchronous function tracing."""
    return get_tracer().trace_function(name, span_type, tags)


def trace_async(name: Optional[str] = None, span_type: SpanType = SpanType.INTERNAL_FUNCTION,
                tags: Optional[Dict[str, Any]] = None):
    """Convenience decorator for asynchronous function tracing."""
    return get_tracer().trace_function(name, span_type, tags)


def trace_trading(operation: str, symbol: str, side: Optional[str] = None):
    """Convenience decorator for trading operations."""
    return get_tracer().trace_trading_operation(operation, symbol, side)


def trace_market_data(symbol: str, data_type: str):
    """Convenience decorator for market data operations."""
    return get_tracer().trace_market_data(symbol, data_type)


def trace_strategy(strategy_name: str):
    """Convenience decorator for strategy execution."""
    return get_tracer().trace_strategy(strategy_name)


def trace_db(query_type: str, table: Optional[str] = None):
    """Convenience decorator for database queries."""
    return get_tracer().trace_database_query(query_type, table)


def trace_api(service: str, endpoint: str):
    """Convenience decorator for external API calls."""
    return get_tracer().trace_external_api(service, endpoint)


# Import asyncio for coroutine detection
import asyncio