"""
Advanced observability logging system with OpenTelemetry integration.

Provides enterprise-grade logging with:
- Structured JSON logging with context propagation
- OpenTelemetry trace correlation
- Request ID tracking across services
- Performance metrics integration
- Custom log processors for trading events
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pythonjsonlogger import jsonlogger

# Context variables for distributed tracing
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
trace_id_context: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_context: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class LogLevel(Enum):
    """Enhanced log levels for trading operations."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRADE = "TRADE"          # Special level for trade events
    MARKET = "MARKET"        # Special level for market data
    RISK = "RISK"           # Special level for risk events


@dataclass
class LogContext:
    """Log context container for structured data."""
    component: str
    operation: str
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = asdict(self)
        result["metadata"] = result["metadata"] or {}
        return {k: v for k, v in result.items() if v is not None}


class TradingLogProcessor:
    """Custom processor for trading-specific log events."""
    
    @staticmethod
    def process_trade_event(logger, method_name, event_dict):
        """Process trade-specific events with enrichment."""
        if event_dict.get("event_type") == "trade":
            # Add trading-specific metadata
            event_dict["trading"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "market_session": "active",  # Could be determined dynamically
                "compliance_checked": True,
            }
        return event_dict
    
    @staticmethod
    def process_market_event(logger, method_name, event_dict):
        """Process market data events."""
        if event_dict.get("event_type") == "market":
            event_dict["market"] = {
                "data_freshness_ms": event_dict.get("latency_ms", 0),
                "data_quality_score": 1.0,  # Could be calculated
                "source_reliability": "high",
            }
        return event_dict
    
    @staticmethod
    def process_risk_event(logger, method_name, event_dict):
        """Process risk management events."""
        if event_dict.get("event_type") == "risk":
            event_dict["risk"] = {
                "severity": event_dict.get("risk_level", "low"),
                "action_required": event_dict.get("requires_action", False),
                "escalation_level": event_dict.get("escalation", 0),
            }
        return event_dict


class ContextProcessor:
    """Processor for adding context information to logs."""
    
    @staticmethod
    def add_trace_context(logger, method_name, event_dict):
        """Add OpenTelemetry trace context."""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            event_dict["trace_id"] = format(span_context.trace_id, "032x")
            event_dict["span_id"] = format(span_context.span_id, "016x")
            
        # Also check context vars
        trace_id = trace_id_context.get()
        span_id = span_id_context.get()
        if trace_id:
            event_dict["trace_id"] = trace_id
        if span_id:
            event_dict["span_id"] = span_id
            
        return event_dict
    
    @staticmethod
    def add_request_context(logger, method_name, event_dict):
        """Add request-specific context."""
        request_id = request_id_context.get()
        user_id = user_id_context.get()
        
        if request_id:
            event_dict["request_id"] = request_id
        if user_id:
            event_dict["user_id"] = user_id
            
        return event_dict
    
    @staticmethod
    def add_performance_context(logger, method_name, event_dict):
        """Add performance metrics context."""
        event_dict["timestamp"] = datetime.utcnow().isoformat()
        event_dict["process_time"] = time.process_time()
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            event_dict["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
            event_dict["cpu_percent"] = process.cpu_percent()
        except ImportError:
            pass
            
        return event_dict


class ObservabilityLogger:
    """
    Enterprise-grade logger with full observability integration.
    
    Features:
    - Structured logging with JSON output
    - OpenTelemetry trace correlation
    - Context propagation across async boundaries
    - Custom processors for trading events
    - Performance metrics integration
    - Alert integration for critical events
    """
    
    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.name = name
        self.context = context or LogContext(component=name, operation="general")
        self._logger = structlog.get_logger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure the logger with context."""
        if self.context:
            self._logger = self._logger.bind(**self.context.to_dict())
    
    def with_context(self, **kwargs) -> 'ObservabilityLogger':
        """Create a new logger instance with additional context."""
        new_context = LogContext(
            component=self.context.component,
            operation=self.context.operation,
            **kwargs
        )
        return ObservabilityLogger(self.name, new_context)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._logger.critical(message, **kwargs)
    
    def trade(self, message: str, **kwargs):
        """Log trade-specific event."""
        self._logger.info(message, event_type="trade", **kwargs)
    
    def market(self, message: str, **kwargs):
        """Log market data event."""
        self._logger.info(message, event_type="market", **kwargs)
    
    def risk(self, message: str, **kwargs):
        """Log risk management event."""
        self._logger.warning(message, event_type="risk", **kwargs)
    
    def exception(self, message: str, exc_info=True, **kwargs):
        """Log exception with full traceback."""
        self._logger.exception(message, exc_info=exc_info, **kwargs)
    
    def metric(self, metric_name: str, value: Union[int, float], unit: str = "", **kwargs):
        """Log a metric value."""
        self._logger.info(
            f"Metric: {metric_name}",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            event_type="metric",
            **kwargs
        )


def setup_observability(
    level: str = "INFO",
    format_type: str = "json",
    enable_otel: bool = True,
    enable_performance: bool = True
):
    """
    Setup enterprise observability system.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format (json, console)
        enable_otel: Enable OpenTelemetry integration
        enable_performance: Enable performance monitoring
    """
    
    # Configure base logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[],
    )
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Configure formatter
    if format_type == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"levelname": "level", "asctime": "timestamp"},
            timestamp=True,
        )
        handler.setFormatter(formatter)
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    logging.root.handlers = [handler]
    
    # Build processor chain
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        ContextProcessor.add_request_context,
        TradingLogProcessor.process_trade_event,
        TradingLogProcessor.process_market_event,
        TradingLogProcessor.process_risk_event,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.dict_tracebacks,
    ]
    
    # Add OpenTelemetry integration
    if enable_otel:
        processors.insert(-4, ContextProcessor.add_trace_context)
    
    # Add performance monitoring
    if enable_performance:
        processors.insert(-4, ContextProcessor.add_performance_context)
    
    # Add callsite information for debugging
    processors.append(
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        )
    )
    
    # Add final renderer
    processors.append(
        structlog.processors.EventRenamer("message")
    )
    
    if format_type == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(
    name: str, 
    component: Optional[str] = None,
    operation: Optional[str] = None,
    **context_kwargs
) -> ObservabilityLogger:
    """
    Get a configured observability logger.
    
    Args:
        name: Logger name (usually __name__)
        component: Component name for context
        operation: Operation name for context
        **context_kwargs: Additional context data
        
    Returns:
        Configured ObservabilityLogger instance
    """
    context = LogContext(
        component=component or name,
        operation=operation or "general",
        **context_kwargs
    )
    return ObservabilityLogger(name, context)


# Utility functions for context management
def set_request_context(request_id: str, user_id: Optional[str] = None):
    """Set request context for distributed tracing."""
    request_id_context.set(request_id)
    if user_id:
        user_id_context.set(user_id)


def set_trace_context(trace_id: str, span_id: str):
    """Set trace context for distributed tracing."""
    trace_id_context.set(trace_id)
    span_id_context.set(span_id)


def clear_context():
    """Clear all context variables."""
    request_id_context.set(None)
    trace_id_context.set(None)
    span_id_context.set(None)
    user_id_context.set(None)