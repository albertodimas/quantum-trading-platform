"""
Enterprise Message Queue System for Quantum Trading Platform.

This module provides advanced asynchronous messaging capabilities including:
- Multi-broker support (Redis, RabbitMQ, Kafka)
- Event-driven architecture with pattern matching
- Message persistence and reliability guarantees
- Dead letter queues and retry mechanisms
- Real-time trading event processing
- Distributed task orchestration
"""

from .message_broker import MessageBroker, get_message_broker
from .redis_broker import RedisBroker, RedisStreamBroker
from .rabbitmq_broker import RabbitMQBroker
from .event_bus import EventBus, Event, EventHandler
from .patterns import (
    PublishSubscribePattern,
    RequestResponsePattern,
    WorkerQueuePattern,
    PriorityQueuePattern
)
from .handlers import (
    TradingEventHandler,
    MarketDataHandler,
    RiskEventHandler,
    SystemEventHandler
)
from .serializers import (
    JsonMessageSerializer,
    ProtobufMessageSerializer,
    AvroMessageSerializer
)
from .middleware import (
    MessageMiddleware,
    AuthenticationMiddleware,
    CompressionMiddleware,
    EncryptionMiddleware,
    LoggingMiddleware,
    MetricsMiddleware
)
from .reliability import (
    MessageReliability,
    RetryPolicy,
    DeadLetterQueue,
    MessagePersistence
)

__all__ = [
    # Core
    "MessageBroker",
    "get_message_broker",
    
    # Brokers
    "RedisBroker",
    "RedisStreamBroker",
    "RabbitMQBroker",
    
    # Event System
    "EventBus",
    "Event",
    "EventHandler",
    
    # Patterns
    "PublishSubscribePattern",
    "RequestResponsePattern",
    "WorkerQueuePattern",
    "PriorityQueuePattern",
    
    # Handlers
    "TradingEventHandler",
    "MarketDataHandler",
    "RiskEventHandler",
    "SystemEventHandler",
    
    # Serialization
    "JsonMessageSerializer",
    "ProtobufMessageSerializer",
    "AvroMessageSerializer",
    
    # Middleware
    "MessageMiddleware",
    "AuthenticationMiddleware",
    "CompressionMiddleware",
    "EncryptionMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    
    # Reliability
    "MessageReliability",
    "RetryPolicy",
    "DeadLetterQueue",
    "MessagePersistence",
]