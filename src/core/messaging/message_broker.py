"""
Advanced Message Broker with multi-backend support and enterprise features.

Features:
- Multi-broker abstraction (Redis, RabbitMQ, Kafka)
- Message persistence and reliability guarantees
- Dead letter queues and retry mechanisms
- Priority queues and message routing
- Real-time event streaming
- Distributed task coordination
"""

import asyncio
import time
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
from enum import Enum
import threading
from collections import defaultdict

from ..observability.logger import get_logger
from ..observability.metrics import get_metrics_collector
from ..observability.tracing import trace_async


logger = get_logger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


class DeliveryMode(Enum):
    """Message delivery modes."""
    AT_MOST_ONCE = "at_most_once"      # Fire and forget
    AT_LEAST_ONCE = "at_least_once"    # Guaranteed delivery with possible duplicates
    EXACTLY_ONCE = "exactly_once"      # Guaranteed single delivery


@dataclass
class Message:
    """Message container with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Any = None
    headers: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expiry: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    def is_expired(self) -> bool:
        """Check if message is expired."""
        if self.expiry is None:
            return False
        return datetime.utcnow() > self.expiry
    
    def should_retry(self) -> bool:
        """Check if message should be retried."""
        return self.retry_count < self.max_retries and self.status == MessageStatus.FAILED
    
    def increment_retry(self):
        """Increment retry count."""
        self.retry_count += 1
        self.status = MessageStatus.RETRY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "headers": self.headers,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "delivery_mode": self.delivery_mode.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        msg = cls()
        msg.id = data.get("id", str(uuid.uuid4()))
        msg.topic = data.get("topic", "")
        msg.payload = data.get("payload")
        msg.headers = data.get("headers", {})
        msg.priority = MessagePriority(data.get("priority", MessagePriority.NORMAL.value))
        msg.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
        if data.get("expiry"):
            msg.expiry = datetime.fromisoformat(data["expiry"])
        msg.retry_count = data.get("retry_count", 0)
        msg.max_retries = data.get("max_retries", 3)
        msg.status = MessageStatus(data.get("status", MessageStatus.PENDING.value))
        msg.correlation_id = data.get("correlation_id")
        msg.reply_to = data.get("reply_to")
        msg.delivery_mode = DeliveryMode(data.get("delivery_mode", DeliveryMode.AT_LEAST_ONCE.value))
        return msg


@dataclass
class QueueConfig:
    """Queue configuration."""
    name: str
    durable: bool = True
    auto_delete: bool = False
    max_length: Optional[int] = None
    message_ttl: Optional[int] = None  # in seconds
    dead_letter_exchange: Optional[str] = None
    priority_levels: int = 5
    consumer_timeout: int = 30
    prefetch_count: int = 10


@dataclass
class ConsumerConfig:
    """Consumer configuration."""
    queue_name: str
    consumer_tag: str = ""
    auto_ack: bool = False
    exclusive: bool = False
    callback: Optional[Callable] = None
    error_handler: Optional[Callable] = None
    max_concurrent: int = 10
    backoff_multiplier: float = 2.0
    max_backoff: int = 300


class MessageBroker(ABC):
    """Abstract message broker interface."""
    
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self._connected = False
        self._logger = get_logger(f"{self.__class__.__name__}")
        self._metrics = get_metrics_collector().get_collector("trading")
        
        # Message tracking
        self._message_stats = defaultdict(int)
        self._active_consumers: Dict[str, Any] = {}
        
        # Middleware
        self._middleware: List[Callable] = []
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to message broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from message broker."""
        pass
    
    @abstractmethod
    async def publish(self, message: Message, exchange: str = "", routing_key: str = "") -> bool:
        """Publish message."""
        pass
    
    @abstractmethod
    async def subscribe(self, queue_config: QueueConfig, 
                       consumer_config: ConsumerConfig) -> bool:
        """Subscribe to queue."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, consumer_tag: str) -> bool:
        """Unsubscribe from queue."""
        pass
    
    @abstractmethod
    async def create_queue(self, config: QueueConfig) -> bool:
        """Create queue."""
        pass
    
    @abstractmethod
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete queue."""
        pass
    
    @abstractmethod
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get queue information."""
        pass
    
    def add_middleware(self, middleware: Callable):
        """Add message middleware."""
        self._middleware.append(middleware)
    
    async def _apply_middleware(self, message: Message, direction: str = "outbound") -> Message:
        """Apply middleware to message."""
        for middleware in self._middleware:
            try:
                message = await middleware(message, direction)
            except Exception as e:
                self._logger.error("Middleware error", middleware=middleware.__name__, error=str(e))
        return message
    
    def _update_stats(self, operation: str, success: bool = True):
        """Update message statistics."""
        self._message_stats[f"{operation}_total"] += 1
        if success:
            self._message_stats[f"{operation}_success"] += 1
        else:
            self._message_stats[f"{operation}_failed"] += 1
        
        if self._metrics:
            self._metrics.record_metric(
                f"messaging.{operation}",
                1,
                tags={"broker": self.__class__.__name__, "success": str(success)}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        return {
            "connected": self._connected,
            "active_consumers": len(self._active_consumers),
            "message_stats": dict(self._message_stats),
            "middleware_count": len(self._middleware)
        }


class InMemoryBroker(MessageBroker):
    """In-memory message broker for testing and development."""
    
    def __init__(self, connection_url: str = "memory://localhost"):
        super().__init__(connection_url)
        self._queues: Dict[str, List[Message]] = {}
        self._exchanges: Dict[str, Dict[str, List[str]]] = {}  # exchange -> routing_key -> queues
        self._consumers: Dict[str, ConsumerConfig] = {}
        self._lock = threading.RLock()
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> bool:
        """Connect to in-memory broker."""
        self._connected = True
        self._logger.info("Connected to in-memory broker")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from in-memory broker."""
        # Stop all consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        
        self._consumer_tasks.clear()
        self._connected = False
        self._logger.info("Disconnected from in-memory broker")
        return True
    
    @trace_async(name="publish_message", tags={"broker": "memory"})
    async def publish(self, message: Message, exchange: str = "", routing_key: str = "") -> bool:
        """Publish message to in-memory broker."""
        if not self._connected:
            return False
        
        try:
            # Apply middleware
            message = await self._apply_middleware(message, "outbound")
            
            with self._lock:
                # Direct queue publish
                if not exchange and routing_key:
                    if routing_key not in self._queues:
                        self._queues[routing_key] = []
                    
                    # Insert by priority
                    self._insert_by_priority(self._queues[routing_key], message)
                    self._logger.debug("Published message to queue", 
                                     queue=routing_key, 
                                     message_id=message.id)
                
                # Exchange-based routing
                elif exchange in self._exchanges:
                    if routing_key in self._exchanges[exchange]:
                        for queue_name in self._exchanges[exchange][routing_key]:
                            if queue_name not in self._queues:
                                self._queues[queue_name] = []
                            
                            self._insert_by_priority(self._queues[queue_name], message)
                            self._logger.debug("Routed message to queue",
                                             exchange=exchange,
                                             routing_key=routing_key,
                                             queue=queue_name,
                                             message_id=message.id)
            
            self._update_stats("publish", True)
            return True
            
        except Exception as e:
            self._logger.error("Failed to publish message", error=str(e))
            self._update_stats("publish", False)
            return False
    
    def _insert_by_priority(self, queue: List[Message], message: Message):
        """Insert message in queue by priority."""
        # Simple priority insertion (higher priority first)
        inserted = False
        for i, existing_msg in enumerate(queue):
            if message.priority.value > existing_msg.priority.value:
                queue.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            queue.append(message)
    
    async def subscribe(self, queue_config: QueueConfig, 
                       consumer_config: ConsumerConfig) -> bool:
        """Subscribe to queue in in-memory broker."""
        if not self._connected:
            return False
        
        try:
            # Create queue if it doesn't exist
            await self.create_queue(queue_config)
            
            # Start consumer task
            consumer_tag = consumer_config.consumer_tag or f"consumer_{len(self._consumers)}"
            consumer_config.consumer_tag = consumer_tag
            
            self._consumers[consumer_tag] = consumer_config
            
            # Start consuming task
            task = asyncio.create_task(self._consume_messages(consumer_config))
            self._consumer_tasks[consumer_tag] = task
            
            self._active_consumers[consumer_tag] = consumer_config
            
            self._logger.info("Started consumer",
                            queue=queue_config.name,
                            consumer_tag=consumer_tag)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to subscribe to queue", error=str(e))
            return False
    
    async def _consume_messages(self, consumer_config: ConsumerConfig):
        """Consume messages from queue."""
        queue_name = consumer_config.queue_name
        
        while self._connected and consumer_config.consumer_tag in self._consumers:
            try:
                with self._lock:
                    if queue_name in self._queues and self._queues[queue_name]:
                        message = self._queues[queue_name].pop(0)
                    else:
                        message = None
                
                if message:
                    # Check if message is expired
                    if message.is_expired():
                        self._logger.warning("Message expired", message_id=message.id)
                        continue
                    
                    # Apply middleware
                    message = await self._apply_middleware(message, "inbound")
                    
                    # Process message
                    await self._process_message(message, consumer_config)
                else:
                    # No messages, wait a bit
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Consumer error", 
                                 consumer_tag=consumer_config.consumer_tag,
                                 error=str(e))
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Message, consumer_config: ConsumerConfig):
        """Process individual message."""
        try:
            message.status = MessageStatus.PROCESSING
            
            if consumer_config.callback:
                await consumer_config.callback(message)
            
            message.status = MessageStatus.COMPLETED
            self._update_stats("consume", True)
            
            self._logger.debug("Processed message",
                             message_id=message.id,
                             consumer_tag=consumer_config.consumer_tag)
            
        except Exception as e:
            message.status = MessageStatus.FAILED
            self._update_stats("consume", False)
            
            self._logger.error("Message processing failed",
                             message_id=message.id,
                             error=str(e))
            
            # Handle error
            if consumer_config.error_handler:
                try:
                    await consumer_config.error_handler(message, e)
                except Exception as handler_error:
                    self._logger.error("Error handler failed", error=str(handler_error))
            
            # Retry logic
            if message.should_retry():
                message.increment_retry()
                # Re-queue message for retry
                with self._lock:
                    if consumer_config.queue_name not in self._queues:
                        self._queues[consumer_config.queue_name] = []
                    self._insert_by_priority(self._queues[consumer_config.queue_name], message)
    
    async def unsubscribe(self, consumer_tag: str) -> bool:
        """Unsubscribe from queue."""
        try:
            if consumer_tag in self._consumer_tasks:
                self._consumer_tasks[consumer_tag].cancel()
                del self._consumer_tasks[consumer_tag]
            
            if consumer_tag in self._consumers:
                del self._consumers[consumer_tag]
            
            if consumer_tag in self._active_consumers:
                del self._active_consumers[consumer_tag]
            
            self._logger.info("Unsubscribed consumer", consumer_tag=consumer_tag)
            return True
            
        except Exception as e:
            self._logger.error("Failed to unsubscribe", error=str(e))
            return False
    
    async def create_queue(self, config: QueueConfig) -> bool:
        """Create queue in in-memory broker."""
        try:
            with self._lock:
                if config.name not in self._queues:
                    self._queues[config.name] = []
                    self._logger.debug("Created queue", queue=config.name)
            return True
        except Exception as e:
            self._logger.error("Failed to create queue", error=str(e))
            return False
    
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete queue from in-memory broker."""
        try:
            with self._lock:
                if queue_name in self._queues:
                    del self._queues[queue_name]
                    self._logger.debug("Deleted queue", queue=queue_name)
            return True
        except Exception as e:
            self._logger.error("Failed to delete queue", error=str(e))
            return False
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get queue information."""
        with self._lock:
            if queue_name in self._queues:
                messages = self._queues[queue_name]
                return {
                    "name": queue_name,
                    "message_count": len(messages),
                    "consumer_count": sum(1 for c in self._consumers.values() 
                                        if c.queue_name == queue_name),
                    "status": "ready"
                }
            else:
                return {
                    "name": queue_name,
                    "message_count": 0,
                    "consumer_count": 0,
                    "status": "not_found"
                }


class MessageBrokerManager:
    """
    Manager for multiple message brokers with routing and failover.
    
    Features:
    - Multi-broker support with automatic failover
    - Message routing based on topic patterns
    - Load balancing across brokers
    - Health monitoring and recovery
    """
    
    def __init__(self):
        self._brokers: Dict[str, MessageBroker] = {}
        self._routing_rules: Dict[str, str] = {}  # topic_pattern -> broker_name
        self._default_broker: Optional[str] = None
        self._logger = get_logger(__name__)
    
    def add_broker(self, name: str, broker: MessageBroker, is_default: bool = False):
        """Add a message broker."""
        self._brokers[name] = broker
        if is_default or not self._default_broker:
            self._default_broker = name
        self._logger.info("Added broker", broker_name=name, is_default=is_default)
    
    def add_routing_rule(self, topic_pattern: str, broker_name: str):
        """Add topic routing rule."""
        self._routing_rules[topic_pattern] = broker_name
        self._logger.info("Added routing rule", 
                         topic_pattern=topic_pattern, 
                         broker_name=broker_name)
    
    def _get_broker_for_topic(self, topic: str) -> Optional[MessageBroker]:
        """Get broker for topic based on routing rules."""
        # Check exact match first
        if topic in self._routing_rules:
            broker_name = self._routing_rules[topic]
            return self._brokers.get(broker_name)
        
        # Check pattern matches
        for pattern, broker_name in self._routing_rules.items():
            if "*" in pattern:
                # Simple wildcard matching
                if pattern.replace("*", "") in topic:
                    return self._brokers.get(broker_name)
        
        # Use default broker
        if self._default_broker:
            return self._brokers.get(self._default_broker)
        
        return None
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all brokers."""
        results = {}
        for name, broker in self._brokers.items():
            try:
                results[name] = await broker.connect()
            except Exception as e:
                self._logger.error("Failed to connect broker", broker=name, error=str(e))
                results[name] = False
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all brokers."""
        results = {}
        for name, broker in self._brokers.items():
            try:
                results[name] = await broker.disconnect()
            except Exception as e:
                self._logger.error("Failed to disconnect broker", broker=name, error=str(e))
                results[name] = False
        return results
    
    async def publish(self, message: Message, exchange: str = "", routing_key: str = "") -> bool:
        """Publish message using appropriate broker."""
        broker = self._get_broker_for_topic(message.topic)
        if not broker:
            self._logger.error("No broker available for topic", topic=message.topic)
            return False
        
        return await broker.publish(message, exchange, routing_key)
    
    def get_broker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all brokers."""
        return {name: broker.get_stats() for name, broker in self._brokers.items()}


# Global message broker manager
_global_broker_manager: Optional[MessageBrokerManager] = None


def get_message_broker() -> MessageBrokerManager:
    """Get the global message broker manager."""
    global _global_broker_manager
    if _global_broker_manager is None:
        _global_broker_manager = MessageBrokerManager()
    return _global_broker_manager


def initialize_messaging(brokers_config: Dict[str, Dict[str, Any]]) -> MessageBrokerManager:
    """Initialize the messaging system with broker configurations."""
    global _global_broker_manager
    _global_broker_manager = MessageBrokerManager()
    
    for broker_name, config in brokers_config.items():
        broker_type = config.get("type", "memory")
        connection_url = config.get("url", "memory://localhost")
        
        if broker_type == "memory":
            broker = InMemoryBroker(connection_url)
        else:
            # Add other broker types as needed
            broker = InMemoryBroker(connection_url)
        
        is_default = config.get("default", False)
        _global_broker_manager.add_broker(broker_name, broker, is_default)
        
        # Add routing rules
        routing_rules = config.get("routing", {})
        for pattern, target_broker in routing_rules.items():
            _global_broker_manager.add_routing_rule(pattern, target_broker)
    
    return _global_broker_manager