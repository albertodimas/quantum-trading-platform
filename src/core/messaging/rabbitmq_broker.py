"""
RabbitMQ Message Broker implementation.

Features:
- Full AMQP 0.9.1 protocol support
- Exchange and queue management
- Message persistence and durability
- Dead letter queues and TTL
- Consumer acknowledgments and prefetch
- Connection recovery and clustering
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass

try:
    import aio_pika
    from aio_pika import Message as AMQPMessage, DeliveryMode as AMQPDeliveryMode
    from aio_pika.exceptions import AMQPException, ConnectionClosed
    RABBITMQ_AVAILABLE = True
except ImportError:
    aio_pika = None
    AMQPMessage = None
    AMQPDeliveryMode = None
    AMQPException = Exception
    ConnectionClosed = Exception
    RABBITMQ_AVAILABLE = False

from .message_broker import (
    MessageBroker, Message, QueueConfig, ConsumerConfig,
    MessageStatus, MessagePriority, DeliveryMode
)
from ..observability.logger import get_logger
from ..observability.tracing import trace_async


logger = get_logger(__name__)


@dataclass
class RabbitMQConfig:
    """RabbitMQ connection configuration."""
    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    heartbeat: int = 60
    connection_timeout: int = 10
    max_channels: int = 100
    confirm_delivery: bool = True
    recovery_interval: int = 5
    cluster_nodes: List[str] = None


class RabbitMQBroker(MessageBroker):
    """RabbitMQ AMQP message broker implementation."""
    
    def __init__(self, connection_url: str, config: Optional[RabbitMQConfig] = None):
        super().__init__(connection_url)
        self.config = config or RabbitMQConfig()
        self._connection = None
        self._channel = None
        self._exchanges: Dict[str, Any] = {}
        self._queues: Dict[str, Any] = {}
        self._consumers: Dict[str, Any] = {}
        
    async def connect(self) -> bool:
        """Connect to RabbitMQ."""
        if not RABBITMQ_AVAILABLE:
            self._logger.error("aio-pika library not available. Install with: pip install aio-pika")
            return False
        
        try:
            # Build connection URL
            if self.connection_url.startswith("amqp://"):
                url = self.connection_url
            else:
                url = f"amqp://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.virtual_host}"
            
            # Connect to RabbitMQ
            self._connection = await aio_pika.connect_robust(
                url,
                heartbeat=self.config.heartbeat,
                connection_timeout=self.config.connection_timeout,
                recovery_interval=self.config.recovery_interval
            )
            
            # Create channel
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=1)
            
            if self.config.confirm_delivery:
                await self._channel.confirm_delivery()
            
            self._connected = True
            self._logger.info("Connected to RabbitMQ",
                            host=self.config.host,
                            port=self.config.port,
                            vhost=self.config.virtual_host)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to connect to RabbitMQ", error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from RabbitMQ."""
        try:
            # Cancel all consumers
            for consumer_tag, consumer in self._consumers.items():
                try:
                    await consumer.cancel()
                except Exception as e:
                    self._logger.warning("Error canceling consumer", 
                                       consumer_tag=consumer_tag, 
                                       error=str(e))
            
            self._consumers.clear()
            
            # Close channel and connection
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
            
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
            
            self._connected = False
            self._logger.info("Disconnected from RabbitMQ")
            return True
            
        except Exception as e:
            self._logger.error("Error disconnecting from RabbitMQ", error=str(e))
            return False
    
    @trace_async(name="rabbitmq_publish", tags={"broker": "rabbitmq"})
    async def publish(self, message: Message, exchange: str = "", routing_key: str = "") -> bool:
        """Publish message to RabbitMQ."""
        if not self._connected or not self._channel:
            return False
        
        try:
            # Apply middleware
            message = await self._apply_middleware(message, "outbound")
            
            # Convert delivery mode
            delivery_mode = AMQPDeliveryMode.PERSISTENT if message.delivery_mode == DeliveryMode.AT_LEAST_ONCE else AMQPDeliveryMode.NOT_PERSISTENT
            
            # Create AMQP message
            amqp_message = AMQPMessage(
                body=json.dumps(message.payload).encode(),
                headers={
                    **message.headers,
                    'message_id': message.id,
                    'topic': message.topic,
                    'priority': message.priority.value,
                    'timestamp': message.timestamp.isoformat(),
                    'correlation_id': message.correlation_id,
                    'reply_to': message.reply_to,
                    'retry_count': message.retry_count,
                    'max_retries': message.max_retries
                },
                delivery_mode=delivery_mode,
                priority=message.priority.value,
                message_id=message.id,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                expiration=int((message.expiry - message.timestamp).total_seconds() * 1000) if message.expiry else None
            )
            
            # Get or create exchange
            if exchange and exchange not in self._exchanges:
                await self._create_exchange(exchange)
            
            # Publish message
            await self._channel.default_exchange.publish(
                amqp_message,
                routing_key=routing_key or message.topic
            )
            
            self._logger.debug("Published message to RabbitMQ",
                             exchange=exchange,
                             routing_key=routing_key,
                             message_id=message.id)
            
            self._update_stats("publish", True)
            return True
            
        except Exception as e:
            self._logger.error("Failed to publish message to RabbitMQ", error=str(e))
            self._update_stats("publish", False)
            return False
    
    async def _create_exchange(self, exchange_name: str, exchange_type: str = "direct"):
        """Create RabbitMQ exchange."""
        try:
            exchange = await self._channel.declare_exchange(
                exchange_name,
                type=getattr(aio_pika.ExchangeType, exchange_type.upper()),
                durable=True
            )
            self._exchanges[exchange_name] = exchange
            self._logger.info("Created exchange", 
                            exchange=exchange_name, 
                            type=exchange_type)
            return exchange
            
        except Exception as e:
            self._logger.error("Failed to create exchange", 
                             exchange=exchange_name, 
                             error=str(e))
            raise
    
    async def subscribe(self, queue_config: QueueConfig, 
                       consumer_config: ConsumerConfig) -> bool:
        """Subscribe to RabbitMQ queue."""
        if not self._connected or not self._channel:
            return False
        
        try:
            # Create queue if it doesn't exist
            await self.create_queue(queue_config)
            
            # Get queue
            queue = self._queues[queue_config.name]
            
            # Set QoS for prefetch
            await self._channel.set_qos(
                prefetch_count=consumer_config.prefetch_count
            )
            
            # Start consuming
            consumer = await queue.consume(
                callback=lambda message: self._process_rabbitmq_message(message, consumer_config),
                consumer_tag=consumer_config.consumer_tag,
                exclusive=consumer_config.exclusive,
                no_ack=consumer_config.auto_ack
            )
            
            consumer_tag = consumer_config.consumer_tag or consumer.consumer_tag
            self._consumers[consumer_tag] = consumer
            self._active_consumers[consumer_tag] = consumer_config
            
            self._logger.info("Subscribed to RabbitMQ queue",
                            queue=queue_config.name,
                            consumer_tag=consumer_tag)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to subscribe to RabbitMQ queue", error=str(e))
            return False
    
    async def _process_rabbitmq_message(self, amqp_message, consumer_config: ConsumerConfig):
        """Process RabbitMQ message."""
        try:
            # Parse message
            message = self._parse_amqp_message(amqp_message)
            
            # Apply middleware
            message = await self._apply_middleware(message, "inbound")
            
            # Process message
            message.status = MessageStatus.PROCESSING
            
            if consumer_config.callback:
                await consumer_config.callback(message)
            
            # Acknowledge message if not auto-ack
            if not consumer_config.auto_ack:
                amqp_message.ack()
            
            message.status = MessageStatus.COMPLETED
            self._update_stats("consume", True)
            
            self._logger.debug("Processed RabbitMQ message",
                             message_id=message.id,
                             consumer_tag=consumer_config.consumer_tag)
            
        except Exception as e:
            message.status = MessageStatus.FAILED
            self._update_stats("consume", False)
            
            self._logger.error("RabbitMQ message processing failed",
                             message_id=getattr(message, 'id', 'unknown'),
                             error=str(e))
            
            # Handle error
            if consumer_config.error_handler:
                try:
                    await consumer_config.error_handler(message, e)
                except Exception as handler_error:
                    self._logger.error("Error handler failed", error=str(handler_error))
            
            # Reject message if not auto-ack
            if not consumer_config.auto_ack:
                # Requeue for retry if retries available
                if hasattr(message, 'should_retry') and message.should_retry():
                    amqp_message.reject(requeue=True)
                else:
                    amqp_message.reject(requeue=False)
    
    def _parse_amqp_message(self, amqp_message) -> Message:
        """Parse AMQP message to internal Message format."""
        headers = amqp_message.headers or {}
        
        message = Message()
        message.id = headers.get('message_id', amqp_message.message_id or '')
        message.topic = headers.get('topic', '')
        message.payload = json.loads(amqp_message.body.decode())
        message.headers = {k: v for k, v in headers.items() 
                          if k not in ['message_id', 'topic', 'priority', 'timestamp', 
                                     'correlation_id', 'reply_to', 'retry_count', 'max_retries']}
        message.priority = MessagePriority(headers.get('priority', MessagePriority.NORMAL.value))
        message.correlation_id = headers.get('correlation_id', amqp_message.correlation_id)
        message.reply_to = headers.get('reply_to', amqp_message.reply_to)
        message.retry_count = headers.get('retry_count', 0)
        message.max_retries = headers.get('max_retries', 3)
        
        # Set delivery mode
        if amqp_message.delivery_mode == AMQPDeliveryMode.PERSISTENT:
            message.delivery_mode = DeliveryMode.AT_LEAST_ONCE
        else:
            message.delivery_mode = DeliveryMode.AT_MOST_ONCE
        
        return message
    
    async def unsubscribe(self, consumer_tag: str) -> bool:
        """Unsubscribe from RabbitMQ queue."""
        try:
            if consumer_tag in self._consumers:
                consumer = self._consumers[consumer_tag]
                await consumer.cancel()
                del self._consumers[consumer_tag]
            
            if consumer_tag in self._active_consumers:
                del self._active_consumers[consumer_tag]
            
            self._logger.info("Unsubscribed from RabbitMQ queue", consumer_tag=consumer_tag)
            return True
            
        except Exception as e:
            self._logger.error("Failed to unsubscribe from RabbitMQ", error=str(e))
            return False
    
    async def create_queue(self, config: QueueConfig) -> bool:
        """Create RabbitMQ queue."""
        if not self._connected or not self._channel:
            return False
        
        try:
            # Create queue
            queue = await self._channel.declare_queue(
                config.name,
                durable=config.durable,
                auto_delete=config.auto_delete,
                arguments={
                    **({"x-max-length": config.max_length} if config.max_length else {}),
                    **({"x-message-ttl": config.message_ttl * 1000} if config.message_ttl else {}),
                    **({"x-dead-letter-exchange": config.dead_letter_exchange} if config.dead_letter_exchange else {}),
                    "x-max-priority": config.priority_levels
                }
            )
            
            self._queues[config.name] = queue
            
            self._logger.info("Created RabbitMQ queue",
                            queue=config.name,
                            durable=config.durable)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to create RabbitMQ queue", error=str(e))
            return False
    
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete RabbitMQ queue."""
        if not self._connected or not self._channel:
            return False
        
        try:
            if queue_name in self._queues:
                queue = self._queues[queue_name]
                await queue.delete()
                del self._queues[queue_name]
            
            self._logger.info("Deleted RabbitMQ queue", queue=queue_name)
            return True
            
        except Exception as e:
            self._logger.error("Failed to delete RabbitMQ queue", error=str(e))
            return False
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get RabbitMQ queue information."""
        if not self._connected or not self._channel:
            return {"status": "disconnected"}
        
        try:
            if queue_name in self._queues:
                queue = self._queues[queue_name]
                
                # Get queue declaration info
                declare_result = await self._channel.declare_queue(queue_name, passive=True)
                
                return {
                    "name": queue_name,
                    "message_count": declare_result.method.message_count,
                    "consumer_count": declare_result.method.consumer_count,
                    "status": "ready"
                }
            else:
                return {
                    "name": queue_name,
                    "status": "not_found"
                }
                
        except Exception as e:
            self._logger.error("Error getting RabbitMQ queue info", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def create_exchange_binding(self, queue_name: str, exchange_name: str, 
                                    routing_key: str = "") -> bool:
        """Create binding between queue and exchange."""
        try:
            if queue_name not in self._queues:
                raise ValueError(f"Queue {queue_name} not found")
            
            if exchange_name not in self._exchanges:
                await self._create_exchange(exchange_name)
            
            queue = self._queues[queue_name]
            exchange = self._exchanges[exchange_name]
            
            await queue.bind(exchange, routing_key=routing_key)
            
            self._logger.info("Created exchange binding",
                            queue=queue_name,
                            exchange=exchange_name,
                            routing_key=routing_key)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to create exchange binding", error=str(e))
            return False
    
    async def remove_exchange_binding(self, queue_name: str, exchange_name: str, 
                                    routing_key: str = "") -> bool:
        """Remove binding between queue and exchange."""
        try:
            if queue_name not in self._queues or exchange_name not in self._exchanges:
                return False
            
            queue = self._queues[queue_name]
            exchange = self._exchanges[exchange_name]
            
            await queue.unbind(exchange, routing_key=routing_key)
            
            self._logger.info("Removed exchange binding",
                            queue=queue_name,
                            exchange=exchange_name,
                            routing_key=routing_key)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to remove exchange binding", error=str(e))
            return False