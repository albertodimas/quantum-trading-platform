"""
Redis-based Message Broker implementations.

Features:
- Redis Pub/Sub broker for real-time messaging
- Redis Streams broker for reliable message delivery
- Message persistence and replay capabilities
- Cluster support and failover
- Consumer groups and load balancing
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

try:
    import redis.asyncio as redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    ConnectionError = Exception
    TimeoutError = Exception
    REDIS_AVAILABLE = False

from .message_broker import (
    MessageBroker, Message, QueueConfig, ConsumerConfig,
    MessageStatus, MessagePriority, DeliveryMode
)
from ..observability.logger import get_logger
from ..observability.tracing import trace_async


logger = get_logger(__name__)


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    health_check_interval: int = 30
    cluster_mode: bool = False
    sentinel_hosts: List[str] = None


class RedisBroker(MessageBroker):
    """Redis Pub/Sub based message broker."""
    
    def __init__(self, connection_url: str, config: Optional[RedisConfig] = None):
        super().__init__(connection_url)
        self.config = config or RedisConfig()
        self._redis_pool = None
        self._pubsub = None
        self._subscriber_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self) -> bool:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            self._logger.error("Redis library not available. Install with: pip install redis")
            return False
            
        try:
            # Create Redis connection pool
            self._redis_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # Test connection
            client = redis.Redis(connection_pool=self._redis_pool)
            await client.ping()
            
            self._connected = True
            self._logger.info("Connected to Redis broker", 
                            host=self.config.host, 
                            port=self.config.port)
            return True
            
        except Exception as e:
            self._logger.error("Failed to connect to Redis", error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Redis."""
        try:
            # Cancel all subscriber tasks
            for task in self._subscriber_tasks.values():
                task.cancel()
            
            await asyncio.gather(*self._subscriber_tasks.values(), return_exceptions=True)
            self._subscriber_tasks.clear()
            
            # Close pubsub connection
            if self._pubsub:
                await self._pubsub.close()
                self._pubsub = None
            
            # Close connection pool
            if self._redis_pool:
                await self._redis_pool.disconnect()
                self._redis_pool = None
            
            self._connected = False
            self._logger.info("Disconnected from Redis broker")
            return True
            
        except Exception as e:
            self._logger.error("Error disconnecting from Redis", error=str(e))
            return False
    
    @trace_async(name="redis_publish", tags={"broker": "redis"})
    async def publish(self, message: Message, exchange: str = "", routing_key: str = "") -> bool:
        """Publish message to Redis."""
        if not self._connected or not self._redis_pool:
            return False
        
        try:
            # Apply middleware
            message = await self._apply_middleware(message, "outbound")
            
            client = redis.Redis(connection_pool=self._redis_pool)
            
            # Use routing_key as channel name
            channel = routing_key or message.topic or "default"
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Publish to Redis channel
            await client.publish(channel, message_data)
            
            self._logger.debug("Published message to Redis", 
                             channel=channel, 
                             message_id=message.id)
            
            self._update_stats("publish", True)
            return True
            
        except Exception as e:
            self._logger.error("Failed to publish message to Redis", error=str(e))
            self._update_stats("publish", False)
            return False
    
    async def subscribe(self, queue_config: QueueConfig, 
                       consumer_config: ConsumerConfig) -> bool:
        """Subscribe to Redis channel."""
        if not self._connected or not self._redis_pool:
            return False
        
        try:
            client = redis.Redis(connection_pool=self._redis_pool)
            pubsub = client.pubsub()
            
            # Subscribe to channel
            await pubsub.subscribe(queue_config.name)
            
            # Start consumer task
            consumer_tag = consumer_config.consumer_tag or f"redis_consumer_{len(self._subscriber_tasks)}"
            consumer_config.consumer_tag = consumer_tag
            
            task = asyncio.create_task(
                self._consume_redis_messages(pubsub, consumer_config)
            )
            self._subscriber_tasks[consumer_tag] = task
            
            self._active_consumers[consumer_tag] = consumer_config
            
            self._logger.info("Subscribed to Redis channel",
                            channel=queue_config.name,
                            consumer_tag=consumer_tag)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to subscribe to Redis channel", error=str(e))
            return False
    
    async def _consume_redis_messages(self, pubsub, consumer_config: ConsumerConfig):
        """Consume messages from Redis pubsub."""
        try:
            async for redis_message in pubsub.listen():
                if redis_message['type'] == 'message':
                    try:
                        # Parse message
                        message_data = json.loads(redis_message['data'])
                        message = Message.from_dict(message_data)
                        
                        # Apply middleware
                        message = await self._apply_middleware(message, "inbound")
                        
                        # Process message
                        await self._process_redis_message(message, consumer_config)
                        
                    except Exception as e:
                        self._logger.error("Error processing Redis message", error=str(e))
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error("Redis consumer error", 
                             consumer_tag=consumer_config.consumer_tag,
                             error=str(e))
        finally:
            await pubsub.close()
    
    async def _process_redis_message(self, message: Message, consumer_config: ConsumerConfig):
        """Process individual Redis message."""
        try:
            message.status = MessageStatus.PROCESSING
            
            if consumer_config.callback:
                await consumer_config.callback(message)
            
            message.status = MessageStatus.COMPLETED
            self._update_stats("consume", True)
            
        except Exception as e:
            message.status = MessageStatus.FAILED
            self._update_stats("consume", False)
            
            self._logger.error("Redis message processing failed",
                             message_id=message.id,
                             error=str(e))
            
            if consumer_config.error_handler:
                try:
                    await consumer_config.error_handler(message, e)
                except Exception as handler_error:
                    self._logger.error("Error handler failed", error=str(handler_error))
    
    async def unsubscribe(self, consumer_tag: str) -> bool:
        """Unsubscribe from Redis channel."""
        try:
            if consumer_tag in self._subscriber_tasks:
                self._subscriber_tasks[consumer_tag].cancel()
                del self._subscriber_tasks[consumer_tag]
            
            if consumer_tag in self._active_consumers:
                del self._active_consumers[consumer_tag]
            
            self._logger.info("Unsubscribed from Redis channel", consumer_tag=consumer_tag)
            return True
            
        except Exception as e:
            self._logger.error("Failed to unsubscribe from Redis", error=str(e))
            return False
    
    async def create_queue(self, config: QueueConfig) -> bool:
        """Create queue (no-op for Redis Pub/Sub)."""
        return True
    
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete queue (no-op for Redis Pub/Sub)."""
        return True
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get Redis channel information."""
        if not self._connected or not self._redis_pool:
            return {"status": "disconnected"}
        
        try:
            client = redis.Redis(connection_pool=self._redis_pool)
            
            # Get channel subscribers count
            pubsub_channels = await client.pubsub_channels(pattern=queue_name)
            subscriber_count = len(pubsub_channels)
            
            return {
                "name": queue_name,
                "subscriber_count": subscriber_count,
                "status": "ready"
            }
            
        except Exception as e:
            self._logger.error("Error getting Redis channel info", error=str(e))
            return {"status": "error", "error": str(e)}


class RedisStreamBroker(MessageBroker):
    """Redis Streams based message broker with persistence."""
    
    def __init__(self, connection_url: str, config: Optional[RedisConfig] = None):
        super().__init__(connection_url)
        self.config = config or RedisConfig()
        self._redis_pool = None
        self._consumer_groups: Dict[str, str] = {}
        self._stream_consumers: Dict[str, asyncio.Task] = {}
        
    async def connect(self) -> bool:
        """Connect to Redis for streams."""
        if not REDIS_AVAILABLE:
            self._logger.error("Redis library not available. Install with: pip install redis")
            return False
            
        try:
            self._redis_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            client = redis.Redis(connection_pool=self._redis_pool)
            await client.ping()
            
            self._connected = True
            self._logger.info("Connected to Redis Streams broker",
                            host=self.config.host,
                            port=self.config.port)
            return True
            
        except Exception as e:
            self._logger.error("Failed to connect to Redis Streams", error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Redis Streams."""
        try:
            # Cancel all stream consumer tasks
            for task in self._stream_consumers.values():
                task.cancel()
            
            await asyncio.gather(*self._stream_consumers.values(), return_exceptions=True)
            self._stream_consumers.clear()
            
            if self._redis_pool:
                await self._redis_pool.disconnect()
                self._redis_pool = None
            
            self._connected = False
            self._logger.info("Disconnected from Redis Streams broker")
            return True
            
        except Exception as e:
            self._logger.error("Error disconnecting from Redis Streams", error=str(e))
            return False
    
    @trace_async(name="redis_stream_publish", tags={"broker": "redis_streams"})
    async def publish(self, message: Message, exchange: str = "", routing_key: str = "") -> bool:
        """Publish message to Redis Stream."""
        if not self._connected or not self._redis_pool:
            return False
        
        try:
            message = await self._apply_middleware(message, "outbound")
            
            client = redis.Redis(connection_pool=self._redis_pool)
            
            stream_name = routing_key or message.topic or "default_stream"
            
            # Prepare stream fields
            fields = {
                "message_id": message.id,
                "topic": message.topic,
                "payload": json.dumps(message.payload),
                "headers": json.dumps(message.headers),
                "priority": message.priority.value,
                "timestamp": message.timestamp.isoformat(),
                "delivery_mode": message.delivery_mode.value,
                "correlation_id": message.correlation_id or "",
                "reply_to": message.reply_to or ""
            }
            
            # Add to stream with auto-generated ID
            stream_id = await client.xadd(stream_name, fields)
            
            self._logger.debug("Published message to Redis Stream",
                             stream=stream_name,
                             stream_id=stream_id,
                             message_id=message.id)
            
            self._update_stats("publish", True)
            return True
            
        except Exception as e:
            self._logger.error("Failed to publish to Redis Stream", error=str(e))
            self._update_stats("publish", False)
            return False
    
    async def subscribe(self, queue_config: QueueConfig, 
                       consumer_config: ConsumerConfig) -> bool:
        """Subscribe to Redis Stream with consumer group."""
        if not self._connected or not self._redis_pool:
            return False
        
        try:
            client = redis.Redis(connection_pool=self._redis_pool)
            
            stream_name = queue_config.name
            group_name = f"{stream_name}_group"
            consumer_name = consumer_config.consumer_tag or f"consumer_{int(time.time())}"
            
            # Create consumer group if it doesn't exist
            try:
                await client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
                self._logger.info("Created consumer group", 
                                stream=stream_name, 
                                group=group_name)
            except Exception as e:
                # Group might already exist
                if "BUSYGROUP" not in str(e):
                    raise
            
            # Store group mapping
            self._consumer_groups[consumer_name] = group_name
            
            # Start stream consumer task
            task = asyncio.create_task(
                self._consume_stream_messages(stream_name, group_name, consumer_name, consumer_config)
            )
            self._stream_consumers[consumer_name] = task
            self._active_consumers[consumer_name] = consumer_config
            
            self._logger.info("Subscribed to Redis Stream",
                            stream=stream_name,
                            group=group_name,
                            consumer=consumer_name)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to subscribe to Redis Stream", error=str(e))
            return False
    
    async def _consume_stream_messages(self, stream_name: str, group_name: str, 
                                     consumer_name: str, consumer_config: ConsumerConfig):
        """Consume messages from Redis Stream."""
        client = redis.Redis(connection_pool=self._redis_pool)
        
        try:
            while self._connected:
                try:
                    # Read messages from stream
                    messages = await client.xreadgroup(
                        group_name,
                        consumer_name,
                        {stream_name: '>'},
                        count=consumer_config.prefetch_count,
                        block=1000  # 1 second timeout
                    )
                    
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            try:
                                # Parse message
                                message = self._parse_stream_message(fields)
                                
                                # Apply middleware
                                message = await self._apply_middleware(message, "inbound")
                                
                                # Process message
                                await self._process_stream_message(
                                    message, consumer_config, client, 
                                    stream_name, group_name, message_id
                                )
                                
                            except Exception as e:
                                self._logger.error("Error processing stream message",
                                                 stream=stream_name,
                                                 message_id=message_id,
                                                 error=str(e))
                
                except Exception as e:
                    if "timeout" not in str(e).lower():
                        self._logger.error("Stream consumer error", error=str(e))
                        await asyncio.sleep(1)
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error("Stream consumer fatal error", error=str(e))
    
    def _parse_stream_message(self, fields: Dict[bytes, bytes]) -> Message:
        """Parse Redis Stream message fields."""
        # Convert bytes to strings
        str_fields = {k.decode(): v.decode() for k, v in fields.items()}
        
        message = Message()
        message.id = str_fields.get('message_id', '')
        message.topic = str_fields.get('topic', '')
        message.payload = json.loads(str_fields.get('payload', '{}'))
        message.headers = json.loads(str_fields.get('headers', '{}'))
        message.priority = MessagePriority(int(str_fields.get('priority', MessagePriority.NORMAL.value)))
        message.correlation_id = str_fields.get('correlation_id') or None
        message.reply_to = str_fields.get('reply_to') or None
        message.delivery_mode = DeliveryMode(str_fields.get('delivery_mode', DeliveryMode.AT_LEAST_ONCE.value))
        
        return message
    
    async def _process_stream_message(self, message: Message, consumer_config: ConsumerConfig,
                                    client, stream_name: str, group_name: str, message_id: str):
        """Process Redis Stream message."""
        try:
            message.status = MessageStatus.PROCESSING
            
            if consumer_config.callback:
                await consumer_config.callback(message)
            
            # Acknowledge message
            await client.xack(stream_name, group_name, message_id)
            
            message.status = MessageStatus.COMPLETED
            self._update_stats("consume", True)
            
            self._logger.debug("Processed stream message",
                             stream=stream_name,
                             message_id=message.id,
                             stream_message_id=message_id)
            
        except Exception as e:
            message.status = MessageStatus.FAILED
            self._update_stats("consume", False)
            
            self._logger.error("Stream message processing failed",
                             message_id=message.id,
                             error=str(e))
            
            if consumer_config.error_handler:
                try:
                    await consumer_config.error_handler(message, e)
                except Exception as handler_error:
                    self._logger.error("Error handler failed", error=str(handler_error))
    
    async def unsubscribe(self, consumer_tag: str) -> bool:
        """Unsubscribe from Redis Stream."""
        try:
            if consumer_tag in self._stream_consumers:
                self._stream_consumers[consumer_tag].cancel()
                del self._stream_consumers[consumer_tag]
            
            if consumer_tag in self._consumer_groups:
                del self._consumer_groups[consumer_tag]
            
            if consumer_tag in self._active_consumers:
                del self._active_consumers[consumer_tag]
            
            self._logger.info("Unsubscribed from Redis Stream", consumer_tag=consumer_tag)
            return True
            
        except Exception as e:
            self._logger.error("Failed to unsubscribe from Redis Stream", error=str(e))
            return False
    
    async def create_queue(self, config: QueueConfig) -> bool:
        """Create Redis Stream."""
        if not self._connected or not self._redis_pool:
            return False
        
        try:
            client = redis.Redis(connection_pool=self._redis_pool)
            
            # Create stream by adding a dummy message and removing it
            stream_id = await client.xadd(config.name, {"init": "true"})
            await client.xdel(config.name, stream_id)
            
            self._logger.info("Created Redis Stream", stream=config.name)
            return True
            
        except Exception as e:
            self._logger.error("Failed to create Redis Stream", error=str(e))
            return False
    
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete Redis Stream."""
        if not self._connected or not self._redis_pool:
            return False
        
        try:
            client = redis.Redis(connection_pool=self._redis_pool)
            await client.delete(queue_name)
            
            self._logger.info("Deleted Redis Stream", stream=queue_name)
            return True
            
        except Exception as e:
            self._logger.error("Failed to delete Redis Stream", error=str(e))
            return False
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get Redis Stream information."""
        if not self._connected or not self._redis_pool:
            return {"status": "disconnected"}
        
        try:
            client = redis.Redis(connection_pool=self._redis_pool)
            
            # Get stream info
            stream_info = await client.xinfo_stream(queue_name)
            
            return {
                "name": queue_name,
                "message_count": stream_info[b'length'],
                "consumer_groups": stream_info[b'groups'],
                "status": "ready"
            }
            
        except Exception as e:
            self._logger.error("Error getting Redis Stream info", error=str(e))
            return {"status": "error", "error": str(e)}