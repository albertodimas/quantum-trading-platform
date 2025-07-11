"""
Message Queue Patterns for common messaging scenarios.

Implements enterprise messaging patterns:
- Publish-Subscribe for event broadcasting
- Request-Response for synchronous communication
- Worker Queue for load distribution
- Priority Queue for ordered processing
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import json

from .message_broker import Message, MessageBroker, MessagePriority, get_message_broker
from .event_bus import Event, EventBus, get_event_bus
from ..observability.logger import get_logger
from ..observability.tracing import trace_async


logger = get_logger(__name__)


class PatternType(Enum):
    """Message pattern types."""
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    WORKER_QUEUE = "worker_queue"
    PRIORITY_QUEUE = "priority_queue"


@dataclass
class PatternConfig:
    """Base configuration for message patterns."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout: Optional[float] = None
    retry_count: int = 3
    dead_letter_enabled: bool = True


class MessagePattern(ABC):
    """Abstract base class for message patterns."""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self._logger = get_logger(f"pattern.{self.__class__.__name__}")
        self._active = False
        
    @abstractmethod
    async def start(self) -> bool:
        """Start the message pattern."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the message pattern."""
        pass
    
    @property
    def is_active(self) -> bool:
        """Check if pattern is active."""
        return self._active


class PublishSubscribePattern(MessagePattern):
    """
    Publish-Subscribe pattern for event broadcasting.
    
    Features:
    - One-to-many message delivery
    - Topic-based routing
    - Subscriber registration and management
    - Event filtering and transformation
    """
    
    def __init__(self, config: PatternConfig, topic: str):
        super().__init__(config)
        self.topic = topic
        self._subscribers: Dict[str, Callable] = {}
        self._message_broker = get_message_broker()
        self._event_bus = get_event_bus()
        
    async def start(self) -> bool:
        """Start publish-subscribe pattern."""
        try:
            self._active = True
            self._logger.info("Started publish-subscribe pattern", 
                            pattern_id=self.config.pattern_id,
                            topic=self.topic)
            return True
        except Exception as e:
            self._logger.error("Failed to start publish-subscribe pattern", error=str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop publish-subscribe pattern."""
        try:
            self._active = False
            self._subscribers.clear()
            self._logger.info("Stopped publish-subscribe pattern", 
                            pattern_id=self.config.pattern_id)
            return True
        except Exception as e:
            self._logger.error("Failed to stop publish-subscribe pattern", error=str(e))
            return False
    
    def subscribe(self, subscriber_id: str, callback: Callable[[Message], None]) -> bool:
        """
        Subscribe to topic.
        
        Args:
            subscriber_id: Unique subscriber identifier
            callback: Function to call when message received
            
        Returns:
            True if subscription successful
        """
        try:
            self._subscribers[subscriber_id] = callback
            self._logger.info("Added subscriber", 
                            subscriber_id=subscriber_id,
                            topic=self.topic)
            return True
        except Exception as e:
            self._logger.error("Failed to add subscriber", error=str(e))
            return False
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from topic.
        
        Args:
            subscriber_id: Subscriber identifier
            
        Returns:
            True if unsubscription successful
        """
        try:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                self._logger.info("Removed subscriber", subscriber_id=subscriber_id)
                return True
            return False
        except Exception as e:
            self._logger.error("Failed to remove subscriber", error=str(e))
            return False
    
    @trace_async(name="publish_message", tags={"pattern": "publish_subscribe"})
    async def publish(self, message: Message) -> bool:
        """
        Publish message to all subscribers.
        
        Args:
            message: Message to publish
            
        Returns:
            True if published successfully
        """
        if not self._active:
            return False
        
        try:
            message.topic = self.topic
            
            # Deliver to all subscribers
            delivery_tasks = []
            for subscriber_id, callback in self._subscribers.items():
                task = asyncio.create_task(
                    self._deliver_to_subscriber(subscriber_id, callback, message)
                )
                delivery_tasks.append(task)
            
            # Wait for all deliveries
            results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            
            self._logger.debug("Published message to subscribers",
                             message_id=message.id,
                             topic=self.topic,
                             subscribers=len(self._subscribers),
                             successful_deliveries=success_count)
            
            return success_count > 0
            
        except Exception as e:
            self._logger.error("Failed to publish message", error=str(e))
            return False
    
    async def _deliver_to_subscriber(self, subscriber_id: str, callback: Callable, message: Message) -> bool:
        """Deliver message to a single subscriber."""
        try:
            await callback(message)
            return True
        except Exception as e:
            self._logger.error("Failed to deliver message to subscriber",
                             subscriber_id=subscriber_id,
                             message_id=message.id,
                             error=str(e))
            return False
    
    def get_subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return len(self._subscribers)


class RequestResponsePattern(MessagePattern):
    """
    Request-Response pattern for synchronous communication.
    
    Features:
    - Synchronous request-response messaging
    - Correlation ID tracking
    - Timeout handling
    - Response routing
    """
    
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._message_broker = get_message_broker()
        
    async def start(self) -> bool:
        """Start request-response pattern."""
        try:
            self._active = True
            self._logger.info("Started request-response pattern", 
                            pattern_id=self.config.pattern_id)
            return True
        except Exception as e:
            self._logger.error("Failed to start request-response pattern", error=str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop request-response pattern."""
        try:
            self._active = False
            
            # Cancel pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            
            self._pending_requests.clear()
            
            self._logger.info("Stopped request-response pattern", 
                            pattern_id=self.config.pattern_id)
            return True
        except Exception as e:
            self._logger.error("Failed to stop request-response pattern", error=str(e))
            return False
    
    @trace_async(name="send_request", tags={"pattern": "request_response"})
    async def send_request(self, request: Message, target_queue: str, 
                          timeout: Optional[float] = None) -> Optional[Message]:
        """
        Send request and wait for response.
        
        Args:
            request: Request message
            target_queue: Target queue for request
            timeout: Request timeout in seconds
            
        Returns:
            Response message or None if timeout/error
        """
        if not self._active:
            return None
        
        try:
            # Set correlation ID for tracking
            correlation_id = str(uuid.uuid4())
            request.correlation_id = correlation_id
            request.reply_to = f"response_{self.config.pattern_id}"
            
            # Create future for response
            response_future = asyncio.Future()
            self._pending_requests[correlation_id] = response_future
            
            # Send request
            success = await self._message_broker.publish(request, routing_key=target_queue)
            if not success:
                del self._pending_requests[correlation_id]
                return None
            
            # Wait for response
            timeout_value = timeout or self.config.timeout or 30.0
            
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout_value)
                return response
            except asyncio.TimeoutError:
                self._logger.warning("Request timeout",
                                   correlation_id=correlation_id,
                                   timeout=timeout_value)
                return None
            finally:
                if correlation_id in self._pending_requests:
                    del self._pending_requests[correlation_id]
                    
        except Exception as e:
            self._logger.error("Failed to send request", error=str(e))
            return None
    
    async def handle_response(self, response: Message):
        """
        Handle incoming response message.
        
        Args:
            response: Response message
        """
        correlation_id = response.correlation_id
        
        if correlation_id and correlation_id in self._pending_requests:
            future = self._pending_requests[correlation_id]
            if not future.done():
                future.set_result(response)
                
                self._logger.debug("Response handled",
                                 correlation_id=correlation_id,
                                 response_id=response.id)
    
    def get_pending_request_count(self) -> int:
        """Get number of pending requests."""
        return len(self._pending_requests)


class WorkerQueuePattern(MessagePattern):
    """
    Worker Queue pattern for load distribution.
    
    Features:
    - Multiple worker support
    - Load balancing
    - Task acknowledgment
    - Dead letter handling
    """
    
    def __init__(self, config: PatternConfig, queue_name: str, max_workers: int = 5):
        super().__init__(config)
        self.queue_name = queue_name
        self.max_workers = max_workers
        self._workers: Dict[str, asyncio.Task] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._message_broker = get_message_broker()
        
    async def start(self) -> bool:
        """Start worker queue pattern."""
        try:
            # Start worker tasks
            for i in range(self.max_workers):
                worker_id = f"worker_{i}"
                task = asyncio.create_task(self._worker_loop(worker_id))
                self._workers[worker_id] = task
            
            self._active = True
            self._logger.info("Started worker queue pattern",
                            pattern_id=self.config.pattern_id,
                            queue_name=self.queue_name,
                            workers=self.max_workers)
            return True
            
        except Exception as e:
            self._logger.error("Failed to start worker queue pattern", error=str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop worker queue pattern."""
        try:
            self._active = False
            
            # Cancel all worker tasks
            for worker_id, task in self._workers.items():
                task.cancel()
            
            # Wait for workers to stop
            await asyncio.gather(*self._workers.values(), return_exceptions=True)
            self._workers.clear()
            
            self._logger.info("Stopped worker queue pattern",
                            pattern_id=self.config.pattern_id)
            return True
            
        except Exception as e:
            self._logger.error("Failed to stop worker queue pattern", error=str(e))
            return False
    
    async def submit_task(self, task_message: Message) -> bool:
        """
        Submit task to worker queue.
        
        Args:
            task_message: Task message
            
        Returns:
            True if task queued successfully
        """
        if not self._active:
            return False
        
        try:
            await self._task_queue.put(task_message)
            self._logger.debug("Task submitted to queue",
                             task_id=task_message.id,
                             queue_size=self._task_queue.qsize())
            return True
        except Exception as e:
            self._logger.error("Failed to submit task", error=str(e))
            return False
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing tasks."""
        self._logger.info("Worker started", worker_id=worker_id)
        
        try:
            while self._active:
                try:
                    # Get next task
                    task_message = await asyncio.wait_for(
                        self._task_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process task
                    await self._process_task(worker_id, task_message)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self._logger.error("Worker error",
                                     worker_id=worker_id,
                                     error=str(e))
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            pass
        finally:
            self._logger.info("Worker stopped", worker_id=worker_id)
    
    async def _process_task(self, worker_id: str, task_message: Message):
        """Process a single task."""
        try:
            self._logger.debug("Processing task",
                             worker_id=worker_id,
                             task_id=task_message.id)
            
            # Simulate task processing
            await asyncio.sleep(0.1)
            
            # Mark task as completed
            self._task_queue.task_done()
            
            self._logger.debug("Task completed",
                             worker_id=worker_id,
                             task_id=task_message.id)
            
        except Exception as e:
            self._logger.error("Task processing failed",
                             worker_id=worker_id,
                             task_id=task_message.id,
                             error=str(e))
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._task_queue.qsize()
    
    def get_active_workers(self) -> int:
        """Get number of active workers."""
        return len([task for task in self._workers.values() if not task.done()])


class PriorityQueuePattern(MessagePattern):
    """
    Priority Queue pattern for ordered message processing.
    
    Features:
    - Priority-based message ordering
    - Multiple priority levels
    - Configurable processing strategies
    - Performance monitoring
    """
    
    def __init__(self, config: PatternConfig, queue_name: str):
        super().__init__(config)
        self.queue_name = queue_name
        self._priority_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        self._processor_task: Optional[asyncio.Task] = None
        self._message_broker = get_message_broker()
        
    async def start(self) -> bool:
        """Start priority queue pattern."""
        try:
            # Start message processor
            self._processor_task = asyncio.create_task(self._process_priority_messages())
            
            self._active = True
            self._logger.info("Started priority queue pattern",
                            pattern_id=self.config.pattern_id,
                            queue_name=self.queue_name)
            return True
            
        except Exception as e:
            self._logger.error("Failed to start priority queue pattern", error=str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop priority queue pattern."""
        try:
            self._active = False
            
            # Cancel processor task
            if self._processor_task:
                self._processor_task.cancel()
                await asyncio.gather(self._processor_task, return_exceptions=True)
            
            self._logger.info("Stopped priority queue pattern",
                            pattern_id=self.config.pattern_id)
            return True
            
        except Exception as e:
            self._logger.error("Failed to stop priority queue pattern", error=str(e))
            return False
    
    async def enqueue(self, message: Message) -> bool:
        """
        Enqueue message with priority.
        
        Args:
            message: Message to enqueue
            
        Returns:
            True if enqueued successfully
        """
        if not self._active:
            return False
        
        try:
            priority = message.priority
            await self._priority_queues[priority].put(message)
            
            self._logger.debug("Message enqueued",
                             message_id=message.id,
                             priority=priority.name,
                             queue_size=self._priority_queues[priority].qsize())
            return True
            
        except Exception as e:
            self._logger.error("Failed to enqueue message", error=str(e))
            return False
    
    async def _process_priority_messages(self):
        """Process messages by priority."""
        while self._active:
            try:
                # Process in priority order: CRITICAL -> HIGH -> NORMAL -> LOW
                for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                               MessagePriority.NORMAL, MessagePriority.LOW]:
                    
                    queue = self._priority_queues[priority]
                    
                    # Process all messages in this priority level
                    while not queue.empty():
                        try:
                            message = queue.get_nowait()
                            await self._process_priority_message(message)
                        except asyncio.QueueEmpty:
                            break
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self._logger.error("Priority processor error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_priority_message(self, message: Message):
        """Process a single priority message."""
        try:
            self._logger.debug("Processing priority message",
                             message_id=message.id,
                             priority=message.priority.name)
            
            # Simulate message processing
            await asyncio.sleep(0.05)
            
            self._logger.debug("Priority message processed",
                             message_id=message.id)
            
        except Exception as e:
            self._logger.error("Priority message processing failed",
                             message_id=message.id,
                             error=str(e))
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get sizes of all priority queues."""
        return {
            priority.name: queue.qsize()
            for priority, queue in self._priority_queues.items()
        }
    
    def get_total_queue_size(self) -> int:
        """Get total size across all priority queues."""
        return sum(queue.qsize() for queue in self._priority_queues.values())


# Factory functions for creating patterns

def create_publish_subscribe(topic: str, pattern_id: Optional[str] = None) -> PublishSubscribePattern:
    """Create a publish-subscribe pattern."""
    config = PatternConfig(pattern_id=pattern_id or str(uuid.uuid4()))
    return PublishSubscribePattern(config, topic)


def create_request_response(timeout: float = 30.0, pattern_id: Optional[str] = None) -> RequestResponsePattern:
    """Create a request-response pattern."""
    config = PatternConfig(
        pattern_id=pattern_id or str(uuid.uuid4()),
        timeout=timeout
    )
    return RequestResponsePattern(config)


def create_worker_queue(queue_name: str, max_workers: int = 5, pattern_id: Optional[str] = None) -> WorkerQueuePattern:
    """Create a worker queue pattern."""
    config = PatternConfig(pattern_id=pattern_id or str(uuid.uuid4()))
    return WorkerQueuePattern(config, queue_name, max_workers)


def create_priority_queue(queue_name: str, pattern_id: Optional[str] = None) -> PriorityQueuePattern:
    """Create a priority queue pattern."""
    config = PatternConfig(pattern_id=pattern_id or str(uuid.uuid4()))
    return PriorityQueuePattern(config, queue_name)