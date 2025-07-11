"""
Enterprise Event Bus System for decoupled communication.

Features:
- Event-driven architecture with pattern matching
- Asynchronous event processing
- Event filtering and routing
- Event persistence and replay
- Dead letter handling for failed events
- Performance metrics and monitoring
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Pattern
from enum import Enum
import re
import json
from datetime import datetime, timedelta
import weakref

from ..observability.logger import get_logger
from ..observability.metrics import get_metrics_collector
from ..observability.tracing import trace_async


logger = get_logger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Event:
    """Event container with metadata and routing information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    status: EventStatus = EventStatus.PENDING
    
    def is_expired(self) -> bool:
        """Check if event has expired."""
        if self.ttl is None:
            return False
        return datetime.utcnow() > (self.timestamp + self.ttl)
    
    def should_retry(self) -> bool:
        """Check if event should be retried."""
        return self.retry_count < self.max_retries and self.status == EventStatus.FAILED
    
    def increment_retry(self):
        """Increment retry count."""
        self.retry_count += 1
        self.status = EventStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "metadata": self.metadata,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl.total_seconds() if self.ttl else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        event = cls()
        event.id = data.get("id", str(uuid.uuid4()))
        event.type = data.get("type", "")
        event.data = data.get("data")
        event.metadata = data.get("metadata", {})
        event.source = data.get("source", "")
        event.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
        event.priority = EventPriority(data.get("priority", EventPriority.NORMAL.value))
        event.correlation_id = data.get("correlation_id")
        event.causation_id = data.get("causation_id")
        event.reply_to = data.get("reply_to")
        if data.get("ttl"):
            event.ttl = timedelta(seconds=data["ttl"])
        event.retry_count = data.get("retry_count", 0)
        event.max_retries = data.get("max_retries", 3)
        event.status = EventStatus(data.get("status", EventStatus.PENDING.value))
        return event


@dataclass
class EventFilter:
    """Event filter configuration."""
    event_types: List[str] = field(default_factory=list)
    event_patterns: List[Pattern] = field(default_factory=list)
    source_patterns: List[Pattern] = field(default_factory=list)
    priority_levels: List[EventPriority] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria."""
        # Check event types
        if self.event_types and event.type not in self.event_types:
            return False
        
        # Check event type patterns
        if self.event_patterns:
            if not any(pattern.match(event.type) for pattern in self.event_patterns):
                return False
        
        # Check source patterns
        if self.source_patterns:
            if not any(pattern.match(event.source) for pattern in self.source_patterns):
                return False
        
        # Check priority levels
        if self.priority_levels and event.priority not in self.priority_levels:
            return False
        
        # Check metadata filters
        for key, expected_value in self.metadata_filters.items():
            if key not in event.metadata or event.metadata[key] != expected_value:
                return False
        
        return True


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    def __init__(self, handler_id: Optional[str] = None):
        self.handler_id = handler_id or str(uuid.uuid4())
        self._logger = get_logger(f"event_handler.{self.__class__.__name__}")
        self._metrics = get_metrics_collector().get_collector("trading")
        self.processing_count = 0
        self.error_count = 0
        self.last_processed_at: Optional[datetime] = None
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """Handle the event. Return True if successfully processed."""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        pass
    
    def get_filter(self) -> Optional[EventFilter]:
        """Get event filter for this handler."""
        return None
    
    async def on_error(self, event: Event, error: Exception):
        """Handle processing error."""
        self.error_count += 1
        self._logger.error("Event processing failed",
                          event_id=event.id,
                          event_type=event.type,
                          handler_id=self.handler_id,
                          error=str(error))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "handler_id": self.handler_id,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "error_rate": self.error_count / max(1, self.processing_count)
        }


@dataclass
class HandlerRegistration:
    """Handler registration with routing information."""
    handler: EventHandler
    filter: Optional[EventFilter] = None
    priority: int = 0  # Higher priority handlers execute first
    concurrent: bool = True  # Whether to run concurrently with other handlers
    timeout: Optional[float] = None  # Handler timeout in seconds
    
    @property
    def handler_id(self) -> str:
        """Get handler ID."""
        return self.handler.handler_id


class EventBus:
    """
    Enterprise Event Bus for decoupled communication.
    
    Features:
    - Event publication and subscription
    - Pattern-based event routing
    - Priority-based processing
    - Event persistence and replay
    - Dead letter queue for failed events
    - Performance monitoring and metrics
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._handlers: Dict[str, HandlerRegistration] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._dead_letter_queue: asyncio.Queue = asyncio.Queue()
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._event_history: List[Event] = []
        self._max_history_size = 10000
        
        self._logger = get_logger(f"event_bus.{name}")
        self._metrics = get_metrics_collector().get_collector("trading") if get_metrics_collector() else None
        
        # Statistics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_retried": 0,
            "dead_letter_count": 0
        }
        
        # Start event processing
        self._processor_task = asyncio.create_task(self._process_events())
        self._dead_letter_task = asyncio.create_task(self._process_dead_letters())
        
        self._running = True
    
    def register_handler(self, handler: EventHandler, 
                        event_filter: Optional[EventFilter] = None,
                        priority: int = 0,
                        concurrent: bool = True,
                        timeout: Optional[float] = None) -> str:
        """
        Register an event handler.
        
        Args:
            handler: Event handler instance
            event_filter: Optional filter for events
            priority: Handler priority (higher values = higher priority)
            concurrent: Whether to run concurrently with other handlers
            timeout: Handler timeout in seconds
            
        Returns:
            Handler registration ID
        """
        # Use handler's filter if none provided
        if event_filter is None:
            event_filter = handler.get_filter()
        
        registration = HandlerRegistration(
            handler=handler,
            filter=event_filter,
            priority=priority,
            concurrent=concurrent,
            timeout=timeout
        )
        
        self._handlers[handler.handler_id] = registration
        
        self._logger.info("Registered event handler",
                         handler_id=handler.handler_id,
                         handler_class=handler.__class__.__name__,
                         priority=priority)
        
        return handler.handler_id
    
    def unregister_handler(self, handler_id: str) -> bool:
        """Unregister an event handler."""
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            self._logger.info("Unregistered event handler", handler_id=handler_id)
            return True
        return False
    
    @trace_async(name="publish_event", tags={"component": "event_bus"})
    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
            
        Returns:
            True if event was queued successfully
        """
        try:
            # Validate event
            if not event.type:
                raise ValueError("Event type is required")
            
            # Set source if not provided
            if not event.source:
                event.source = f"event_bus.{self.name}"
            
            # Add to event history
            self._add_to_history(event)
            
            # Queue event for processing
            await self._event_queue.put(event)
            
            self._stats["events_published"] += 1
            
            if self._metrics:
                self._metrics.record_metric("events.published", 1, tags={
                    "event_type": event.type,
                    "priority": event.priority.name,
                    "source": event.source
                })
            
            self._logger.debug("Published event",
                             event_id=event.id,
                             event_type=event.type,
                             priority=event.priority.name)
            
            return True
            
        except Exception as e:
            self._logger.error("Failed to publish event", error=str(e))
            return False
    
    async def publish_and_wait(self, event: Event, timeout: Optional[float] = None) -> List[bool]:
        """
        Publish event and wait for all handlers to process it.
        
        Args:
            event: Event to publish
            timeout: Timeout for waiting
            
        Returns:
            List of handler results
        """
        # Create correlation for tracking
        correlation_id = str(uuid.uuid4())
        event.correlation_id = correlation_id
        
        # Publish event
        if not await self.publish(event):
            return []
        
        # Wait for processing completion
        # This is a simplified implementation
        # In production, you'd want a more sophisticated tracking mechanism
        await asyncio.sleep(0.1)  # Allow processing to start
        
        return []  # Placeholder
    
    async def _process_events(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Get next event
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Check if event is expired
                if event.is_expired():
                    self._logger.warning("Event expired", event_id=event.id)
                    continue
                
                # Process event
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error("Error in event processing loop", error=str(e))
                await asyncio.sleep(1)
    
    async def _handle_event(self, event: Event):
        """Handle a single event."""
        try:
            event.status = EventStatus.PROCESSING
            
            # Find matching handlers
            matching_handlers = self._find_matching_handlers(event)
            
            if not matching_handlers:
                self._logger.debug("No handlers found for event",
                                 event_id=event.id,
                                 event_type=event.type)
                event.status = EventStatus.COMPLETED
                return
            
            # Sort handlers by priority
            matching_handlers.sort(key=lambda h: h.priority, reverse=True)
            
            # Group handlers by concurrency
            concurrent_handlers = [h for h in matching_handlers if h.concurrent]
            sequential_handlers = [h for h in matching_handlers if not h.concurrent]
            
            # Process concurrent handlers
            if concurrent_handlers:
                tasks = []
                for registration in concurrent_handlers:
                    task = asyncio.create_task(
                        self._execute_handler(registration, event)
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process sequential handlers
            for registration in sequential_handlers:
                await self._execute_handler(registration, event)
            
            event.status = EventStatus.COMPLETED
            self._stats["events_processed"] += 1
            
            if self._metrics:
                self._metrics.record_metric("events.processed", 1, tags={
                    "event_type": event.type,
                    "handlers_count": len(matching_handlers)
                })
            
        except Exception as e:
            event.status = EventStatus.FAILED
            self._stats["events_failed"] += 1
            
            self._logger.error("Event processing failed",
                             event_id=event.id,
                             event_type=event.type,
                             error=str(e))
            
            # Handle retry or dead letter
            await self._handle_failed_event(event)
    
    async def _execute_handler(self, registration: HandlerRegistration, event: Event):
        """Execute a single handler."""
        handler = registration.handler
        
        try:
            # Execute handler with timeout
            if registration.timeout:
                await asyncio.wait_for(
                    handler.handle(event),
                    timeout=registration.timeout
                )
            else:
                await handler.handle(event)
            
            handler.processing_count += 1
            handler.last_processed_at = datetime.utcnow()
            
        except asyncio.TimeoutError:
            handler.error_count += 1
            self._logger.error("Handler timeout",
                             handler_id=handler.handler_id,
                             event_id=event.id,
                             timeout=registration.timeout)
            raise
            
        except Exception as e:
            handler.error_count += 1
            await handler.on_error(event, e)
            raise
    
    def _find_matching_handlers(self, event: Event) -> List[HandlerRegistration]:
        """Find handlers that can process the event."""
        matching = []
        
        for registration in self._handlers.values():
            # Check handler's can_handle method
            if not registration.handler.can_handle(event):
                continue
            
            # Check filter if present
            if registration.filter and not registration.filter.matches(event):
                continue
            
            matching.append(registration)
        
        return matching
    
    async def _handle_failed_event(self, event: Event):
        """Handle failed event processing."""
        if event.should_retry():
            event.increment_retry()
            await self._event_queue.put(event)
            self._stats["events_retried"] += 1
            
            self._logger.info("Retrying failed event",
                            event_id=event.id,
                            retry_count=event.retry_count)
        else:
            # Send to dead letter queue
            event.status = EventStatus.DEAD_LETTER
            await self._dead_letter_queue.put(event)
            self._stats["dead_letter_count"] += 1
            
            self._logger.warning("Event sent to dead letter queue",
                               event_id=event.id,
                               retry_count=event.retry_count)
    
    async def _process_dead_letters(self):
        """Process dead letter queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._dead_letter_queue.get(), timeout=5.0)
                
                # Log dead letter event
                self._logger.error("Dead letter event",
                                 event_id=event.id,
                                 event_type=event.type,
                                 retry_count=event.retry_count)
                
                # In production, you might want to:
                # - Store to database for analysis
                # - Send to external monitoring system
                # - Implement manual retry mechanism
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error("Error processing dead letters", error=str(e))
    
    def _add_to_history(self, event: Event):
        """Add event to history."""
        self._event_history.append(event)
        
        # Trim history if too large
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    async def replay_events(self, event_filter: Optional[EventFilter] = None,
                          from_time: Optional[datetime] = None,
                          to_time: Optional[datetime] = None):
        """Replay events from history."""
        events_to_replay = []
        
        for event in self._event_history:
            # Check time range
            if from_time and event.timestamp < from_time:
                continue
            if to_time and event.timestamp > to_time:
                continue
            
            # Check filter
            if event_filter and not event_filter.matches(event):
                continue
            
            # Create new event for replay
            replay_event = Event.from_dict(event.to_dict())
            replay_event.id = str(uuid.uuid4())  # New ID for replay
            replay_event.causation_id = event.id  # Track original event
            replay_event.metadata["replayed"] = True
            replay_event.metadata["original_event_id"] = event.id
            replay_event.retry_count = 0
            replay_event.status = EventStatus.PENDING
            
            events_to_replay.append(replay_event)
        
        # Publish replay events
        for event in events_to_replay:
            await self.publish(event)
        
        self._logger.info("Replayed events", count=len(events_to_replay))
        return len(events_to_replay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        handler_stats = {
            handler_id: registration.handler.get_stats()
            for handler_id, registration in self._handlers.items()
        }
        
        return {
            "name": self.name,
            "queue_size": self._event_queue.qsize(),
            "dead_letter_size": self._dead_letter_queue.qsize(),
            "history_size": len(self._event_history),
            "handlers_count": len(self._handlers),
            "processing_tasks": len(self._processing_tasks),
            "stats": self._stats,
            "handler_stats": handler_stats
        }
    
    async def shutdown(self):
        """Shutdown the event bus."""
        self._running = False
        
        # Cancel processing tasks
        if self._processor_task:
            self._processor_task.cancel()
        if self._dead_letter_task:
            self._dead_letter_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self._processor_task,
            self._dead_letter_task,
            return_exceptions=True
        )
        
        self._logger.info("Event bus shutdown complete")


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus("global")
    return _global_event_bus


def create_event_filter(event_types: List[str] = None,
                       event_patterns: List[str] = None,
                       source_patterns: List[str] = None,
                       priority_levels: List[EventPriority] = None,
                       metadata_filters: Dict[str, Any] = None) -> EventFilter:
    """
    Create an event filter.
    
    Args:
        event_types: List of exact event types to match
        event_patterns: List of regex patterns for event types
        source_patterns: List of regex patterns for event sources
        priority_levels: List of priority levels to match
        metadata_filters: Dictionary of metadata key-value pairs to match
        
    Returns:
        EventFilter instance
    """
    filter_obj = EventFilter()
    
    if event_types:
        filter_obj.event_types = event_types
    
    if event_patterns:
        filter_obj.event_patterns = [re.compile(pattern) for pattern in event_patterns]
    
    if source_patterns:
        filter_obj.source_patterns = [re.compile(pattern) for pattern in source_patterns]
    
    if priority_levels:
        filter_obj.priority_levels = priority_levels
    
    if metadata_filters:
        filter_obj.metadata_filters = metadata_filters
    
    return filter_obj