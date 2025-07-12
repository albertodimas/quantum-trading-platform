"""
Unit tests for Event Bus System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid
from typing import List, Dict, Any

from src.core.architecture import (
    EventBus,
    Event,
    EventPriority,
    EventHandler,
    EventFilter
)
from src.models.trading import Order, Trade, OrderType, OrderSide


# Test events
class TestEvent(Event):
    """Generic test event"""
    def __init__(self, event_type: str, data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        super().__init__(event_type, data, priority)


class OrderCreatedEvent(Event):
    """Order created event"""
    def __init__(self, order: Order):
        super().__init__(
            event_type="order.created",
            data={"order": order},
            priority=EventPriority.HIGH
        )


class TradeExecutedEvent(Event):
    """Trade executed event"""
    def __init__(self, trade: Trade):
        super().__init__(
            event_type="trade.executed",
            data={"trade": trade},
            priority=EventPriority.HIGH
        )


class MarketDataEvent(Event):
    """Market data event"""
    def __init__(self, symbol: str, price: float):
        super().__init__(
            event_type="market.tick",
            data={"symbol": symbol, "price": price},
            priority=EventPriority.LOW
        )


class SystemEvent(Event):
    """System event"""
    def __init__(self, level: str, message: str):
        super().__init__(
            event_type=f"system.{level}",
            data={"message": message},
            priority=EventPriority.CRITICAL if level == "error" else EventPriority.NORMAL
        )


# Test handlers
class TestEventHandler(EventHandler):
    """Test event handler"""
    
    def __init__(self):
        self.handled_events: List[Event] = []
        self.call_count = 0
        self.should_fail = False
        self.delay = 0
    
    async def handle(self, event: Event):
        """Handle event"""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise Exception("Handler failed")
        
        self.handled_events.append(event)
        self.call_count += 1


class OrderEventHandler(EventHandler):
    """Handler for order events"""
    
    def __init__(self):
        self.orders_created: List[Order] = []
        self.orders_filled: List[Order] = []
    
    async def handle(self, event: Event):
        """Handle order events"""
        if event.type == "order.created":
            self.orders_created.append(event.data["order"])
        elif event.type == "order.filled":
            self.orders_filled.append(event.data["order"])


class SlowEventHandler(EventHandler):
    """Slow event handler for testing timeouts"""
    
    def __init__(self, delay: float):
        self.delay = delay
        self.completed = False
    
    async def handle(self, event: Event):
        """Handle event slowly"""
        await asyncio.sleep(self.delay)
        self.completed = True


# Fixtures
@pytest.fixture
async def event_bus():
    """Create event bus instance"""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def sample_order():
    """Create sample order"""
    return Order(
        id=str(uuid.uuid4()),
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.1,
        price=50000.0,
        status="PENDING"
    )


@pytest.fixture
def sample_trade():
    """Create sample trade"""
    return Trade(
        id=str(uuid.uuid4()),
        order_id="order123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        entry_price=50000.0,
        entry_time=datetime.now()
    )


@pytest.mark.unit
class TestEventBus:
    """Test event bus functionality"""
    
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish"""
        handler = TestEventHandler()
        
        # Subscribe to events
        event_bus.subscribe("test.event", handler)
        
        # Publish event
        event = TestEvent("test.event", {"message": "Hello"})
        await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        assert handler.call_count == 1
        assert handler.handled_events[0] == event
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event"""
        handler1 = TestEventHandler()
        handler2 = TestEventHandler()
        handler3 = TestEventHandler()
        
        # Subscribe multiple handlers
        event_bus.subscribe("test.event", handler1)
        event_bus.subscribe("test.event", handler2)
        event_bus.subscribe("test.event", handler3)
        
        # Publish event
        event = TestEvent("test.event", {"message": "Broadcast"})
        await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # All handlers should receive the event
        assert handler1.call_count == 1
        assert handler2.call_count == 1
        assert handler3.call_count == 1
    
    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test wildcard event subscription"""
        handler = TestEventHandler()
        
        # Subscribe to wildcard pattern
        event_bus.subscribe("order.*", handler)
        
        # Publish different order events
        event1 = TestEvent("order.created", {"id": "1"})
        event2 = TestEvent("order.updated", {"id": "2"})
        event3 = TestEvent("order.cancelled", {"id": "3"})
        event4 = TestEvent("trade.executed", {"id": "4"})  # Should not match
        
        await event_bus.publish(event1)
        await event_bus.publish(event2)
        await event_bus.publish(event3)
        await event_bus.publish(event4)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Should receive only order events
        assert handler.call_count == 3
        event_types = [e.type for e in handler.handled_events]
        assert "trade.executed" not in event_types
    
    @pytest.mark.asyncio
    async def test_priority_processing(self, event_bus):
        """Test priority-based event processing"""
        handler = TestEventHandler()
        event_bus.subscribe("*", handler)
        
        # Publish events with different priorities
        critical_event = TestEvent("system.critical", {"msg": "Critical"}, EventPriority.CRITICAL)
        high_event = TestEvent("order.created", {"msg": "High"}, EventPriority.HIGH)
        normal_event = TestEvent("data.update", {"msg": "Normal"}, EventPriority.NORMAL)
        low_event = TestEvent("market.tick", {"msg": "Low"}, EventPriority.LOW)
        
        # Publish in reverse priority order
        await event_bus.publish(low_event)
        await event_bus.publish(normal_event)
        await event_bus.publish(high_event)
        await event_bus.publish(critical_event)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Should be processed in priority order
        assert handler.call_count == 4
        priorities = [e.priority for e in handler.handled_events]
        
        # Verify critical events are processed first
        assert priorities[0] == EventPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, event_bus):
        """Test event filtering"""
        handler = TestEventHandler()
        
        # Create filter for specific symbol
        def symbol_filter(event: Event) -> bool:
            return event.data.get("symbol") == "BTC/USDT"
        
        event_filter = EventFilter(filter_func=symbol_filter)
        event_bus.subscribe("market.*", handler, event_filter)
        
        # Publish events
        btc_event = MarketDataEvent("BTC/USDT", 50000)
        eth_event = MarketDataEvent("ETH/USDT", 3000)
        sol_event = MarketDataEvent("SOL/USDT", 100)
        
        await event_bus.publish(btc_event)
        await event_bus.publish(eth_event)
        await event_bus.publish(sol_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Should only receive BTC event
        assert handler.call_count == 1
        assert handler.handled_events[0].data["symbol"] == "BTC/USDT"
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events"""
        handler = TestEventHandler()
        
        # Subscribe
        subscription_id = event_bus.subscribe("test.event", handler)
        
        # Publish first event
        await event_bus.publish(TestEvent("test.event", {"num": 1}))
        await asyncio.sleep(0.1)
        assert handler.call_count == 1
        
        # Unsubscribe
        event_bus.unsubscribe(subscription_id)
        
        # Publish second event
        await event_bus.publish(TestEvent("test.event", {"num": 2}))
        await asyncio.sleep(0.1)
        
        # Should not receive second event
        assert handler.call_count == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, event_bus):
        """Test error handling in event handlers"""
        good_handler = TestEventHandler()
        bad_handler = TestEventHandler()
        bad_handler.should_fail = True
        
        # Subscribe both handlers
        event_bus.subscribe("test.event", good_handler)
        event_bus.subscribe("test.event", bad_handler)
        
        # Publish event
        event = TestEvent("test.event", {"data": "test"})
        await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Good handler should still receive event
        assert good_handler.call_count == 1
        assert bad_handler.call_count == 0  # Failed before incrementing
    
    @pytest.mark.asyncio
    async def test_handler_timeout(self, event_bus):
        """Test handler timeout"""
        fast_handler = TestEventHandler()
        slow_handler = SlowEventHandler(delay=2.0)
        
        # Configure timeout
        event_bus._handler_timeout = 0.5
        
        # Subscribe handlers
        event_bus.subscribe("test.event", fast_handler)
        event_bus.subscribe("test.event", slow_handler)
        
        # Publish event
        await event_bus.publish(TestEvent("test.event", {"data": "test"}))
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Fast handler should complete
        assert fast_handler.call_count == 1
        # Slow handler should timeout
        assert not slow_handler.completed
    
    @pytest.mark.asyncio
    async def test_event_replay(self, event_bus):
        """Test event replay functionality"""
        handler = TestEventHandler()
        
        # Enable event history
        event_bus._store_history = True
        event_bus._max_history_size = 10
        
        # Publish events
        events = []
        for i in range(5):
            event = TestEvent("test.event", {"num": i})
            events.append(event)
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Subscribe handler after events published
        event_bus.subscribe("test.event", handler)
        
        # Replay events
        await event_bus.replay_events("test.event", handler)
        
        # Should receive all historical events
        assert handler.call_count == 5
        assert len(handler.handled_events) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_publishing(self, event_bus):
        """Test concurrent event publishing"""
        handler = TestEventHandler()
        event_bus.subscribe("test.*", handler)
        
        # Publish many events concurrently
        async def publish_events(start: int, count: int):
            for i in range(start, start + count):
                event = TestEvent(f"test.event{i % 3}", {"num": i})
                await event_bus.publish(event)
        
        # Create concurrent publishers
        tasks = [
            publish_events(0, 10),
            publish_events(10, 10),
            publish_events(20, 10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Should receive all events
        assert handler.call_count == 30
    
    @pytest.mark.asyncio
    async def test_event_metrics(self, event_bus):
        """Test event metrics collection"""
        handler = TestEventHandler()
        handler.delay = 0.1  # Add some processing time
        
        event_bus.subscribe("test.event", handler)
        
        # Publish events
        for i in range(5):
            await event_bus.publish(TestEvent("test.event", {"num": i}))
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Get metrics
        metrics = event_bus.get_metrics()
        
        assert metrics["events_published"] >= 5
        assert metrics["events_processed"] >= 5
        assert metrics["handlers_registered"] >= 1
        assert "average_processing_time" in metrics
        assert metrics["average_processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_typed_event_handling(self, event_bus, sample_order, sample_trade):
        """Test handling of typed events"""
        order_handler = OrderEventHandler()
        
        # Subscribe to order events
        event_bus.subscribe("order.*", order_handler)
        
        # Publish typed events
        await event_bus.publish(OrderCreatedEvent(sample_order))
        
        # Create filled order event
        filled_order = sample_order.copy()
        filled_order.status = "FILLED"
        await event_bus.publish(Event("order.filled", {"order": filled_order}))
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify handling
        assert len(order_handler.orders_created) == 1
        assert len(order_handler.orders_filled) == 1
        assert order_handler.orders_created[0].id == sample_order.id
        assert order_handler.orders_filled[0].status == "FILLED"
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, event_bus):
        """Test dead letter queue for failed events"""
        handler = TestEventHandler()
        handler.should_fail = True
        
        # Enable dead letter queue
        event_bus._enable_dead_letter_queue = True
        event_bus._max_retries = 3
        
        # Subscribe failing handler
        event_bus.subscribe("test.event", handler)
        
        # Publish event
        event = TestEvent("test.event", {"important": "data"})
        await event_bus.publish(event)
        
        # Wait for retries
        await asyncio.sleep(0.5)
        
        # Check dead letter queue
        dlq = event_bus.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0]["event"].id == event.id
        assert dlq[0]["retry_count"] >= 3
    
    @pytest.mark.asyncio
    async def test_event_expiration(self, event_bus):
        """Test event expiration"""
        handler = TestEventHandler()
        event_bus.subscribe("test.event", handler)
        
        # Create expired event
        expired_event = TestEvent("test.event", {"data": "old"})
        expired_event.timestamp = datetime.now() - timedelta(hours=1)
        expired_event.expires_at = datetime.now() - timedelta(minutes=30)
        
        # Create valid event
        valid_event = TestEvent("test.event", {"data": "new"})
        
        # Publish both
        await event_bus.publish(expired_event)
        await event_bus.publish(valid_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Should only process valid event
        assert handler.call_count == 1
        assert handler.handled_events[0].data["data"] == "new"
    
    @pytest.mark.asyncio
    async def test_event_bus_shutdown(self, event_bus):
        """Test graceful shutdown"""
        handler = TestEventHandler()
        handler.delay = 0.5  # Slow handler
        
        event_bus.subscribe("test.event", handler)
        
        # Publish events
        for i in range(5):
            await event_bus.publish(TestEvent("test.event", {"num": i}))
        
        # Start shutdown
        await event_bus.stop()
        
        # Try to publish after shutdown
        with pytest.raises(RuntimeError):
            await event_bus.publish(TestEvent("test.event", {"num": 99}))
    
    @pytest.mark.asyncio
    async def test_event_persistence(self, event_bus):
        """Test event persistence (if implemented)"""
        # This would test saving events to persistent storage
        # For now, test in-memory history
        event_bus._store_history = True
        
        events = []
        for i in range(5):
            event = TestEvent(f"persist.event.{i}", {"num": i})
            events.append(event)
            await event_bus.publish(event)
        
        # Get event history
        history = event_bus.get_event_history()
        
        assert len(history) >= 5
        # Verify events are in history
        event_ids = {e.id for e in events}
        history_ids = {h.id for h in history}
        assert event_ids.issubset(history_ids)


@pytest.mark.unit 
class TestEventPriority:
    """Test event priority handling"""
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, event_bus):
        """Test that events are processed in priority order"""
        handler = TestEventHandler()
        event_bus.subscribe("*", handler)
        
        # Create a batch of events with mixed priorities
        events = [
            TestEvent("test.low", {"num": 1}, EventPriority.LOW),
            TestEvent("test.critical", {"num": 2}, EventPriority.CRITICAL),
            TestEvent("test.normal", {"num": 3}, EventPriority.NORMAL),
            TestEvent("test.high", {"num": 4}, EventPriority.HIGH),
            TestEvent("test.critical2", {"num": 5}, EventPriority.CRITICAL),
        ]
        
        # Publish all at once
        for event in events:
            await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Check processing order
        processed_priorities = [e.priority for e in handler.handled_events]
        
        # Critical events should be first
        critical_count = sum(1 for p in processed_priorities[:2] if p == EventPriority.CRITICAL)
        assert critical_count == 2


@pytest.mark.unit
class TestEventFiltering:
    """Test advanced event filtering"""
    
    @pytest.mark.asyncio
    async def test_complex_filter(self, event_bus):
        """Test complex event filtering logic"""
        handler = TestEventHandler()
        
        # Create complex filter
        def complex_filter(event: Event) -> bool:
            # Filter by multiple conditions
            if event.type.startswith("order."):
                return event.data.get("quantity", 0) > 0.05
            elif event.type.startswith("market."):
                return event.data.get("price", 0) > 40000
            return True
        
        event_filter = EventFilter(filter_func=complex_filter)
        event_bus.subscribe("*", handler, event_filter)
        
        # Publish various events
        events = [
            Event("order.created", {"quantity": 0.1}),      # Pass
            Event("order.created", {"quantity": 0.01}),     # Fail
            MarketDataEvent("BTC/USDT", 50000),            # Pass
            MarketDataEvent("BTC/USDT", 30000),            # Fail
            TestEvent("other.event", {"data": "test"}),     # Pass
        ]
        
        for event in events:
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Should receive 3 events
        assert handler.call_count == 3