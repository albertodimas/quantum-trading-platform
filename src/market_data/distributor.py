"""
Data Distributor for Market Data

Manages subscriptions and distributes normalized market data to consumers.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Set, Callable, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from ..core.architecture import EventBus
from ..core.messaging import MessageBroker
from ..core.observability import get_logger

logger = get_logger(__name__)

@dataclass
class Subscription:
    """Represents a data subscription"""
    subscriber_id: str
    symbol: str
    data_types: Set[str]
    callback: Callable
    filters: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class SubscriptionManager:
    """
    Manages market data subscriptions
    """
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.subscriber_index: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logger        
    def add_subscription(
        self,
        subscriber_id: str,
        symbol: str,
        data_types: List[str],
        callback: Callable,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new subscription"""
        subscription = Subscription(
            subscriber_id=subscriber_id,
            symbol=symbol,
            data_types=set(data_types),
            callback=callback,
            filters=filters
        )
        
        # Add to symbol subscriptions
        self.subscriptions[symbol].append(subscription)
        
        # Update subscriber index
        self.subscriber_index[subscriber_id].add(symbol)
        
        self.logger.info(
            f"Added subscription: {subscriber_id} for {symbol} "
            f"with types {data_types}"
        )
        
        return f"{subscriber_id}:{symbol}"
        
    def remove_subscription(
        self,
        subscriber_id: str,
        symbol: Optional[str] = None
    ):
        """Remove subscription(s)"""
        if symbol:
            # Remove specific symbol subscription
            self.subscriptions[symbol] = [
                sub for sub in self.subscriptions[symbol]
                if sub.subscriber_id != subscriber_id
            ]
            self.subscriber_index[subscriber_id].discard(symbol)
        else:
            # Remove all subscriptions for subscriber
            symbols = list(self.subscriber_index[subscriber_id])
            for sym in symbols:
                self.subscriptions[sym] = [
                    sub for sub in self.subscriptions[sym]
                    if sub.subscriber_id != subscriber_id
                ]
            del self.subscriber_index[subscriber_id]
            
        self.logger.info(f"Removed subscription: {subscriber_id} {symbol or 'all'}")
        
    def get_subscriptions(
        self,
        symbol: str,
        data_type: Optional[str] = None
    ) -> List[Subscription]:
        """Get subscriptions for a symbol and optional data type"""
        subs = self.subscriptions.get(symbol, [])
        
        if data_type:
            subs = [s for s in subs if data_type in s.data_types]
            
        return subs
        
    def get_subscriber_symbols(self, subscriber_id: str) -> Set[str]:
        """Get all symbols a subscriber is subscribed to"""
        return self.subscriber_index.get(subscriber_id, set())
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get subscription statistics"""
        total_subs = sum(len(subs) for subs in self.subscriptions.values())
        
        return {
            'total_subscriptions': total_subs,
            'unique_subscribers': len(self.subscriber_index),
            'subscribed_symbols': len(self.subscriptions),
            'subscriptions_by_symbol': {
                symbol: len(subs) 
                for symbol, subs in self.subscriptions.items()
            }
        }

class DataDistributor:
    """
    Distributes market data to subscribers through various channels
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        message_broker: MessageBroker
    ):
        self.event_bus = event_bus
        self.broker = message_broker
        self.subscription_manager = SubscriptionManager()
        self.logger = logger
        
        # Distribution statistics
        self.distribution_count = 0
        self.error_count = 0
        self.last_distribution_time: Dict[str, datetime] = {}
        
        # Batching settings
        self.batch_size = 100
        self.batch_interval = 0.1  # seconds
        self.pending_distributions: List[Dict[str, Any]] = []
        
        self.is_running = False
        
    async def initialize(self):
        """Initialize distributor"""
        self.is_running = True
        
        # Start batch distribution task
        asyncio.create_task(self._batch_distribution_loop())
        
        self.logger.info("Data distributor initialized")
        
    async def start(self):
        """Start distribution"""
        self.is_running = True
        
    async def stop(self):
        """Stop distribution"""
        self.is_running = False        
    async def subscribe(
        self,
        symbol: str,
        data_types: List[str],
        callback: Callable,
        subscriber_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to market data"""
        if not subscriber_id:
            subscriber_id = f"subscriber_{id(callback)}"
            
        return self.subscription_manager.add_subscription(
            subscriber_id,
            symbol,
            data_types,
            callback,
            filters
        )
        
    async def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from market data"""
        subscriber_id = f"subscriber_{id(callback)}"
        self.subscription_manager.remove_subscription(subscriber_id, symbol)
        
    async def distribute(
        self,
        symbol: str,
        data_type: str,
        data: Any,
        immediate: bool = False
    ):
        """Distribute market data to subscribers"""
        distribution = {
            'symbol': symbol,
            'data_type': data_type,
            'data': data,
            'timestamp': datetime.now()
        }        
        if immediate:
            # Distribute immediately
            await self._distribute_single(distribution)
        else:
            # Add to batch
            self.pending_distributions.append(distribution)
            
            # Check if batch is full
            if len(self.pending_distributions) >= self.batch_size:
                await self._distribute_batch()
                
    async def _distribute_single(self, distribution: Dict[str, Any]):
        """Distribute single data update"""
        symbol = distribution['symbol']
        data_type = distribution['data_type']
        data = distribution['data']
        
        # Get relevant subscriptions
        subscriptions = self.subscription_manager.get_subscriptions(
            symbol, data_type
        )
        
        # Distribute via callbacks
        callback_tasks = []
        for sub in subscriptions:
            # Apply filters if any
            if sub.filters and not self._apply_filters(data, sub.filters):
                continue
                
            # Create task for callback
            task = asyncio.create_task(
                self._safe_callback(sub.callback, symbol, data_type, data)
            )
            callback_tasks.append(task)
            
        # Wait for all callbacks with timeout
        if callback_tasks:
            await asyncio.wait(callback_tasks, timeout=1.0)            
        # Distribute via event bus
        await self.event_bus.publish(f"market_data.{data_type}", {
            'symbol': symbol,
            'data': data,
            'timestamp': distribution['timestamp']
        })
        
        # Distribute via message broker
        await self.broker.publish(
            f"market.{symbol}.{data_type}",
            distribution
        )
        
        # Update statistics
        self.distribution_count += 1
        self.last_distribution_time[f"{symbol}:{data_type}"] = datetime.now()
        
    async def _distribute_batch(self):
        """Distribute pending batch"""
        if not self.pending_distributions:
            return
            
        # Take current batch
        batch = self.pending_distributions[:self.batch_size]
        self.pending_distributions = self.pending_distributions[self.batch_size:]
        
        # Group by symbol and type for efficiency
        grouped = defaultdict(list)
        for dist in batch:
            key = (dist['symbol'], dist['data_type'])
            grouped[key].append(dist['data'])
            
        # Distribute grouped data
        for (symbol, data_type), data_list in grouped.items():
            await self._distribute_single({
                'symbol': symbol,
                'data_type': data_type,
                'data': data_list if len(data_list) > 1 else data_list[0],
                'timestamp': datetime.now()
            })