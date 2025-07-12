"""
Event Simulator for Backtesting

Simulates market events and time progression for realistic backtesting scenarios.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import heapq
import pandas as pd
import numpy as np

from ..core.architecture import EventBus, injectable
from ..core.observability import get_logger

logger = get_logger(__name__)

class EventType(Enum):
    """Types of simulation events"""
    MARKET_DATA = "market_data"
    TIME_UPDATE = "time_update"
    ORDER_FILL = "order_fill"
    POSITION_UPDATE = "position_update"
    RISK_CHECK = "risk_check"
    REBALANCE = "rebalance"
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    DIVIDEND = "dividend"
    SPLIT = "split"
    HALT = "halt"
    NEWS = "news"

@dataclass
class SimulationEvent:
    """Base class for simulation events"""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any]
    priority: int = 0  # Lower number = higher priority    
    def __lt__(self, other):
        """For priority queue comparison"""
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

@dataclass
class MarketEvent(SimulationEvent):
    """Market data update event"""
    def __init__(self, timestamp: datetime, data: Dict[str, Any], symbols: List[str]):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.MARKET_DATA,
            data={'market_data': data, 'symbols': symbols},
            priority=1
        )
        
@dataclass
class TimeEvent(SimulationEvent):
    """Time update event"""
    def __init__(self, timestamp: datetime):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.TIME_UPDATE,
            data={'timestamp': timestamp},
            priority=0  # Highest priority
        )

@injectable
class EventSimulator:
    """
    Simulates market events and manages time progression
    """    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        mode: str = "bar_by_bar"
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.current_time = start_date
        self.logger = logger
        
        # Event queue (priority queue)
        self.event_queue: List[SimulationEvent] = []
        
        # Scheduled events
        self.scheduled_events: Dict[str, List[Callable]] = {
            'daily': [],
            'weekly': [],
            'monthly': [],
            'custom': []
        }
        
        # Trading calendar
        self.trading_calendar = self._create_trading_calendar()
        
    def _create_trading_calendar(self) -> Dict[str, Any]:
        """Create trading calendar with market hours"""
        return {
            'market_open': timedelta(hours=9, minutes=30),  # 9:30 AM
            'market_close': timedelta(hours=16),  # 4:00 PM
            'pre_market_open': timedelta(hours=4),  # 4:00 AM
            'after_market_close': timedelta(hours=20),  # 8:00 PM
            'holidays': [],  # List of holiday dates
            'half_days': []  # List of half-day dates
        }        
    def add_event(self, event: SimulationEvent):
        """Add event to the simulation queue"""
        heapq.heappush(self.event_queue, event)
        
    def schedule_recurring_event(
        self,
        frequency: str,
        callback: Callable,
        start_time: Optional[datetime] = None
    ):
        """Schedule recurring events (daily, weekly, monthly)"""
        if frequency not in self.scheduled_events:
            raise ValueError(f"Invalid frequency: {frequency}")
            
        self.scheduled_events[frequency].append({
            'callback': callback,
            'start_time': start_time or self.start_date
        })
        
    def is_trading_day(self, date: datetime) -> bool:
        """Check if date is a trading day"""
        # Skip weekends
        if date.weekday() >= 5:  # Saturday or Sunday
            return False
            
        # Skip holidays
        if date.date() in self.trading_calendar['holidays']:
            return False
            
        return True
        
    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp"""
        if not self.is_trading_day(timestamp):
            return False            
        time_of_day = timestamp.time()
        market_open = (datetime.min + self.trading_calendar['market_open']).time()
        market_close = (datetime.min + self.trading_calendar['market_close']).time()
        
        # Check for half days
        if timestamp.date() in self.trading_calendar['half_days']:
            market_close = (datetime.min + timedelta(hours=13)).time()  # 1:00 PM
            
        return market_open <= time_of_day <= market_close
        
    async def simulate_market_event(
        self,
        timestamp: datetime,
        market_data: Dict[str, Any]
    ):
        """Simulate a market data event"""
        # Add realistic delays
        if self.mode == "tick_by_tick":
            await asyncio.sleep(0.001)  # 1ms delay
        elif self.mode == "bar_by_bar":
            await asyncio.sleep(0.01)   # 10ms delay
            
        # Add market microstructure noise
        for symbol, data in market_data.items():
            if 'close' in data and data['close'] is not None:
                # Add small random noise
                noise = np.random.normal(0, 0.0001)  # 0.01% noise
                data['close'] *= (1 + noise)
                
        return MarketEvent(timestamp, market_data, list(market_data.keys()))        
    async def simulate_order_fill(
        self,
        order: Dict[str, Any],
        market_price: float,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Simulate order fill with realistic conditions"""
        fill_price = market_price
        fill_quantity = order['quantity']
        
        # Simulate partial fills for large orders
        if order['quantity'] * market_price > 100000:  # Large order
            # Potentially partial fill
            fill_probability = 0.8
            if np.random.random() > fill_probability:
                fill_quantity = int(order['quantity'] * np.random.uniform(0.5, 0.9))
                
        # Add slippage based on order size
        size_impact = (order['quantity'] * market_price) / 1000000  # Impact per $1M
        slippage = size_impact * 0.001  # 0.1% per $1M
        
        if order['side'] == 'buy':
            fill_price *= (1 + slippage)
        else:
            fill_price *= (1 - slippage)
            
        return {
            'order_id': order['id'],
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': timestamp,
            'commission': order['quantity'] * fill_price * 0.001
        }        
    def generate_session_events(self):
        """Generate market session open/close events"""
        current = self.start_date
        
        while current <= self.end_date:
            if self.is_trading_day(current):
                # Market open event
                open_time = current.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                self.add_event(SimulationEvent(
                    timestamp=open_time,
                    event_type=EventType.SESSION_OPEN,
                    data={'session': 'regular'},
                    priority=2
                ))
                
                # Market close event
                close_time = current.replace(
                    hour=16, minute=0, second=0, microsecond=0
                )
                
                # Check for half day
                if current.date() in self.trading_calendar['half_days']:
                    close_time = current.replace(
                        hour=13, minute=0, second=0, microsecond=0
                    )
                    
                self.add_event(SimulationEvent(
                    timestamp=close_time,
                    event_type=EventType.SESSION_CLOSE,
                    data={'session': 'regular'},
                    priority=2
                ))
                
            current += timedelta(days=1)            
    def simulate_corporate_actions(self, symbol: str):
        """Simulate dividends, splits, and other corporate actions"""
        # Dividend simulation (quarterly)
        dividend_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='Q'  # Quarterly
        )
        
        for date in dividend_dates:
            if self.is_trading_day(date):
                self.add_event(SimulationEvent(
                    timestamp=date.replace(hour=9, minute=30),
                    event_type=EventType.DIVIDEND,
                    data={
                        'symbol': symbol,
                        'amount': np.random.uniform(0.10, 0.50),  # $0.10-$0.50
                        'ex_date': date,
                        'pay_date': date + timedelta(days=14)
                    },
                    priority=3
                ))
                
    async def get_next_event(self) -> Optional[SimulationEvent]:
        """Get next event from the queue"""
        if not self.event_queue:
            return None
            
        return heapq.heappop(self.event_queue)
        
    def advance_time(self, new_time: datetime):
        """Advance simulation time and trigger scheduled events"""
        if new_time <= self.current_time:
            return            
        # Check for daily events
        if new_time.date() > self.current_time.date():
            for event in self.scheduled_events['daily']:
                if new_time >= event['start_time']:
                    event['callback'](new_time)
                    
        # Check for weekly events (Monday)
        if new_time.weekday() == 0 and new_time.date() > self.current_time.date():
            for event in self.scheduled_events['weekly']:
                if new_time >= event['start_time']:
                    event['callback'](new_time)
                    
        # Check for monthly events (first trading day)
        if new_time.month != self.current_time.month:
            for event in self.scheduled_events['monthly']:
                if new_time >= event['start_time']:
                    event['callback'](new_time)
                    
        self.current_time = new_time
        
    def reset(self):
        """Reset simulator to initial state"""
        self.current_time = self.start_date
        self.event_queue.clear()
        self.generate_session_events()
        
    def get_simulation_progress(self) -> float:
        """Get simulation progress as percentage"""
        total_duration = (self.end_date - self.start_date).total_seconds()
        elapsed_duration = (self.current_time - self.start_date).total_seconds()
        
        if total_duration == 0:
            return 100.0
            
        return (elapsed_duration / total_duration) * 100