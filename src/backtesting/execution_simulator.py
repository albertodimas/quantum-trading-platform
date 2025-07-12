"""
Execution Simulator for Backtesting

Simulates realistic order execution with slippage, market impact,
partial fills, and other market microstructure effects.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..core.observability import get_logger
from ..orders.order_manager import Order, OrderType, OrderStatus

logger = get_logger(__name__)

class FillModel(Enum):
    """Types of fill simulation models"""
    OPTIMISTIC = "optimistic"  # Always fill at limit price
    REALISTIC = "realistic"    # With slippage and spread
    PESSIMISTIC = "pessimistic"  # Worst-case execution
    RANDOM_WALK = "random_walk"  # Price moves during execution

@dataclass
class SimulatedFill:
    """Represents a simulated order fill"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    market_impact: float
    is_partial: bool = False
    remaining_quantity: float = 0.0    
class ExecutionSimulator:
    """
    Simulates realistic order execution with market microstructure effects
    """
    
    def __init__(
        self,
        slippage_model: str = "linear",
        use_spread: bool = True,
        spread_multiplier: float = 1.0,
        fill_model: FillModel = FillModel.REALISTIC,
        latency_ms: int = 10,
        rejection_rate: float = 0.01
    ):
        self.slippage_model = slippage_model
        self.use_spread = use_spread
        self.spread_multiplier = spread_multiplier
        self.fill_model = fill_model
        self.latency_ms = latency_ms
        self.rejection_rate = rejection_rate
        
        # Execution statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.partial_fills = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
        
        self.logger = logger        
    def calculate_spread(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate bid-ask spread"""
        if not self.use_spread:
            return 0.0
            
        # Use actual spread if available
        if 'bid' in market_data and 'ask' in market_data:
            return (market_data['ask'] - market_data['bid']) * self.spread_multiplier
            
        # Estimate spread based on price and volume
        price = market_data.get('close', 100)
        volume = market_data.get('volume', 1000)
        
        # Base spread as percentage of price
        base_spread_pct = 0.0005  # 0.05%
        
        # Adjust for volume (lower volume = wider spread)
        volume_factor = np.log10(max(volume, 1)) / 5  # Normalize around 100k volume
        volume_adjustment = max(0.5, min(2.0, 1 / volume_factor))
        
        spread = price * base_spread_pct * volume_adjustment * self.spread_multiplier
        return spread        
    def calculate_slippage(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate slippage based on order size and market conditions"""
        price = market_data.get('close', 100)
        volume = market_data.get('volume', 1000)
        order_value = order.quantity * price
        
        if self.slippage_model == "linear":
            # Linear impact model
            market_depth = volume * price * 0.1  # Assume 10% of volume is available
            impact_pct = min(order_value / market_depth, 0.01)  # Cap at 1%
            
        elif self.slippage_model == "square_root":
            # Square-root impact model (more realistic)
            participation_rate = order.quantity / max(volume, 1)
            impact_pct = 0.001 * np.sqrt(participation_rate * 100)
            
        elif self.slippage_model == "market_impact":
            # Sophisticated market impact model
            volatility = market_data.get('volatility', 0.02)  # Default 2% daily vol
            spread = self.calculate_spread(order.symbol, market_data)
            
            # Almgren-Chriss model approximation
            temporary_impact = spread / price + volatility * np.sqrt(order.quantity / volume)
            permanent_impact = 0.1 * temporary_impact
            
            impact_pct = temporary_impact + permanent_impact
        else:
            impact_pct = 0.0            
        # Direction of impact
        if order.side == 'buy':
            slippage = price * impact_pct
        else:
            slippage = -price * impact_pct
            
        return slippage
        
    def simulate_latency(self) -> timedelta:
        """Simulate network and processing latency"""
        # Add random jitter
        base_latency = self.latency_ms
        jitter = np.random.normal(0, self.latency_ms * 0.1)
        total_latency = max(1, base_latency + jitter)
        
        return timedelta(milliseconds=total_latency)
        
    def check_fill_conditions(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if order can be filled at current market conditions"""
        current_price = market_data.get('close')
        
        if not current_price:
            return False, "No market price available"            
        # Check order type conditions
        if order.type == OrderType.MARKET:
            return True, "Market order always fills"
            
        elif order.type == OrderType.LIMIT:
            if order.side == 'buy' and current_price <= order.price:
                return True, "Buy limit price met"
            elif order.side == 'sell' and current_price >= order.price:
                return True, "Sell limit price met"
            return False, "Limit price not met"
            
        elif order.type == OrderType.STOP:
            if order.side == 'buy' and current_price >= order.price:
                return True, "Buy stop triggered"
            elif order.side == 'sell' and current_price <= order.price:
                return True, "Sell stop triggered"
            return False, "Stop price not triggered"
            
        elif order.type == OrderType.STOP_LIMIT:
            # Check if stop is triggered
            if order.side == 'buy' and current_price >= order.stop_price:
                # Then check limit
                if current_price <= order.price:
                    return True, "Buy stop-limit filled"
            elif order.side == 'sell' and current_price <= order.stop_price:
                # Then check limit
                if current_price >= order.price:
                    return True, "Sell stop-limit filled"
            return False, "Stop-limit conditions not met"
            
        return False, f"Unknown order type: {order.type}"        
    async def simulate_fill(
        self,
        order: Order,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SimulatedFill]:
        """Simulate order fill with realistic execution"""
        self.total_orders += 1
        
        # Random rejection
        if np.random.random() < self.rejection_rate:
            self.rejected_orders += 1
            self.logger.warning(f"Order {order.id} randomly rejected")
            return None
            
        # Check fill conditions
        can_fill, reason = self.check_fill_conditions(order, market_data)
        if not can_fill:
            self.logger.debug(f"Order {order.id} not filled: {reason}")
            return None
            
        # Get market data
        symbol_data = market_data.get(order.symbol, {})
        if isinstance(symbol_data, dict):
            current_price = symbol_data.get('close')
            volume = symbol_data.get('volume', 1000)
        else:
            current_price = symbol_data
            volume = 1000            
        if not current_price:
            return None
            
        # Calculate fill price based on order type
        if order.type == OrderType.MARKET:
            base_price = current_price
        elif order.type == OrderType.LIMIT:
            base_price = order.price
        else:
            base_price = current_price  # For stops, fill at market
            
        # Apply spread
        spread = self.calculate_spread(order.symbol, symbol_data)
        if order.side == 'buy':
            base_price += spread / 2
        else:
            base_price -= spread / 2
            
        # Apply slippage
        slippage = self.calculate_slippage(order, symbol_data)
        fill_price = base_price + slippage
        
        # Simulate partial fills for large orders
        fill_quantity = order.quantity
        is_partial = False
        remaining = 0.0        
        # Check for partial fills
        available_liquidity = volume * 0.1  # Assume 10% of volume available
        if order.quantity > available_liquidity and self.fill_model == FillModel.REALISTIC:
            fill_quantity = available_liquidity
            is_partial = True
            remaining = order.quantity - fill_quantity
            self.partial_fills += 1
            
        # Calculate commission
        commission = abs(fill_quantity * fill_price * 0.001)  # 0.1% commission
        
        # Add execution latency
        fill_timestamp = timestamp + self.simulate_latency()
        
        # Update statistics
        self.filled_orders += 1
        self.total_slippage += abs(slippage * fill_quantity)
        self.total_commission += commission
        
        # Create fill
        fill = SimulatedFill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=fill_timestamp,
            commission=commission,
            slippage=slippage,
            market_impact=self.calculate_market_impact(order, symbol_data),
            is_partial=is_partial,
            remaining_quantity=remaining
        )        
        self.logger.info(
            f"Filled order {order.id}: {fill_quantity} @ {fill_price:.4f} "
            f"(slippage: {slippage:.4f})"
        )
        
        return fill
        
    def calculate_market_impact(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate permanent market impact of the order"""
        price = market_data.get('close', 100)
        volume = market_data.get('volume', 1000)
        
        # Simple permanent impact model
        participation = order.quantity / max(volume, 1)
        impact_bps = 10 * np.sqrt(participation)  # Basis points
        
        return price * impact_bps / 10000
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'partial_fills': self.partial_fills,
            'fill_rate': self.filled_orders / self.total_orders if self.total_orders > 0 else 0,
            'rejection_rate': self.rejected_orders / self.total_orders if self.total_orders > 0 else 0,
            'partial_fill_rate': self.partial_fills / self.filled_orders if self.filled_orders > 0 else 0,
            'total_slippage': self.total_slippage,
            'average_slippage': self.total_slippage / self.filled_orders if self.filled_orders > 0 else 0,
            'total_commission': self.total_commission,
            'average_commission': self.total_commission / self.filled_orders if self.filled_orders > 0 else 0,
        }        
    def reset_statistics(self):
        """Reset execution statistics"""
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.partial_fills = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0