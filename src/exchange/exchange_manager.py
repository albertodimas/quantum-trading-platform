"""
Exchange Manager

Manages multiple exchange connections with failover and load balancing.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from decimal import Decimal
import random

from .exchange_interface import (
    ExchangeInterface,
    OrderType, OrderSide, OrderStatus, TimeInForce,
    Order, Trade, Position, MarketData, OrderBook, Ticker, Balance
)
from .binance_exchange import BinanceExchange
from ..core.observability import get_logger
from ..core.architecture import injectable, inject
from ..core.factories import ExchangeFactory

logger = get_logger(__name__)


@injectable
class ExchangeManager:
    """Manages multiple exchange connections"""
    
    def __init__(self, exchange_factory: ExchangeFactory = inject()):
        self.exchange_factory = exchange_factory
        self.exchanges: Dict[str, ExchangeInterface] = {}
        self.primary_exchange: Optional[str] = None
        self.active_exchanges: Set[str] = set()
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized ExchangeManager")
    
    async def add_exchange(
        self, 
        name: str, 
        exchange_type: str,
        config: Dict[str, Any],
        set_as_primary: bool = False
    ) -> None:
        """Add new exchange connection"""
        try:
            # Create exchange instance
            if exchange_type.lower() == "binance":
                exchange = BinanceExchange(
                    api_key=config.get("api_key"),
                    api_secret=config.get("api_secret"),
                    testnet=config.get("testnet", True)
                )
            else:
                raise ValueError(f"Unsupported exchange type: {exchange_type}")
            
            # Connect to exchange
            await exchange.connect()
            
            # Add to managed exchanges
            self.exchanges[name] = exchange
            self.active_exchanges.add(name)
            
            # Set as primary if requested or if it's the first exchange
            if set_as_primary or not self.primary_exchange:
                self.primary_exchange = name
            
            logger.info(f"Added exchange: {name} (type={exchange_type})")
            
        except Exception as e:
            logger.error(f"Failed to add exchange {name}: {str(e)}")
            raise
    
    async def remove_exchange(self, name: str) -> None:
        """Remove exchange connection"""
        if name in self.exchanges:
            try:
                # Disconnect exchange
                await self.exchanges[name].disconnect()
                
                # Remove from collections
                del self.exchanges[name]
                self.active_exchanges.discard(name)
                
                # Update primary if needed
                if self.primary_exchange == name:
                    self.primary_exchange = next(iter(self.active_exchanges), None)
                
                logger.info(f"Removed exchange: {name}")
                
            except Exception as e:
                logger.error(f"Error removing exchange {name}: {str(e)}")
    
    def get_exchange(self, name: Optional[str] = None) -> ExchangeInterface:
        """Get exchange by name or primary exchange"""
        if name:
            if name not in self.exchanges:
                raise ValueError(f"Exchange not found: {name}")
            return self.exchanges[name]
        
        if not self.primary_exchange:
            raise RuntimeError("No primary exchange set")
        
        return self.exchanges[self.primary_exchange]
    
    def get_active_exchanges(self) -> List[str]:
        """Get list of active exchanges"""
        return list(self.active_exchanges)
    
    async def start_monitoring(self) -> None:
        """Start monitoring exchange health"""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_exchanges())
            logger.info("Started exchange monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring exchanges"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Stopped exchange monitoring")
    
    async def close(self) -> None:
        """Close all exchange connections"""
        # Stop monitoring
        await self.stop_monitoring()
        
        # Disconnect all exchanges
        for name in list(self.exchanges.keys()):
            await self.remove_exchange(name)
        
        logger.info("Closed ExchangeManager")
    
    # Aggregated methods across exchanges
    async def get_best_price(
        self, 
        symbol: str, 
        side: OrderSide,
        exchanges: Optional[List[str]] = None
    ) -> Dict[str, Decimal]:
        """Get best price across exchanges"""
        if not exchanges:
            exchanges = list(self.active_exchanges)
        
        prices = {}
        tasks = []
        
        for exchange_name in exchanges:
            if exchange_name in self.active_exchanges:
                exchange = self.exchanges[exchange_name]
                tasks.append(self._get_price_for_exchange(exchange_name, exchange, symbol, side))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for exchange_name, result in zip(exchanges, results):
            if not isinstance(result, Exception) and result is not None:
                prices[exchange_name] = result
        
        return prices
    
    async def get_aggregated_orderbook(
        self, 
        symbol: str, 
        depth: int = 20,
        exchanges: Optional[List[str]] = None
    ) -> Dict[str, OrderBook]:
        """Get orderbooks from multiple exchanges"""
        if not exchanges:
            exchanges = list(self.active_exchanges)
        
        orderbooks = {}
        tasks = []
        
        for exchange_name in exchanges:
            if exchange_name in self.active_exchanges:
                exchange = self.exchanges[exchange_name]
                tasks.append(self._get_orderbook_for_exchange(exchange_name, exchange, symbol, depth))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for exchange_name, result in zip(exchanges, results):
            if not isinstance(result, Exception) and result is not None:
                orderbooks[exchange_name] = result
        
        return orderbooks
    
    async def get_total_balance(
        self, 
        asset: Optional[str] = None,
        exchanges: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Balance]]:
        """Get balances across all exchanges"""
        if not exchanges:
            exchanges = list(self.active_exchanges)
        
        all_balances = {}
        tasks = []
        
        for exchange_name in exchanges:
            if exchange_name in self.active_exchanges:
                exchange = self.exchanges[exchange_name]
                tasks.append(self._get_balance_for_exchange(exchange_name, exchange, asset))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for exchange_name, result in zip(exchanges, results):
            if not isinstance(result, Exception) and result is not None:
                all_balances[exchange_name] = result
        
        return all_balances
    
    # Smart order routing
    async def route_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        routing_strategy: str = "best_price",
        **kwargs
    ) -> Dict[str, Order]:
        """Route order to best exchange based on strategy"""
        
        if routing_strategy == "best_price":
            # Find exchange with best price
            prices = await self.get_best_price(symbol, side)
            if not prices:
                raise RuntimeError("No exchanges available for routing")
            
            # Select best exchange
            if side == OrderSide.BUY:
                # For buy orders, choose lowest ask
                best_exchange = min(prices.items(), key=lambda x: x[1])[0]
            else:
                # For sell orders, choose highest bid
                best_exchange = max(prices.items(), key=lambda x: x[1])[0]
        
        elif routing_strategy == "random":
            # Random routing for load balancing
            best_exchange = random.choice(list(self.active_exchanges))
        
        elif routing_strategy == "primary":
            # Route to primary exchange
            best_exchange = self.primary_exchange
            if not best_exchange:
                raise RuntimeError("No primary exchange set")
        
        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")
        
        # Place order on selected exchange
        exchange = self.get_exchange(best_exchange)
        order = await exchange.create_order(
            symbol=symbol,
            side=side,
            type=type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            **kwargs
        )
        
        logger.info(
            f"Routed order to {best_exchange}: {symbol} {side.value} "
            f"{quantity} @ {price or 'market'}"
        )
        
        return {best_exchange: order}
    
    # Failover support
    async def execute_with_failover(
        self,
        operation: str,
        *args,
        exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """Execute operation with failover to backup exchanges"""
        if not exchanges:
            # Try primary first, then others
            exchanges = [self.primary_exchange] if self.primary_exchange else []
            exchanges.extend([e for e in self.active_exchanges if e != self.primary_exchange])
        
        last_error = None
        
        for exchange_name in exchanges:
            if exchange_name not in self.active_exchanges:
                continue
            
            try:
                exchange = self.exchanges[exchange_name]
                method = getattr(exchange, operation)
                result = await method(*args, **kwargs)
                
                logger.info(f"Successfully executed {operation} on {exchange_name}")
                return result
                
            except Exception as e:
                logger.warning(f"Failed to execute {operation} on {exchange_name}: {str(e)}")
                last_error = e
                continue
        
        # All exchanges failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"No exchanges available for operation: {operation}")
    
    # Private helper methods
    async def _get_price_for_exchange(
        self, 
        exchange_name: str, 
        exchange: ExchangeInterface,
        symbol: str,
        side: OrderSide
    ) -> Optional[Decimal]:
        """Get price from specific exchange"""
        try:
            ticker = await exchange.get_ticker(symbol)
            if side == OrderSide.BUY:
                return ticker.ask_price
            else:
                return ticker.bid_price
        except Exception as e:
            logger.error(f"Failed to get price from {exchange_name}: {str(e)}")
            return None
    
    async def _get_orderbook_for_exchange(
        self, 
        exchange_name: str, 
        exchange: ExchangeInterface,
        symbol: str,
        depth: int
    ) -> Optional[OrderBook]:
        """Get orderbook from specific exchange"""
        try:
            return await exchange.get_orderbook(symbol, depth)
        except Exception as e:
            logger.error(f"Failed to get orderbook from {exchange_name}: {str(e)}")
            return None
    
    async def _get_balance_for_exchange(
        self, 
        exchange_name: str, 
        exchange: ExchangeInterface,
        asset: Optional[str]
    ) -> Optional[Dict[str, Balance]]:
        """Get balance from specific exchange"""
        try:
            return await exchange.get_balance(asset)
        except Exception as e:
            logger.error(f"Failed to get balance from {exchange_name}: {str(e)}")
            return None
    
    async def _monitor_exchanges(self) -> None:
        """Monitor exchange health and update active status"""
        while True:
            try:
                for name, exchange in list(self.exchanges.items()):
                    try:
                        # Check if exchange is alive
                        is_alive = await exchange.is_alive()
                        
                        if is_alive and name not in self.active_exchanges:
                            # Exchange recovered
                            self.active_exchanges.add(name)
                            logger.info(f"Exchange {name} is now active")
                            
                        elif not is_alive and name in self.active_exchanges:
                            # Exchange failed
                            self.active_exchanges.remove(name)
                            logger.warning(f"Exchange {name} is now inactive")
                            
                            # Update primary if needed
                            if self.primary_exchange == name:
                                new_primary = next(iter(self.active_exchanges), None)
                                self.primary_exchange = new_primary
                                if new_primary:
                                    logger.info(f"Switched primary exchange to {new_primary}")
                    
                    except Exception as e:
                        logger.error(f"Error monitoring exchange {name}: {str(e)}")
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in exchange monitoring: {str(e)}")
                await asyncio.sleep(60)