"""
Market Data Aggregator

Central component that collects, normalizes, and distributes market data
from multiple exchanges in real-time.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np

from ..core.architecture import injectable, inject, EventBus
from ..core.observability import get_logger, MetricsCollector
from ..core.cache import CacheManager
from ..core.messaging import MessageBroker
from ..exchange.exchange_interface import ExchangeInterface
from .normalizer import DataNormalizer, NormalizedMarketData
from .collector import ExchangeDataCollector
from .distributor import DataDistributor
from .quality import DataQualityChecker
from .storage import MarketDataStorage

logger = get_logger(__name__)

@dataclass
class AggregatorConfig:
    """Configuration for market data aggregator"""
    exchanges: List[str]
    symbols: List[str]
    data_types: List[str] = field(default_factory=lambda: ['ticker', 'orderbook', 'trades'])
    orderbook_depth: int = 20
    trade_buffer_size: int = 1000
    aggregation_interval: int = 100  # milliseconds
    enable_storage: bool = True
    enable_quality_checks: bool = True
    max_latency_ms: int = 1000
    snapshot_interval: int = 60  # seconds
    compression_enabled: bool = True@injectable
class MarketDataAggregator:
    """
    Aggregates market data from multiple exchanges and provides
    unified access to normalized data
    """
    
    @inject
    def __init__(
        self,
        event_bus: EventBus,
        metrics_collector: MetricsCollector,
        cache_manager: CacheManager,
        message_broker: MessageBroker
    ):
        self.event_bus = event_bus
        self.metrics = metrics_collector
        self.cache = cache_manager
        self.broker = message_broker
        self.logger = logger
        
        # Components
        self.normalizer = DataNormalizer()
        self.quality_checker = DataQualityChecker()
        self.distributor = DataDistributor(event_bus, message_broker)
        
        # State
        self.config: Optional[AggregatorConfig] = None
        self.collectors: Dict[str, ExchangeDataCollector] = {}
        self.storage: Optional[MarketDataStorage] = None
        self.is_running = False
        
        # Market data cache
        self.ticker_cache: Dict[str, Dict[str, NormalizedMarketData]] = defaultdict(dict)
        self.orderbook_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.trades_buffer: Dict[str, List[Any]] = defaultdict(list)
        
        # Aggregation state
        self.last_update_time: Dict[str, datetime] = {}
        self.update_counts: Dict[str, int] = defaultdict(int)
        self.latency_stats: Dict[str, List[float]] = defaultdict(list)        
    async def initialize(self, config: AggregatorConfig, exchanges: Dict[str, ExchangeInterface]):
        """Initialize the aggregator with configuration"""
        self.config = config
        
        # Initialize storage if enabled
        if config.enable_storage:
            self.storage = MarketDataStorage(
                compression_enabled=config.compression_enabled
            )
            await self.storage.initialize()
            
        # Create collectors for each exchange
        for exchange_name, exchange in exchanges.items():
            if exchange_name in config.exchanges:
                collector = ExchangeDataCollector(
                    exchange=exchange,
                    symbols=config.symbols,
                    data_types=config.data_types,
                    callback=self._handle_market_data
                )
                self.collectors[exchange_name] = collector
                
        # Initialize distributor
        await self.distributor.initialize()
        
        # Setup periodic tasks
        asyncio.create_task(self._aggregation_loop())
        asyncio.create_task(self._snapshot_loop())
        asyncio.create_task(self._metrics_loop())
        
        self.logger.info(
            f"Initialized market data aggregator for {len(self.collectors)} exchanges "
            f"and {len(config.symbols)} symbols"
        )        
    async def start(self):
        """Start collecting market data"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start all collectors
        tasks = []
        for exchange_name, collector in self.collectors.items():
            self.logger.info(f"Starting collector for {exchange_name}")
            tasks.append(collector.start())
            
        await asyncio.gather(*tasks)
        
        # Start distributor
        await self.distributor.start()
        
        self.logger.info("Market data aggregator started")
        
    async def stop(self):
        """Stop collecting market data"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop all collectors
        tasks = []
        for collector in self.collectors.values():
            tasks.append(collector.stop())
            
        await asyncio.gather(*tasks)
        
        # Stop distributor
        await self.distributor.stop()
        
        # Flush storage
        if self.storage:
            await self.storage.flush()
            
        self.logger.info("Market data aggregator stopped")        
    async def _handle_market_data(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        data: Any,
        timestamp: datetime
    ):
        """Handle incoming market data from collectors"""
        # Record latency
        latency = (datetime.now() - timestamp).total_seconds() * 1000
        self.latency_stats[exchange].append(latency)
        
        # Check latency threshold
        if latency > self.config.max_latency_ms:
            self.logger.warning(
                f"High latency detected: {exchange} {symbol} {data_type} - {latency:.0f}ms"
            )
            await self.metrics.increment("market_data.high_latency", {
                "exchange": exchange,
                "data_type": data_type
            })
            
        # Quality check if enabled
        if self.config.enable_quality_checks:
            is_valid, issues = await self.quality_checker.check(
                exchange, symbol, data_type, data
            )
            
            if not is_valid:
                self.logger.error(
                    f"Data quality issues: {exchange} {symbol} {data_type} - {issues}"
                )
                await self.metrics.increment("market_data.quality_issues", {
                    "exchange": exchange,
                    "issues": ",".join(issues)
                })
                return                
        # Normalize data
        normalized = await self.normalizer.normalize(
            exchange, symbol, data_type, data, timestamp
        )
        
        # Update cache based on data type
        if data_type == 'ticker':
            self.ticker_cache[symbol][exchange] = normalized
        elif data_type == 'orderbook':
            self.orderbook_cache[symbol][exchange] = normalized
        elif data_type == 'trades':
            self.trades_buffer[symbol].extend(normalized)
            # Trim buffer if too large
            if len(self.trades_buffer[symbol]) > self.config.trade_buffer_size:
                self.trades_buffer[symbol] = self.trades_buffer[symbol][-self.config.trade_buffer_size:]
                
        # Update tracking
        self.last_update_time[f"{exchange}:{symbol}"] = timestamp
        self.update_counts[f"{exchange}:{symbol}"] += 1
        
        # Store if enabled
        if self.storage:
            await self.storage.store(exchange, symbol, data_type, normalized, timestamp)
            
        # Distribute immediately for critical data
        if data_type in ['ticker', 'trades']:
            await self.distributor.distribute(symbol, data_type, normalized)        
    async def _aggregation_loop(self):
        """Periodic aggregation of market data"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.aggregation_interval / 1000)
                
                # Aggregate ticker data
                for symbol in self.config.symbols:
                    if symbol in self.ticker_cache and self.ticker_cache[symbol]:
                        aggregated = await self._aggregate_tickers(symbol)
                        await self.distributor.distribute(
                            symbol, 'aggregated_ticker', aggregated
                        )
                        
                # Aggregate orderbook data
                for symbol in self.config.symbols:
                    if symbol in self.orderbook_cache and self.orderbook_cache[symbol]:
                        aggregated = await self._aggregate_orderbooks(symbol)
                        await self.distributor.distribute(
                            symbol, 'aggregated_orderbook', aggregated
                        )
                        
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {str(e)}")
                await self.metrics.increment("market_data.aggregation_errors")                
    async def _aggregate_tickers(self, symbol: str) -> Dict[str, Any]:
        """Aggregate ticker data from multiple exchanges"""
        tickers = self.ticker_cache[symbol]
        
        if not tickers:
            return {}
            
        # Calculate aggregated metrics
        prices = [t.price for t in tickers.values() if t.price]
        volumes = [t.volume_24h for t in tickers.values() if t.volume_24h]
        
        aggregated = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'exchanges': list(tickers.keys()),
            'best_bid': max([t.bid for t in tickers.values() if t.bid], default=None),
            'best_ask': min([t.ask for t in tickers.values() if t.ask], default=None),
            'mid_price': np.mean(prices) if prices else None,
            'vwap': self._calculate_vwap(tickers),
            'total_volume': sum(volumes) if volumes else 0,
            'price_std': np.std(prices) if len(prices) > 1 else 0,
            'spread': self._calculate_spread(tickers),
            'arbitrage_opportunity': self._check_arbitrage(tickers)
        }
        
        return aggregated        
    async def _aggregate_orderbooks(self, symbol: str) -> Dict[str, Any]:
        """Aggregate orderbook data from multiple exchanges"""
        orderbooks = self.orderbook_cache[symbol]
        
        if not orderbooks:
            return {}
            
        # Combine all bids and asks
        all_bids = []
        all_asks = []
        
        for exchange, ob in orderbooks.items():
            if 'bids' in ob:
                for bid in ob['bids'][:self.config.orderbook_depth]:
                    all_bids.append({
                        'exchange': exchange,
                        'price': bid[0],
                        'quantity': bid[1]
                    })
                    
            if 'asks' in ob:
                for ask in ob['asks'][:self.config.orderbook_depth]:
                    all_asks.append({
                        'exchange': exchange,
                        'price': ask[0],
                        'quantity': ask[1]
                    })
                    
        # Sort and aggregate
        all_bids.sort(key=lambda x: x['price'], reverse=True)
        all_asks.sort(key=lambda x: x['price'])
        
        # Calculate aggregated orderbook
        aggregated_bids = self._aggregate_price_levels(all_bids)
        aggregated_asks = self._aggregate_price_levels(all_asks)        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'exchanges': list(orderbooks.keys()),
            'bids': aggregated_bids[:self.config.orderbook_depth],
            'asks': aggregated_asks[:self.config.orderbook_depth],
            'spread': aggregated_asks[0]['price'] - aggregated_bids[0]['price'] if aggregated_bids and aggregated_asks else None,
            'mid_price': (aggregated_asks[0]['price'] + aggregated_bids[0]['price']) / 2 if aggregated_bids and aggregated_asks else None,
            'total_bid_volume': sum(b['quantity'] for b in aggregated_bids),
            'total_ask_volume': sum(a['quantity'] for a in aggregated_asks),
            'imbalance': self._calculate_orderbook_imbalance(aggregated_bids, aggregated_asks)
        }
        
    def _aggregate_price_levels(self, orders: List[Dict]) -> List[Dict]:
        """Aggregate orders at same price level"""
        price_levels = defaultdict(lambda: {'quantity': 0, 'exchanges': set()})
        
        for order in orders:
            price = order['price']
            price_levels[price]['quantity'] += order['quantity']
            price_levels[price]['exchanges'].add(order['exchange'])
            
        aggregated = []
        for price, data in sorted(price_levels.items(), key=lambda x: x[0], reverse=(orders[0]['price'] > orders[-1]['price'] if orders else False)):
            aggregated.append({
                'price': price,
                'quantity': data['quantity'],
                'exchanges': list(data['exchanges'])
            })
            
        return aggregated        
    def _calculate_vwap(self, tickers: Dict[str, NormalizedMarketData]) -> Optional[float]:
        """Calculate volume-weighted average price"""
        total_volume = 0
        volume_price = 0
        
        for ticker in tickers.values():
            if ticker.price and ticker.volume_24h:
                volume_price += ticker.price * ticker.volume_24h
                total_volume += ticker.volume_24h
                
        return volume_price / total_volume if total_volume > 0 else None
        
    def _calculate_spread(self, tickers: Dict[str, NormalizedMarketData]) -> Dict[str, float]:
        """Calculate various spread metrics"""
        spreads = []
        
        for ticker in tickers.values():
            if ticker.bid and ticker.ask:
                spread = ticker.ask - ticker.bid
                spread_pct = (spread / ticker.mid_price) * 100 if ticker.mid_price else 0
                spreads.append({
                    'absolute': spread,
                    'percentage': spread_pct
                })
                
        if not spreads:
            return {'min': None, 'max': None, 'avg': None}
            
        return {
            'min': min(s['absolute'] for s in spreads),
            'max': max(s['absolute'] for s in spreads),
            'avg': np.mean([s['absolute'] for s in spreads]),
            'avg_pct': np.mean([s['percentage'] for s in spreads])
        }        
    def _check_arbitrage(self, tickers: Dict[str, NormalizedMarketData]) -> Optional[Dict[str, Any]]:
        """Check for arbitrage opportunities between exchanges"""
        if len(tickers) < 2:
            return None
            
        # Find best bid and ask across exchanges
        best_bid = None
        best_bid_exchange = None
        best_ask = None
        best_ask_exchange = None
        
        for exchange, ticker in tickers.items():
            if ticker.bid and (best_bid is None or ticker.bid > best_bid):
                best_bid = ticker.bid
                best_bid_exchange = exchange
                
            if ticker.ask and (best_ask is None or ticker.ask < best_ask):
                best_ask = ticker.ask
                best_ask_exchange = exchange
                
        # Check if arbitrage exists
        if best_bid and best_ask and best_bid > best_ask:
            profit = best_bid - best_ask
            profit_pct = (profit / best_ask) * 100
            
            return {
                'exists': True,
                'buy_exchange': best_ask_exchange,
                'sell_exchange': best_bid_exchange,
                'buy_price': best_ask,
                'sell_price': best_bid,
                'profit': profit,
                'profit_pct': profit_pct
            }
            
        return {'exists': False}        
    def _calculate_orderbook_imbalance(self, bids: List[Dict], asks: List[Dict]) -> float:
        """Calculate orderbook imbalance"""
        if not bids or not asks:
            return 0.0
            
        # Calculate volume at different depth levels
        bid_volume = sum(b['quantity'] for b in bids[:5])  # Top 5 levels
        ask_volume = sum(a['quantity'] for a in asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        # Imbalance ratio: positive = more bids, negative = more asks
        return (bid_volume - ask_volume) / total_volume
        
    async def _snapshot_loop(self):
        """Periodic snapshots of market state"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.snapshot_interval)
                
                # Create market snapshot
                snapshot = await self._create_market_snapshot()
                
                # Store snapshot
                if self.storage:
                    await self.storage.store_snapshot(snapshot)
                    
                # Distribute snapshot
                await self.distributor.distribute_snapshot(snapshot)
                
            except Exception as e:
                self.logger.error(f"Error in snapshot loop: {str(e)}")                
    async def _create_market_snapshot(self) -> Dict[str, Any]:
        """Create comprehensive market snapshot"""
        snapshot = {
            'timestamp': datetime.now(),
            'symbols': {},
            'statistics': {
                'total_updates': sum(self.update_counts.values()),
                'active_exchanges': len([e for e in self.collectors if self.collectors[e].is_connected]),
                'average_latency': {},
                'data_quality': {}
            }
        }
        
        # Add symbol data
        for symbol in self.config.symbols:
            symbol_data = {
                'ticker': self.ticker_cache.get(symbol, {}),
                'orderbook_summary': await self._get_orderbook_summary(symbol),
                'recent_trades': self.trades_buffer.get(symbol, [])[-100:],  # Last 100 trades
                'aggregated_ticker': await self._aggregate_tickers(symbol) if symbol in self.ticker_cache else None
            }
            snapshot['symbols'][symbol] = symbol_data
            
        # Add latency statistics
        for exchange, latencies in self.latency_stats.items():
            if latencies:
                snapshot['statistics']['average_latency'][exchange] = {
                    'avg': np.mean(latencies),
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                }
                
        return snapshot        
    async def _get_orderbook_summary(self, symbol: str) -> Dict[str, Any]:
        """Get orderbook summary for symbol"""
        orderbooks = self.orderbook_cache.get(symbol, {})
        
        if not orderbooks:
            return {}
            
        summary = {
            'exchanges': list(orderbooks.keys()),
            'best_bids': {},
            'best_asks': {},
            'spreads': {},
            'depths': {}
        }
        
        for exchange, ob in orderbooks.items():
            if 'bids' in ob and ob['bids']:
                summary['best_bids'][exchange] = ob['bids'][0][0]
                
            if 'asks' in ob and ob['asks']:
                summary['best_asks'][exchange] = ob['asks'][0][0]
                
            if 'bids' in ob and 'asks' in ob and ob['bids'] and ob['asks']:
                summary['spreads'][exchange] = ob['asks'][0][0] - ob['bids'][0][0]
                
            summary['depths'][exchange] = {
                'bid_depth': len(ob.get('bids', [])),
                'ask_depth': len(ob.get('asks', []))
            }
            
        return summary        
    async def _metrics_loop(self):
        """Periodic metrics reporting"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                # Report update rates
                for key, count in self.update_counts.items():
                    await self.metrics.gauge("market_data.update_rate", count / 60, {
                        "source": key
                    })
                    
                # Report latencies
                for exchange, latencies in self.latency_stats.items():
                    if latencies:
                        await self.metrics.histogram("market_data.latency", latencies, {
                            "exchange": exchange
                        })
                        
                # Clear stats for next period
                self.update_counts.clear()
                self.latency_stats.clear()
                
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {str(e)}")
                
    async def get_ticker(self, symbol: str, exchange: Optional[str] = None) -> Optional[Any]:
        """Get current ticker data"""
        if exchange:
            return self.ticker_cache.get(symbol, {}).get(exchange)
        else:
            # Return aggregated ticker
            return await self._aggregate_tickers(symbol)            
    async def get_orderbook(self, symbol: str, exchange: Optional[str] = None) -> Optional[Any]:
        """Get current orderbook data"""
        if exchange:
            return self.orderbook_cache.get(symbol, {}).get(exchange)
        else:
            # Return aggregated orderbook
            return await self._aggregate_orderbooks(symbol)
            
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Any]:
        """Get recent trades for symbol"""
        trades = self.trades_buffer.get(symbol, [])
        return trades[-limit:] if trades else []
        
    async def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics"""
        return {
            'active_symbols': len([s for s in self.ticker_cache if self.ticker_cache[s]]),
            'total_updates': sum(self.update_counts.values()),
            'connected_exchanges': len([e for e in self.collectors if self.collectors[e].is_connected]),
            'cache_sizes': {
                'tickers': sum(len(e) for e in self.ticker_cache.values()),
                'orderbooks': sum(len(e) for e in self.orderbook_cache.values()),
                'trades': sum(len(t) for t in self.trades_buffer.values())
            }
        }
        
    async def subscribe(self, symbol: str, data_types: List[str], callback: Callable):
        """Subscribe to market data updates"""
        await self.distributor.subscribe(symbol, data_types, callback)
        
    async def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from market data updates"""
        await self.distributor.unsubscribe(symbol, callback)