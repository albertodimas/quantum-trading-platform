"""
Data Provider for Backtesting

Provides historical market data from various sources (CSV, Database, APIs)
with efficient caching and streaming capabilities.
"""

import asyncio
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncIterator, Tuple, Any
from pathlib import Path
import aiofiles
import csv

from ..core.architecture import injectable, inject
from ..core.cache import CacheManager
from ..core.observability import get_logger
from ..exchange.binance_exchange import BinanceMarketData

logger = get_logger(__name__)

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.logger = logger
        
    @abstractmethod
    async def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Load historical data for a symbol"""
        pass        
    @abstractmethod
    async def get_latest_price(
        self, 
        symbol: str, 
        timestamp: datetime
    ) -> Optional[float]:
        """Get latest price for a symbol at given timestamp"""
        pass
        
    @abstractmethod
    async def iter_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, Any]]]:
        """Iterate through time-aligned bars for all symbols"""
        pass
        
    async def get_total_bars(self) -> int:
        """Get total number of bars to process"""
        if not self.data_cache:
            return 0
            
        # Get the symbol with most data points
        max_bars = 0
        for df in self.data_cache.values():
            max_bars = max(max_bars, len(df))
            
        return max_bars
        
    def _align_data(self) -> pd.DataFrame:
        """Align all symbol data to common timestamps"""
        if not self.data_cache:
            return pd.DataFrame()
            
        # Combine all dataframes
        combined = pd.DataFrame()
        for symbol, df in self.data_cache.items():
            df_copy = df.copy()
            # Rename columns to include symbol
            df_copy.columns = [f"{symbol}_{col}" for col in df_copy.columns]
            
            if combined.empty:
                combined = df_copy
            else:
                combined = combined.join(df_copy, how='outer')                
        # Forward fill missing data
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        return combined

@injectable
class HistoricalDataProvider(DataProvider):
    """
    Provides historical data from various sources with caching
    """
    
    @inject
    def __init__(self, cache_manager: CacheManager):
        super().__init__()
        self.cache = cache_manager
        self.aligned_data: Optional[pd.DataFrame] = None
        self.current_index = 0
        
    async def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Load and cache historical data"""
        # Check cache first
        cache_key = f"historical_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached_data = await self.cache.get(cache_key)
        
        if cached_data is not None:
            self.data_cache[symbol] = cached_data
            return cached_data            
        # Generate synthetic data for demonstration
        # In production, this would fetch from real data sources
        data = await self._generate_synthetic_data(
            symbol, start_date, end_date, timeframe
        )
        
        # Cache the data
        await self.cache.set(cache_key, data, ttl=86400)  # 24 hours
        
        self.data_cache[symbol] = data
        return data
        
    async def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        # Convert timeframe to frequency
        freq_map = {
            '1m': 'T', '5m': '5T', '15m': '15T',
            '1h': 'H', '4h': '4H', '1d': 'D'
        }
        freq = freq_map.get(timeframe, 'T')
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate synthetic price data
        num_bars = len(timestamps)
        initial_price = 100.0        
        # Random walk with drift
        returns = np.random.normal(0.0002, 0.02, num_bars)
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=timestamps)
        data['open'] = price_series * (1 + np.random.uniform(-0.002, 0.002, num_bars))
        data['high'] = data['open'] * (1 + np.random.uniform(0, 0.01, num_bars))
        data['low'] = data['open'] * (1 - np.random.uniform(0, 0.01, num_bars))
        data['close'] = price_series
        data['volume'] = np.random.uniform(1000, 10000, num_bars)
        
        return data
        
    async def get_latest_price(
        self, 
        symbol: str, 
        timestamp: datetime
    ) -> Optional[float]:
        """Get latest price at timestamp"""
        if symbol not in self.data_cache:
            return None
            
        df = self.data_cache[symbol]
        # Get the last price before or at timestamp
        valid_prices = df[df.index <= timestamp]
        
        if valid_prices.empty:
            return None
            
        return valid_prices.iloc[-1]['close']        
    async def iter_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, Any]]]:
        """Iterate through aligned bars"""
        if self.aligned_data is None:
            self.aligned_data = self._align_data()
            
        for timestamp, row in self.aligned_data.iterrows():
            market_data = {}
            
            # Extract data for each symbol
            for symbol in self.data_cache.keys():
                market_data[symbol] = {
                    'open': row.get(f'{symbol}_open'),
                    'high': row.get(f'{symbol}_high'),
                    'low': row.get(f'{symbol}_low'),
                    'close': row.get(f'{symbol}_close'),
                    'volume': row.get(f'{symbol}_volume'),
                }
                
            yield timestamp, market_data

@injectable            
class CSVDataProvider(DataProvider):
    """Load historical data from CSV files"""
    
    def __init__(self, data_directory: Path):
        super().__init__()
        self.data_directory = data_directory        
    async def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Load data from CSV file"""
        file_path = self.data_directory / f"{symbol}_{timeframe}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        # Read CSV asynchronously
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
            
        # Parse CSV
        from io import StringIO
        df = pd.read_csv(
            StringIO(content),
            index_col='timestamp',
            parse_dates=True
        )
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        self.data_cache[symbol] = df
        return df        
    async def get_latest_price(
        self, 
        symbol: str, 
        timestamp: datetime
    ) -> Optional[float]:
        """Get latest price from CSV data"""
        if symbol not in self.data_cache:
            return None
            
        df = self.data_cache[symbol]
        valid_prices = df[df.index <= timestamp]
        
        if valid_prices.empty:
            return None
            
        return valid_prices.iloc[-1]['close']
        
    async def iter_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, Any]]]:
        """Iterate through CSV data bars"""
        aligned_data = self._align_data()
        
        for timestamp, row in aligned_data.iterrows():
            market_data = {}
            
            for symbol in self.data_cache.keys():
                market_data[symbol] = {
                    'open': row.get(f'{symbol}_open'),
                    'high': row.get(f'{symbol}_high'),
                    'low': row.get(f'{symbol}_low'),
                    'close': row.get(f'{symbol}_close'),
                    'volume': row.get(f'{symbol}_volume'),
                }
                
            yield timestamp, market_data@injectable
class DatabaseDataProvider(DataProvider):
    """Load historical data from database"""
    
    @inject
    def __init__(self, db_connection):
        super().__init__()
        self.db = db_connection
        
    async def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Load data from database"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = %s
              AND timeframe = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp
        """
        
        # Execute query
        result = await self.db.fetch_all(
            query,
            symbol, timeframe, start_date, end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(result)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            
        self.data_cache[symbol] = df
        return df        
    async def get_latest_price(
        self, 
        symbol: str, 
        timestamp: datetime
    ) -> Optional[float]:
        """Get latest price from database"""
        query = """
            SELECT close
            FROM market_data
            WHERE symbol = %s
              AND timestamp <= %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        result = await self.db.fetch_one(query, symbol, timestamp)
        
        if result:
            return result['close']
            
        return None
        
    async def iter_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, Any]]]:
        """Iterate through database bars"""
        aligned_data = self._align_data()
        
        for timestamp, row in aligned_data.iterrows():
            market_data = {}
            
            for symbol in self.data_cache.keys():
                market_data[symbol] = {
                    'open': row.get(f'{symbol}_open'),
                    'high': row.get(f'{symbol}_high'),
                    'low': row.get(f'{symbol}_low'),
                    'close': row.get(f'{symbol}_close'),
                    'volume': row.get(f'{symbol}_volume'),
                }
                
            yield timestamp, market_data