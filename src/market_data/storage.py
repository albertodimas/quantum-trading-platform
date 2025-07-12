"""
Market Data Storage Module

Handles persistence of market data with support for multiple storage backends.
Includes compression, partitioning, and efficient retrieval mechanisms.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import gzip
import pickle
from pathlib import Path

from ..core.interfaces import Injectable
from ..core.decorators import injectable
from ..core.logger import get_logger
from ..models.market_data import MarketData, OHLCV, Ticker, OrderBook, Trade

logger = get_logger(__name__)


class StorageBackend(Enum):
    """Available storage backends"""
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"
    TIMESERIES_DB = "timeseries_db"
    S3 = "s3"


@dataclass
class StorageConfig:
    """Storage configuration"""
    backend: StorageBackend = StorageBackend.MEMORY
    compression: bool = True
    partition_by: str = "day"  # hour, day, month
    retention_days: int = 365
    batch_size: int = 1000
    
    # Backend-specific settings
    file_path: Optional[Path] = None
    database_url: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None


class DataPartition:
    """Represents a data partition"""
    
    def __init__(self, partition_key: str, data: List[MarketData] = None):
        self.partition_key = partition_key
        self.data = data or []
        self.metadata = {
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "count": len(self.data)
        }
    
    def add(self, item: MarketData):
        """Add data to partition"""
        self.data.append(item)
        self.metadata["updated_at"] = datetime.utcnow()
        self.metadata["count"] = len(self.data)
    
    def compress(self) -> bytes:
        """Compress partition data"""
        return gzip.compress(pickle.dumps(self.data))
    
    @staticmethod
    def decompress(data: bytes) -> List[MarketData]:
        """Decompress partition data"""
        return pickle.loads(gzip.decompress(data))


@injectable
class MarketDataStorage(Injectable):
    """Stores and retrieves market data"""
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self._partitions: Dict[str, DataPartition] = {}
        self._write_buffer: List[MarketData] = []
        self._lock = asyncio.Lock()
        
        # Initialize backend
        self._init_backend()
        
        # Start background tasks
        self._flush_task = None
        self._cleanup_task = None
    
    def _init_backend(self):
        """Initialize storage backend"""
        if self.config.backend == StorageBackend.FILE:
            if self.config.file_path:
                self.config.file_path.mkdir(parents=True, exist_ok=True)
        elif self.config.backend == StorageBackend.DATABASE:
            # Initialize database connection
            pass
        elif self.config.backend == StorageBackend.S3:
            # Initialize S3 client
            pass
    
    async def start(self):
        """Start storage service"""
        self._flush_task = asyncio.create_task(self._flush_periodically())
        self._cleanup_task = asyncio.create_task(self._cleanup_periodically())
        logger.info(f"Market data storage started with {self.config.backend} backend")
    
    async def stop(self):
        """Stop storage service"""
        # Flush remaining data
        await self._flush_buffer()
        
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Market data storage stopped")
    
    async def store(self, data: Union[MarketData, List[MarketData]]):
        """Store market data"""
        if isinstance(data, MarketData):
            data = [data]
        
        async with self._lock:
            self._write_buffer.extend(data)
            
            # Flush if buffer is full
            if len(self._write_buffer) >= self.config.batch_size:
                await self._flush_buffer()
    
    async def retrieve(
        self,
        symbol: str = None,
        exchange: str = None,
        data_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = None
    ) -> AsyncIterator[MarketData]:
        """Retrieve market data"""
        # Determine partitions to read
        partitions = self._get_partitions_in_range(start_time, end_time)
        
        count = 0
        for partition_key in partitions:
            partition_data = await self._read_partition(partition_key)
            
            for item in partition_data:
                # Apply filters
                if symbol and item.symbol != symbol:
                    continue
                if exchange and item.exchange != exchange:
                    continue
                if data_type and item.data_type != data_type:
                    continue
                if start_time and item.timestamp < start_time:
                    continue
                if end_time and item.timestamp > end_time:
                    continue
                
                yield item
                
                count += 1
                if limit and count >= limit:
                    return
    
    async def get_latest(
        self,
        symbol: str,
        exchange: str = None,
        data_type: str = None
    ) -> Optional[MarketData]:
        """Get latest data for symbol"""
        # Check write buffer first
        async with self._lock:
            for item in reversed(self._write_buffer):
                if item.symbol == symbol:
                    if exchange and item.exchange != exchange:
                        continue
                    if data_type and item.data_type != data_type:
                        continue
                    return item
        
        # Check partitions
        latest_partition = self._get_latest_partition()
        if latest_partition:
            partition_data = await self._read_partition(latest_partition)
            for item in reversed(partition_data):
                if item.symbol == symbol:
                    if exchange and item.exchange != exchange:
                        continue
                    if data_type and item.data_type != data_type:
                        continue
                    return item
        
        return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "backend": self.config.backend.value,
            "partitions": len(self._partitions),
            "buffer_size": len(self._write_buffer),
            "total_items": sum(p.metadata["count"] for p in self._partitions.values())
        }
        
        if self.config.backend == StorageBackend.FILE:
            total_size = 0
            if self.config.file_path:
                for file in self.config.file_path.glob("*.gz"):
                    total_size += file.stat().st_size
            stats["disk_usage_mb"] = total_size / (1024 * 1024)
        
        return stats
    
    async def _flush_buffer(self):
        """Flush write buffer to storage"""
        if not self._write_buffer:
            return
        
        # Group by partition
        partitioned_data = {}
        for item in self._write_buffer:
            partition_key = self._get_partition_key(item.timestamp)
            if partition_key not in partitioned_data:
                partitioned_data[partition_key] = []
            partitioned_data[partition_key].append(item)
        
        # Write to partitions
        for partition_key, data in partitioned_data.items():
            await self._write_partition(partition_key, data)
        
        # Clear buffer
        self._write_buffer.clear()
        
        logger.debug(f"Flushed {sum(len(d) for d in partitioned_data.values())} items to storage")
    
    async def _flush_periodically(self):
        """Periodically flush buffer"""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                async with self._lock:
                    await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def _cleanup_periodically(self):
        """Periodically cleanup old data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        removed_count = 0
        
        for partition_key in list(self._partitions.keys()):
            partition_date = self._parse_partition_key(partition_key)
            if partition_date < cutoff_date:
                await self._remove_partition(partition_key)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old partitions")
    
    def _get_partition_key(self, timestamp: datetime) -> str:
        """Get partition key for timestamp"""
        if self.config.partition_by == "hour":
            return timestamp.strftime("%Y%m%d_%H")
        elif self.config.partition_by == "day":
            return timestamp.strftime("%Y%m%d")
        elif self.config.partition_by == "month":
            return timestamp.strftime("%Y%m")
        else:
            raise ValueError(f"Invalid partition_by: {self.config.partition_by}")
    
    def _parse_partition_key(self, key: str) -> datetime:
        """Parse partition key to datetime"""
        if self.config.partition_by == "hour":
            return datetime.strptime(key, "%Y%m%d_%H")
        elif self.config.partition_by == "day":
            return datetime.strptime(key, "%Y%m%d")
        elif self.config.partition_by == "month":
            return datetime.strptime(key, "%Y%m")
    
    def _get_partitions_in_range(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[str]:
        """Get partitions in time range"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=30)
        if not end_time:
            end_time = datetime.utcnow()
        
        partitions = []
        current = start_time
        
        while current <= end_time:
            partitions.append(self._get_partition_key(current))
            
            if self.config.partition_by == "hour":
                current += timedelta(hours=1)
            elif self.config.partition_by == "day":
                current += timedelta(days=1)
            elif self.config.partition_by == "month":
                # Add one month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        
        return partitions
    
    def _get_latest_partition(self) -> Optional[str]:
        """Get latest partition key"""
        if self._partitions:
            return max(self._partitions.keys())
        return None
    
    async def _write_partition(self, partition_key: str, data: List[MarketData]):
        """Write data to partition"""
        if self.config.backend == StorageBackend.MEMORY:
            if partition_key not in self._partitions:
                self._partitions[partition_key] = DataPartition(partition_key)
            for item in data:
                self._partitions[partition_key].add(item)
        
        elif self.config.backend == StorageBackend.FILE:
            partition = DataPartition(partition_key, data)
            compressed_data = partition.compress() if self.config.compression else pickle.dumps(data)
            
            file_path = self.config.file_path / f"{partition_key}.{'gz' if self.config.compression else 'pkl'}"
            file_path.write_bytes(compressed_data)
        
        # TODO: Implement other backends
    
    async def _read_partition(self, partition_key: str) -> List[MarketData]:
        """Read data from partition"""
        if self.config.backend == StorageBackend.MEMORY:
            if partition_key in self._partitions:
                return self._partitions[partition_key].data
            return []
        
        elif self.config.backend == StorageBackend.FILE:
            file_path = self.config.file_path / f"{partition_key}.{'gz' if self.config.compression else 'pkl'}"
            if file_path.exists():
                data = file_path.read_bytes()
                if self.config.compression:
                    return DataPartition.decompress(data)
                else:
                    return pickle.loads(data)
            return []
        
        # TODO: Implement other backends
        return []
    
    async def _remove_partition(self, partition_key: str):
        """Remove partition"""
        if self.config.backend == StorageBackend.MEMORY:
            self._partitions.pop(partition_key, None)
        
        elif self.config.backend == StorageBackend.FILE:
            file_path = self.config.file_path / f"{partition_key}.{'gz' if self.config.compression else 'pkl'}"
            if file_path.exists():
                file_path.unlink()